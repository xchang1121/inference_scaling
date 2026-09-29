"""The autoregressive family: one AR model and the seven algorithms.

``solve`` renders the dataset prompt with the chat template, runs the chosen
algorithm inside the configured sampling scope and finishes the answer after a
thinking segment. Per-problem costs are backend counter deltas by phase:
``reward`` (the verifier's vote pool), ``search`` (the algorithm) and ``finish``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from inference_scaling.app.records import (
    Meter,
    adapter_hashes,
    checkpoint_metadata_hashes,
    importance_trace,
    weight_hashes,
)
from inference_scaling.app.rewards import Reward, best_of_n, memoized, verifier_reward
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig, MHConfig
from inference_scaling.arllm.algorithms.joint_budget_is import JointBudgetISConfig, run_joint_budget_is
from inference_scaling.arllm.algorithms.mh import MHChainResult, run_mh_chain
from inference_scaling.arllm.backends.batching import ContinuousBatchingBackend
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, SelfCertaintyReward, TokenStatisticReward
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.datasets.base import Dataset, Problem
from inference_scaling.shared.model.generation import generation_budget
from inference_scaling.shared.model.loading import (
    release_accelerator_memory,
    resolve_checkpoint_path,
    synchronize_accelerator,
)
from inference_scaling.shared.model.prompting import render_prompt
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence

# Algorithms that sample inside the thinking scope when it is configured.
SCOPED = frozenset({"mh", "mh_power", "is"})
# Algorithms whose target reweights the full-support base policy.
FULL_SUPPORT = frozenset({"mh", "mh_power", "is"})
# Rewards that read how decided the model's next-token distributions are.
CONFIDENCE_REWARDS = frozenset({"self_certainty", "consilience"})


@dataclass(frozen=True)
class _Task:
    problem: Problem
    prompt: TokenSequence
    maximum: int
    sampling: SamplingConfig
    seed: int
    seeds: SeedStream
    # The algorithm's backend, wrapped by the sampling scope and a model reward's statistic recorder.
    backend: Any


def _kept_reward(step: Any) -> float:
    return float(step.selected.rollouts[step.completion_index].reward)


class ARFamily:
    name = "ar"

    def __init__(self, settings: Mapping[str, Any], choices: Any, dataset: Dataset) -> None:
        self.ar = settings["ar"]
        self.choices = choices
        self.dataset = dataset
        self.seed = int(settings["run"]["seed"])
        self.cache_dir = Path(str(settings["run"]["hash_cache_dir"]))
        self.reward_settings: Any = None if choices.reward is None else settings["rewards"][choices.reward]
        self.config = self.ar["algorithms"][choices.algorithm]
        self.max_new_tokens = int(dataset.settings["max_new_tokens"])
        self.workers = int(self.ar["engine"]["continuous_batching"]["workers"])
        if self.workers < 1:
            raise ValueError("ar.engine.continuous_batching.workers must be positive")
        sampling = self.ar["sampling"]
        if choices.algorithm in FULL_SUPPORT and (float(sampling["top_p"]) != 1.0 or sampling["top_k"] is not None):
            raise ValueError(f"{choices.algorithm} reweights the full-support base policy; set top_p = 1 and top_k = null")
        engine = self.ar["engine"]
        for option in ("suffix_replay", "early_rejection"):
            if engine["backend"] == "vllm" and self.config.get(option):
                raise ValueError(f"ar.algorithms.{choices.algorithm}.{option} needs the transformers engine, which "
                                 "samples from request uniform streams and reports reference log-probabilities")
        if choices.reward == "self_certainty" and engine["backend"] == "vllm" and engine["vllm"]["exact_scoring"] == "none":
            raise ValueError("self_certainty reads the whole next-token distribution, which vLLM does not return; "
                             "set ar.engine.backend or ar.engine.vllm.exact_scoring to transformers")
        self.raw: Any = None
        self.backend: Any = None

    # Identity -------------------------------------------------------------------------

    def _resolve(self, path: str, revision: str | None) -> Path:
        model = self.ar["model"]
        return resolve_checkpoint_path(path, revision=revision, cache_dir=model["cache_dir"],
                                       local_files_only=model["local_files_only"])

    def artifacts(self) -> dict[str, Any]:
        """Resolved checkpoint files and their hashes; the configured weight hash is enforced."""

        model = self.ar["model"]
        directory = self._resolve(str(model["path"]), model["revision"])
        identity: dict[str, Any] = {
            "path": str(model["path"]), "resolved": str(directory),
            "weight_sha256": weight_hashes(directory, model["weight_sha256"], cache_dir=self.cache_dir),
            "metadata_sha256": checkpoint_metadata_hashes(directory)}
        if model["adapter"] is not None:
            adapter = self._resolve(str(model["adapter"]["path"]), model["adapter"]["revision"])
            identity["adapter"] = {"path": str(model["adapter"]["path"]), "resolved": str(adapter),
                                   "sha256": adapter_hashes(adapter)}
        if model["tokenizer"] is not None:
            tokenizer = self._resolve(str(model["tokenizer"]), model["tokenizer_revision"])
            identity["tokenizer"] = {"path": str(model["tokenizer"]), "metadata_sha256": checkpoint_metadata_hashes(tokenizer)}
        return {"base": identity}

    # Lifecycle ------------------------------------------------------------------------

    def load(self) -> None:
        engine = self.ar["engine"]
        logprobs = 2 * int(self.config["num_beams"]) if self.choices.algorithm == "beam" else 0
        self.raw = load_backend(self.ar["model"], engine, seed=self.seed, logprobs=logprobs)
        self.backend = self.raw
        # AsyncLLM schedules concurrent requests itself.
        if self.workers > 1 and not getattr(self.raw, "supports_native_continuous_batching", False):
            batching = engine["continuous_batching"]
            self.backend = ContinuousBatchingBackend(
                self.raw, max_batch_size=int(batching["max_batch_size"]),
                max_batch_tokens=int(batching["max_batch_tokens"]),
                batch_wait_seconds=float(batching["batch_wait_seconds"]),
            )
        self.eos = self.raw.tokenizer.eos_token_id
        self.describer = SamplingScope.from_config(self.raw, self.ar, active=False)
        self.thinking_format = thinking_format_from_backend(self.raw, output_settings_from_config(self.ar))

    def synchronize(self) -> None:
        synchronize_accelerator(self.ar["engine"]["device"])

    def close(self) -> None:
        if self.backend is not None and self.backend is not self.raw:
            self.backend.close()
        close_backend(self.raw)
        self.raw = self.backend = None
        release_accelerator_memory()

    # One problem ----------------------------------------------------------------------

    def answer_text(self, prompt: TokenSequence, tokens: TokenSequence) -> str:
        """The graded text: the content after a complete thinking segment.

        With thinking enabled, an unfinished thought has no final answer.
        """

        info = self.describer.describe_output(self.raw, prompt, tokens)
        if self.ar["output"]["thinking_mode"] == "enabled" and info["thinking_status"] != "complete":
            return ""
        return str(info["content_text"])

    def _scope(self, prompt: TokenSequence) -> SamplingScope:
        algorithm, reward = self.choices.algorithm, self.choices.reward
        scope = SamplingScope.from_config(self.raw, self.ar, active=algorithm in SCOPED).for_prompt(prompt)
        # Text rewards and full-scope confidence rewards read the whole sequence.
        if scope.scope == "thinking" and (
            reward == "verifier" or (reward in CONFIDENCE_REWARDS and self.reward_settings["scope"] == "full")
        ):
            scope = scope.full_fallback("reward_uses_full_sequence")
        return scope

    def _reward(self, task: _Task) -> Reward | None:
        kind, settings = self.choices.reward, self.reward_settings
        if kind is None:
            return None
        # Model rewards read the statistic that the task's backend records.
        model: Any
        if kind in CONFIDENCE_REWARDS:
            statistic = task.backend.statistic
            span = {"thinking_format": self.thinking_format if settings["scope"] == "thinking" else None,
                    "scope": settings["scope"]}
            model = SelfCertaintyReward(task.backend, statistic, **span) if kind == "self_certainty" else ConsilienceReward(
                task.backend, statistic, window_fraction=float(settings["window_fraction"]),
                window_tokens=settings["window_tokens"], skip_fraction=float(settings["skip_fraction"]),
                initial_penalty=float(settings["initial_penalty"]), **span)
            # A generated sequence needs a scoring forward only when generation cannot compute the statistic.
            passes = 0 if self.raw.records(statistic, task.sampling) else 1
            return Reward(float(settings["temperature"]), memoized(model.batch), passes, model.describe(), model)

        def sample_pool(size: int) -> list[str]:
            # Pool seeds do not depend on the algorithm, so algorithms share one pool per draw.
            requests = [GenerationRequest(task.prompt, task.maximum, task.sampling,
                                          task.seeds.derive("vote-pool", task.problem.id, index),
                                          f"vote-pool:{task.problem.id}:{index}") for index in range(size)]
            return [self.answer_text(task.prompt, sample.token_ids) for sample in self.backend.sample_batch(requests)]

        # Best-of-N votes among its own candidates.
        return verifier_reward(settings, dataset=self.dataset, problem=task.problem, answer_text=self.answer_text,
                               sample_pool=None if self.choices.algorithm == "best_of_n" else sample_pool)

    def solve(self, problem: Problem, seeds: SeedStream) -> dict[str, Any]:
        algorithm = self.choices.algorithm
        prompt_text = self.dataset.prompt(problem)
        system = self.ar["prompt"]["system"]
        messages = ([{"role": "system", "content": system}] if system is not None else []) + [
            {"role": "user", "content": prompt_text}]
        rendered = render_prompt(self.raw.tokenizer, messages, self.ar)
        prompt = tuple(self.raw.encode(rendered, add_special_tokens=False))
        budget = generation_budget(self.max_new_tokens, len(prompt), [self.raw],
                                   context_window=self.ar["engine"]["context_window"])
        maximum = int(budget["effective_max_new_tokens"])
        sampling = self.ar["sampling"]
        policy = SamplingConfig(temperature=float(sampling["temperature"]), top_p=float(sampling["top_p"]),
                                top_k=sampling["top_k"], eos_token_id=self.eos)
        scope = self._scope(prompt)
        backend = scope.wrap(self.backend, prompt)
        if self.choices.reward in CONFIDENCE_REWARDS:
            # A confidence reward reads a token statistic of the generated tokens, which generation records.
            settings = self.reward_settings
            backend = StatisticRecorder(backend, TokenStatistic(
                SamplingConfig(temperature=float(settings["score_temperature"])),
                int(settings["top_k"]) if self.choices.reward == "consilience" else None))
        task = _Task(problem, prompt, maximum, policy, seeds.derive(algorithm, problem.id), seeds, backend)
        # Concurrent problems share the backend counters, so only sequential runs have per-problem costs.
        meter = Meter({"base": self.raw} if self.workers == 1 else {})
        with meter.phase("reward"):
            reward = self._reward(task)
        with meter.phase("search"):
            tokens, trace, value = getattr(self, "_" + algorithm)(task, reward, meter)
        with meter.phase("finish"):
            tokens, info = scope.finish(self.backend, prompt, tokens, max_new_tokens=maximum, sampling=policy,
                                        seed=seeds.derive(algorithm, problem.id, "final-content"))
        fallbacks = [info["sampling_fallback_reason"]] if info["sampling_fallback_reason"] else []
        confidence = getattr(reward, "model", None)
        if isinstance(confidence, TokenStatisticReward) and (reason := confidence.fallback(prompt, tokens)):
            fallbacks.append(f"{self.choices.reward}:{reason}")
        trace = {"generation_budget": budget, **trace}
        if reward is not None:
            trace["reward"] = dict(reward.description)
        return {
            "prompt_tokens": len(prompt),
            "output": {
                "text": self.raw.decode(tokens),
                "thinking": info["thinking_text"],
                "content": info["content_text"],
                "thinking_status": info["thinking_status"],
                "sampling_scope": info["sampling_scope"],
                "tokens": len(tokens),
                "ended_by_eos": info["ended_by_eos"],
                "length_exhausted": info["generation_budget_exhausted"],
            },
            "answer_text": self.answer_text(prompt, tokens),
            "reward": value,
            "trace": trace,
            "cost": meter.cost(
                lambda delta: delta["generation_forward_token_slots"] + delta["score_forward_token_slots"],
                lambda _role, delta: delta["estimated_dense_forward_flops"],
            ),
            "fallbacks": fallbacks,
        }

    # Algorithms: (task, reward, meter) -> (tokens, trace, reward of the output) --------

    def _sample(self, task: _Task, reward: Reward | None, meter: Meter):
        sample = task.backend.sample_batch([GenerationRequest(task.prompt, task.maximum, task.sampling, task.seed,
                                                              f"sample:{task.problem.id}")])[0]
        return sample.token_ids, {}, None

    def _direct(self, task: _Task, meter: Meter, beams: int):
        tokens = tuple(task.backend.direct_generate(task.prompt, max_new_tokens=task.maximum, num_beams=beams))
        # Native generate() may pad finished beams after EOS.
        tokens = tokens[: tokens.index(self.eos) + 1] if self.eos in tokens else tokens
        # Native generate() bypasses the backend counters: charge every beam every step.
        slots = beams * (len(task.prompt) + max(0, len(tokens) - 1))
        meter.add({"generation_forward_token_slots": slots,
                   "estimated_dense_forward_flops": 2 * self.raw.parameter_count * slots})
        return tokens, {"num_beams": beams, "estimated_forward_token_slots": slots}, None

    def _greedy(self, task: _Task, reward: Reward | None, meter: Meter):
        return self._direct(task, meter, 1)

    def _beam(self, task: _Task, reward: Reward | None, meter: Meter):
        return self._direct(task, meter, int(self.config["num_beams"]))

    def _best_of_n(self, task: _Task, reward: Reward | None, meter: Meter):
        requests = [GenerationRequest(task.prompt, task.maximum, task.sampling,
                                      task.seeds.derive("best_of_n", task.problem.id, index),
                                      f"best-of-n:{task.problem.id}:{index}")
                    for index in range(int(self.config["samples"]))]
        assert reward is not None
        return best_of_n(reward, self.dataset, task.problem, task.prompt,
                         [sample.token_ids for sample in task.backend.sample_batch(requests)], self.answer_text,
                         random.Random(task.seeds.derive("best_of_n", task.problem.id, "tie-break")))

    def _mh_config(self, task: _Task, *, alpha: float, reward_temperature: float | None,
                   early_rejection: bool) -> MHConfig:
        config = self.config
        return MHConfig(alpha=alpha, reward_temperature=reward_temperature, total_length=task.maximum,
                        block_size=min(int(config["block_size"]), task.maximum),
                        steps_per_block=int(config["steps_per_block"]), suffix_schedule=str(config["suffix_schedule"]),
                        iterations=config["iterations"], suffix_replay=bool(config["suffix_replay"]),
                        early_rejection=early_rejection)

    @staticmethod
    def _mh_trace(result: MHChainResult) -> dict[str, Any]:
        return {"updates": result.attempts, "skipped_updates": result.skipped, "accepted": result.accepted,
                "acceptance_rate": result.acceptance_rate, "mean_proposed_suffix_length": result.mean_proposed_suffix_length,
                "mean_accepted_token_changes": result.mean_accepted_token_changes,
                "replayed_tokens": result.replayed_tokens, "early_rejected": result.early_rejected,
                "proposal_sources": result.proposal_sources}

    def _mh_power(self, task: _Task, reward: Reward | None, meter: Meter):
        # The base policy is the task's sampling policy; the proposal temperature is relative to it.
        config = self.config
        result = run_mh_chain(
            task.backend, task.prompt,
            self._mh_config(task, alpha=float(config["alpha"]), reward_temperature=None,
                            early_rejection=bool(config["early_rejection"])),
            SeedStream(task.seed), base=task.sampling,
            proposal=replace(task.sampling, temperature=task.sampling.temperature * float(config["proposal_temperature"])),
        )
        return result.token_ids, self._mh_trace(result), None

    def _mh(self, task: _Task, reward: Reward | None, meter: Meter):
        assert reward is not None
        config = self.config
        history = config["frozen_history"]
        frozen = config["proposal"] == "frozen_history"
        samples = task.backend.sample_batch([
            GenerationRequest(task.prompt, task.maximum, task.sampling,
                              task.seeds.derive("reward_mh", task.problem.id, "history", index),
                              f"reward-mh-history:{task.problem.id}:{index}")
            for index in range(int(history["samples"]))
        ]) if frozen else []
        result = run_mh_chain(
            task.backend, task.prompt,
            self._mh_config(task, alpha=1.0, reward_temperature=reward.temperature, early_rejection=False),
            SeedStream(task.seed), base=task.sampling, proposal=task.sampling, reward=reward.batch,
            history=[(sample.token_ids, sample.token_logprobs, sample.token_cdf_bounds) for sample in samples],
            history_mixture=float(history["mixture"]) if frozen else 0.0,
        )
        return result.token_ids, self._mh_trace(result), float(result.reward)

    def _is(self, task: _Task, reward: Reward | None, meter: Meter):
        assert reward is not None
        config = self.config
        if config["planning"] == "fixed":
            fixed = config["fixed"]
            result = run_conditional_is(
                task.backend, task.prompt,
                ConditionalISConfig(candidate_count=int(fixed["candidate_count"]),
                                    rollout_count=int(fixed["rollout_count"]),
                                    block_size=min(int(fixed["block_size"]), task.maximum),
                                    total_length=task.maximum, reward_temperature=reward.temperature,
                                    block_first=bool(config["block_first"])),
                reward.batch, SeedStream(task.seed), sampling=task.sampling,
            )
            return result.token_ids, importance_trace(list(result.steps)), _kept_reward(result.steps[-1])
        joint = config["joint"]
        adaptive = config["chunk_adaptive"] if config["planning"] == "chunk_adaptive" else {}
        settings = JointBudgetISConfig(
            forward_token_budget=int(joint["forward_token_budget"]), total_length=task.maximum,
            block_sizes=tuple(joint["block_sizes"]), candidate_counts=tuple(joint["candidate_counts"]),
            rollout_counts=tuple(joint["rollout_counts"]), pilot_candidates=int(joint["pilot_candidates"]),
            pilot_rollouts=int(joint["pilot_rollouts"]), pilot_fraction=float(joint["pilot_fraction"]),
            reward_temperature=reward.temperature, reward_forward_passes=reward.forward_passes,
            relative_variance_floor=float(joint["relative_variance_floor"]),
            expected_output_tokens=joint["expected_output_tokens"], planning_mode=str(config["planning"]),
            block_first=bool(config["block_first"]), **adaptive,
        )
        joint_result = run_joint_budget_is(task.backend, task.prompt, settings, reward.batch,
                                           SeedStream(task.seed), sampling=task.sampling)
        trace = importance_trace([step.evaluation for step in joint_result.steps])
        for summary, step in zip(trace["steps"], joint_result.steps, strict=True):
            summary.update(plan=asdict(step.plan), pilot_forward_tokens=step.pilot_actual_cost,
                           forward_tokens=step.actual_cost, expected_remaining_tokens=step.expected_remaining,
                           adjustment=step.adjustment)
        trace.update(stopping_reason=joint_result.stopping_reason,
                     reserved_forward_tokens=joint_result.reserved_forward_tokens,
                     planned_forward_tokens_used=joint_result.actual_forward_tokens,
                     length_probe_forward_tokens=joint_result.length_probe_forward_tokens)
        return joint_result.token_ids, trace, _kept_reward(joint_result.steps[-1].evaluation)


__all__ = ["ARFamily"]

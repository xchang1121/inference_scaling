"""The diffusion family: one LLaDA model and the seven algorithms.

``sampling`` is the decoding policy of plain samples and of IS candidates and
completions; ``exact_sampling`` (random remasking) has tractable trajectory
probabilities and drives block beam search, trajectory power MH and the frozen
history of reward MH. Rewards are
text rewards only: a diffusion model has no autoregressive log-probability of
its output.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from inference_scaling.app.records import (
    Meter,
    adapter_hashes,
    cached_file_sha256,
    checkpoint_metadata_hashes,
    importance_trace,
)
from inference_scaling.app.rewards import Reward, best_index, text_reward
from inference_scaling.datasets.base import Dataset, Problem
from inference_scaling.dllm.algorithms.config import (
    DiffusionBlockBeamConfig,
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionPowerMHConfig,
)
from inference_scaling.dllm.algorithms.is_sampling import run_conditional_diffusion_is
from inference_scaling.dllm.algorithms.mh import run_diffusion_reward_mh
from inference_scaling.dllm.algorithms.mh_acceleration import run_diffusion_replay_mixture_mh
from inference_scaling.dllm.algorithms.search import run_diffusion_block_beam, run_diffusion_trajectory_power_mh
from inference_scaling.dllm.backends.loader import load_llada_backend
from inference_scaling.dllm.config import DiffusionSamplingConfig, sampling_from_settings
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.model.loading import release_accelerator_memory, synchronize_accelerator
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


def pinned_weight_hashes(model: Mapping[str, Any], cache_dir: Path) -> dict[str, str]:
    """Check every pinned weight file of a ``dllm.model`` section by size and SHA-256."""

    directory = Path(str(model["path"]))
    names, sizes, hashes = model["weight_files"], model["weight_bytes"], model["weight_sha256"]
    if not len(names) == len(sizes) == len(hashes):
        raise ValueError("weight_files, weight_bytes and weight_sha256 differ in length")
    for name, size in zip(names, sizes, strict=True):
        if not (directory / name).is_file() or (directory / name).stat().st_size != size:
            raise FileNotFoundError(f"{directory / name} is absent or has the wrong size")
    return {name: cached_file_sha256(directory / name, cache_dir=cache_dir, expected=str(digest))
            for name, digest in zip(names, hashes, strict=True)}


class DLLMFamily:
    name = "dllm"
    workers = 1

    def __init__(self, settings: Mapping[str, Any], choices: Any, dataset: Dataset) -> None:
        if choices.reward in {"logprob", "consilience"}:
            raise ValueError(f"the {choices.reward} reward reads autoregressive token probabilities; "
                             "use --model ar, or --reward verifier or vote")
        self.dllm = settings["dllm"]
        self.choices = choices
        self.dataset = dataset
        self.cache_dir = Path(str(settings["run"]["hash_cache_dir"]))
        self.reward_settings: Any = None if choices.reward is None else settings["rewards"][choices.reward]
        self.config = self.dllm["algorithms"][choices.algorithm]
        self.sampling = sampling_from_settings(self.dllm["sampling"])
        self.exact = sampling_from_settings(self.dllm["exact_sampling"])
        # The dataset limit is sized for autoregressive reasoning models; dLLM canvases have their own cap.
        maximum = min(int(dataset.settings["max_new_tokens"]), int(self.dllm["max_new_tokens"]))
        # Diffusion decodes whole blocks: the output length is the largest multiple that fits.
        self.length = maximum - maximum % self.sampling.block_length
        if self.length <= 0:
            raise ValueError("max_new_tokens is shorter than one diffusion block")
        self.backend: Any = None

    def artifacts(self) -> dict[str, Any]:
        """Pinned weight files, checked by size and SHA-256."""

        model = self.dllm["model"]
        directory = Path(str(model["path"]))
        identity: dict[str, Any] = {
            "path": str(directory),
            "weight_sha256": pinned_weight_hashes(model, self.cache_dir),
            "metadata_sha256": checkpoint_metadata_hashes(directory),
        }
        if model["adapter"] is not None:
            identity["adapter"] = {"path": str(model["adapter"]["path"]),
                                   "sha256": adapter_hashes(Path(str(model["adapter"]["path"])))}
        return {"base": identity}

    def load(self) -> None:
        self.backend = load_llada_backend(self.dllm["model"], self.dllm["engine"])

    def synchronize(self) -> None:
        synchronize_accelerator(self.dllm["engine"]["device"])

    def close(self) -> None:
        self.backend = None
        release_accelerator_memory()

    # One problem ----------------------------------------------------------------------

    def answer_text(self, _prompt: TokenSequence, tokens: TokenSequence) -> str:
        return self.backend.decode(tokens)

    def _samples(self, prompt: TokenSequence, sampling: DiffusionSamplingConfig, seeds: list[int], label: str):
        return self.backend.sample_batch([
            DiffusionGenerationRequest(prefix=prompt, generation_length=self.length, sampling=sampling, seed=seed,
                                       request_id=f"{label}:{index}", stop_at_eos=True)
            for index, seed in enumerate(seeds)
        ])

    def _reward(self, problem: Problem, prompt_text: str, prompt: TokenSequence, seeds: SeedStream) -> Reward | None:
        kind = self.choices.reward
        if kind is None or (kind == "vote" and self.choices.algorithm == "best_of_n"):
            return None
        pool: list[str] = []
        if kind == "vote":
            samples = self._samples(prompt, self.sampling, [
                seeds.derive("vote-pool", problem.id, index) for index in range(int(self.reward_settings["pool_size"]))
            ], f"vote-pool:{problem.id}")
            pool = [self.answer_text(prompt, sample.token_ids) for sample in samples]
        return text_reward(kind, self.reward_settings, dataset=self.dataset, problem=problem, prompt_text=prompt_text,
                           answer_text=self.answer_text, pool=pool)

    def solve(self, problem: Problem, seeds: SeedStream) -> dict[str, Any]:
        algorithm = self.choices.algorithm
        prompt_text = self.dataset.prompt(problem)
        prompt = self.backend.encode_chat(prompt_text, system_text=self.dllm["prompt"]["system"])
        seed = seeds.derive(algorithm, problem.id)
        meter = Meter({"base": self.backend})
        with meter.phase("reward"):
            reward = self._reward(problem, prompt_text, prompt, seeds)
        with meter.phase("search"):
            tokens, trace, value = getattr(self, "_" + algorithm)(problem, prompt, seed, seeds, reward)
        text = self.answer_text(prompt, tokens)
        eos = getattr(self.backend.tokenizer, "eos_token_id", None)
        ended = eos is not None and eos in tokens
        if reward is not None:
            trace["reward"] = dict(reward.description)
        snapshot = self.backend.snapshot()
        body, head = snapshot.active_parameters - snapshot.head_parameters, snapshot.head_parameters
        return {
            "prompt_tokens": len(prompt),
            "output": {"text": text, "thinking": None, "content": text, "thinking_status": None,
                       "sampling_scope": "full", "tokens": len(tokens), "ended_by_eos": ended,
                       "length_exhausted": not ended},
            "answer_text": text,
            "reward": value,
            "trace": {"generation_length": self.length, **trace},
            "cost": meter.cost(lambda delta: delta["model_token_slots"], lambda _role, delta: (
                2 * body * delta["model_token_slots"] + 2 * head * delta["head_token_slots"])),
            "fallbacks": [],
        }

    # Algorithms: (problem, prompt, seed, seeds, reward) -> (tokens, trace, reward of the output)

    def _sample(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        return self._samples(prompt, self.sampling, [seed], f"sample:{problem.id}")[0].token_ids, {}, None

    def _greedy(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        greedy = replace(self.sampling, temperature=0.0)
        return self._samples(prompt, greedy, [seed], f"greedy:{problem.id}")[0].token_ids, {}, None

    def _beam(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        config = self.config
        result = run_diffusion_block_beam(
            backend=self.backend, prompt=prompt,
            config=DiffusionBlockBeamConfig(total_length=self.length,
                                            decision_block_size=min(int(config["decision_block_size"]), self.length),
                                            width=int(config["width"]), branching_factor=int(config["branching_factor"])),
            sampling=self.exact, seed=seed,
        )
        return result.best.token_ids, {"stages": len(result.stages),
                                       "trajectory_logprob": result.best.trajectory_logprob}, None

    def _best_of_n(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        samples = self._samples(prompt, self.sampling, [
            seeds.derive("best_of_n", problem.id, index) for index in range(int(self.config["samples"]))
        ], f"best-of-n:{problem.id}")
        sequences = [sample.token_ids for sample in samples]
        texts = [self.answer_text(prompt, tokens) for tokens in sequences]
        rng = random.Random(seeds.derive("best_of_n", problem.id, "tie-break"))
        values = None if reward is None else [float(value) for value in reward.batch(prompt, sequences)]
        chosen = best_index(self.dataset, texts, values, rng)
        grades = [self.dataset.grade(text, problem) for text in texts]
        return sequences[chosen], {
            "selected_index": chosen,
            "candidates": [{"answer": grade.answer, "correct": grade.correct,
                            "reward": None if values is None else values[index]}
                           for index, grade in enumerate(grades)],
        }, None if values is None else values[chosen]

    def _mh_power(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        config = self.config
        result = run_diffusion_trajectory_power_mh(
            backend=self.backend, prompt=prompt,
            config=DiffusionPowerMHConfig(total_length=self.length,
                                          decision_block_size=min(int(config["decision_block_size"]), self.length),
                                          updates_per_stage=int(config["updates_per_stage"]), alpha=float(config["alpha"])),
            sampling=self.exact, seed=seed,
        )
        return result.final.token_ids, {"updates": len(result.steps),
                                        "accepted": sum(step.accepted for step in result.steps),
                                        "acceptance_rate": result.acceptance_rate}, None

    def _mh(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        assert reward is not None
        config = self.config
        settings = DiffusionMHConfig(total_length=self.length, updates=int(config["updates"]),
                                     reward_temperature=reward.temperature)
        trace: dict[str, Any] = {}
        if config["proposal"] == "frozen_history":
            # Frozen exact-policy trajectories mixed into the independence proposal.
            history = config["frozen_history"]
            result: Any = run_diffusion_replay_mixture_mh(
                backend=self.backend, prompt=prompt, config=settings, sampling=self.exact,
                history=self._samples(prompt, self.exact, [
                    seeds.derive("reward_mh", problem.id, "history", index) for index in range(int(history["samples"]))
                ], f"reward-mh-history:{problem.id}"),
                history_probability=float(history["mixture"]), reward=reward.batch, seed=seed,
            )
            trace["history_draws"] = result.history_draws
        else:
            result = run_diffusion_reward_mh(backend=self.backend, prompt=prompt, config=settings,
                                             sampling=self.sampling, reward=reward.batch, seed=seed)
        trace.update(updates=len(result.steps), accepted=sum(step.accepted for step in result.steps),
                     acceptance_rate=result.acceptance_rate)
        return result.final.token_ids, trace, float(result.final_reward)

    def _is(self, problem: Problem, prompt: TokenSequence, seed: int, seeds: SeedStream, reward: Reward | None):
        assert reward is not None
        config = self.config
        result = run_conditional_diffusion_is(
            backend=self.backend, prompt=prompt,
            config=DiffusionISConfig(candidate_count=int(config["candidate_count"]),
                                     rollout_count=int(config["rollout_count"]),
                                     block_size=min(int(config["decision_block_size"]), self.length),
                                     total_length=self.length, reward_temperature=reward.temperature),
            sampling=self.sampling, seed=seed, reward=reward.batch,
        )
        # The last block completes the sequence, so its single empty completion carries the output's reward.
        return result.token_ids, importance_trace(result.steps), result.steps[-1].selected.rollouts[0].reward


__all__ = ["DLLMFamily"]

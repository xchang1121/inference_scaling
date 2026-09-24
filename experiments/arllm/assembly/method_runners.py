"""Assemble one AR method: backends, reward, algorithm config and diagnostics.

``run_method`` is the single entry used by every AR GSM8K experiment. It applies
the sampling scope (full output or thinking segment) around a runner from
``RUNNERS``; each runner turns its TOML tables into an algorithm config, builds
the reward through :func:`problem_reward`, calls the algorithm
and reports ``(tokens, diagnostics)``.
"""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from experiments.arllm.assembly.common import (
    answer_counts,
    direct_generate,
    fraction_text,
    sample_one,
    trim_eos,
)
from inference_scaling.arllm.algorithms import run_conditional_is, run_mh_chain, run_reward_mh_chain
from inference_scaling.arllm.algorithms.config import (
    ConditionalISConfig,
    IteratedConditionalISConfig,
    MHConfig,
    RewardMHConfig,
)
from inference_scaling.arllm.backends import AbsorbingEOSBackend, ScoreCachingBackend
from inference_scaling.arllm.backends.reference import ReferencePolicyBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.rewards.factory import (
    BATCH_DEPENDENT_SOURCES,
    MODEL_REWARD_SOURCES,
    NORMALIZED_CONFIDENCE_SOURCES,
    PILOT_SOURCES,
    REWARD_SOURCES,
    REWARD_TARGET_NAMES,
    SequenceReward,
    build_reward,
    model_reward_from_config,
    reward_temperature_from_config,
)
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, confidence_rewards
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.arllm.types import GenerationRequest, TokenSequence
from inference_scaling.archive.arllm.block_conditional_is import (
    BlockConditionalISConfig,
    run_block_conditional_is,
)
from inference_scaling.experimental.arllm.iterated_is import run_iterated_conditional_is
from inference_scaling.shared.evaluation import (
    NUMERIC_ANSWERS,
    GSM8KProblem,
    extract_numeric_answer,
    gsm8k_verifier_reward,
)
from inference_scaling.shared.rewards.consensus import modal_answer
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.model.generation import generation_config_for_prompt
from inference_scaling.shared.rng import SeedStream

Diagnostics = dict[str, Any]
SCOPED_METHODS = frozenset({"mh", "reward_mh", "verifier_mh"})
CONDITIONAL_METHODS = frozenset({
    "conditional_is",
    "iterated_conditional_is",
    "conditional_is_small_proposal",
    "verifier_conditional_is",
    "verifier_conditional_is_small_proposal",
    "block_conditional_is",
    "verifier_block_conditional_is",
})
# Methods whose reported results came from the archived block conditional IS.
ARCHIVED_CONDITIONAL_METHODS = frozenset({
    "block_conditional_is",
    "verifier_block_conditional_is",
    "conditional_is_small_proposal",
    "verifier_conditional_is_small_proposal",
})
# Small-proposal ablations are named for the [conditional_is] fields they change.
METHOD_VARIANTS = {
    "_unclipped": {"importance_log_ratio_clip": None, "apply_importance_correction": True},
    "_uncorrected": {"importance_log_ratio_clip": None, "apply_importance_correction": False},
}
# Archived methods draw with the seeds of the names their results were reported under.
REPORTED_NAMES = {
    "block_conditional_is": "conditional_is",
    "verifier_block_conditional_is": "verifier_conditional_is",
}


def resolve_method(method: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The method a variant name runs and the config it runs with."""
    for suffix, fields in METHOD_VARIANTS.items():
        if method.endswith(suffix):
            config = copy.deepcopy(config)
            config["conditional_is"].update(fields)
            return method.removesuffix(suffix), config
    return method, config


def method_reward_source(config: dict[str, Any], method: str) -> str:
    """Reward of a method: ``[reward].source`` when set, else its reported default (Best-of-N votes)."""
    if method.startswith("verifier_"):
        return "verifier"
    chosen = config.get("reward", {}).get("source")
    if chosen is not None or method == "best_of_n":
        return str(chosen or "self_consistency")
    if method == "iterated_conditional_is":
        return str(config.get("iterated_is", {}).get("reward", "frozen_consensus"))
    table = config.get("conditional_is", {})
    if method in ARCHIVED_CONDITIONAL_METHODS:
        return str(table.get("block_reward", table.get("reward", "self_consistency")))
    return str(table.get("reward", "frozen_consensus"))


@dataclass(frozen=True, slots=True)
class MethodRun:
    """Inputs shared by every runner for one problem."""

    method: str
    backend: Any
    problem: GSM8KProblem
    prompt: TokenSequence
    config: dict[str, Any]
    seeds: SeedStream
    proposal_backend: Any | None
    maximum: int
    sampling_temperature: float
    seed: int


def run_best_of_n_selection(
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    *,
    max_new_tokens: int,
    samples: int,
    temperature: float,
    seeds: SeedStream,
    problem_index: int,
    reward_source: str,
    config: dict[str, Any],
) -> tuple[TokenSequence, Diagnostics]:
    sampling = SamplingConfig(
        temperature=temperature,
        eos_token_id=backend.tokenizer.eos_token_id,
    )
    requests = [
        GenerationRequest(
            prompt,
            max_new_tokens,
            sampling,
            seeds.derive("best_of_n", problem_index, sample_index),
            f"best-of-n:{problem_index}:{sample_index}",
        )
        for sample_index in range(samples)
    ]
    candidates = backend.sample_batch(requests)
    sequences = [candidate.token_ids for candidate in candidates]
    parsed_answers = [extract_numeric_answer(backend.decode(tokens)) for tokens in sequences]
    raw_rewards: tuple[float, ...] | None = None
    reward: SequenceReward | None = None
    if reward_source in NORMALIZED_CONFIDENCE_SOURCES:
        raw_rewards, selection_rewards = confidence_rewards(
            backend, prompt, sequences, sampling=sampling, source=reward_source,
        )
    else:
        reward = problem_reward(
            reward_source, backend=backend, problem=problem, prompt=prompt, config=config,
            sampling=sampling, maximum=max_new_tokens, seeds=seeds,
        )
        score = reward.verifier.batch if reward.verifier is not None else reward.batch
        if reward_source == "sequence_log_probability":
            # Generation already returns exact log-probabilities under ``sampling``;
            # reusing them avoids a second full-sequence model forward pass.
            selection_rewards = tuple(
                reward.model_reward.from_token_logprobs(prompt, candidate.token_ids, candidate.token_logprobs)
                for candidate in candidates
            )
        elif score is not None:
            selection_rewards = tuple(score(prompt, sequences))
        else:
            selection_rewards = tuple(reward.pointwise(prompt, tokens) for tokens in sequences)
        if reward.model_reward is not None:
            raw_rewards = selection_rewards
    # The mean log-probability reward already ranks by likelihood.
    chosen = max(
        range(len(candidates)),
        key=lambda index: (
            (selection_rewards[index], -index)
            if reward_source == "sequence_log_probability"
            else (selection_rewards[index], candidates[index].logprob, -index)
        ),
    )
    model_reward = reward.model_reward if reward is not None else None
    verifier_reward = reward.verifier if reward is not None else None
    return candidates[chosen].token_ids, {
        "candidate_count": samples,
        "selected_index": chosen,
        "reward_source": reward_source,
        "uses_test_gold_oracle": bool(
            verifier_reward is not None
            and verifier_reward.verifier.spec.requires_reference
        ),
        "verifier": (
            verifier_reward.describe() if verifier_reward is not None else None
        ),
        "model_reward": model_reward.describe() if model_reward is not None else None,
        "reward_normalization": (
            "per-decision min-max over candidate completions"
            if reward_source in NORMALIZED_CONFIDENCE_SOURCES
            else "mean_per_effective_token" if reward_source == "sequence_log_probability"
            else None
        ),
        "selection_rewards": list(selection_rewards),
        "raw_reward_values": list(raw_rewards) if raw_rewards is not None else None,
        "raw_confidence_rewards": (
            list(raw_rewards)
            if raw_rewards is not None
            and reward_source in NORMALIZED_CONFIDENCE_SOURCES | {"consilience"}
            else None
        ),
        "answer_counts": answer_counts(parsed_answers),
    }


def consensus_pilots(
    backend: Any, prompt: TokenSequence, *, maximum: int, sampling: SamplingConfig,
    samples: int, seeds: SeedStream, problem_index: int,
) -> tuple[list[str], Diagnostics]:
    """Independent base-model texts that the frozen answer-agreement rewards compare against."""

    if samples <= 0:
        raise ValueError("frozen-consensus pilot_samples must be positive")
    requests = [
        GenerationRequest(
            prompt, maximum, sampling, seeds.derive("frozen_consensus", problem_index, index),
            f"frozen-consensus:{problem_index}:pilot:{index}",
        )
        for index in range(samples)
    ]
    texts = [backend.decode(sample.token_ids) for sample in backend.sample_batch(requests)]
    answers = [NUMERIC_ANSWERS.answer(text) for text in texts]
    return texts, {
        "pilot_samples": samples,
        "pilot_answer_counts": answer_counts(answers),
        "frozen_consensus_answer": fraction_text(modal_answer(NUMERIC_ANSWERS, answers)),
    }


def problem_reward(
    source: str, *, backend: Any, problem: GSM8KProblem, prompt: TokenSequence, config: dict[str, Any],
    sampling: SamplingConfig | None, maximum: int, seeds: SeedStream,
) -> SequenceReward:
    """Bind a reward source to one GSM8K problem: numeric answers, pilots and the reference."""

    pilots: list[str] = []
    diagnostics: Diagnostics = {}
    if source in PILOT_SOURCES:
        assert sampling is not None
        pilots, diagnostics = consensus_pilots(
            backend, prompt, maximum=maximum, sampling=sampling,
            samples=int(config.get("iterated_is", {}).get("pilot_samples", 8)),
            seeds=seeds, problem_index=problem.index,
        )
    reward = build_reward(
        source, backend=backend, config=config, sampling=sampling, rule=NUMERIC_ANSWERS,
        decode=lambda _prompt, tokens: backend.decode(tokens), pilots=pilots,
        verifier=gsm8k_verifier_reward(config, problem, backend.decode) if source == "verifier" else None,
    )
    reward.diagnostics.update(diagnostics)
    return reward


def conditional_diagnostics(result: Any) -> Diagnostics:
    ess: list[float] = []
    within_candidate_log_weight_dispersion: list[float] = []
    raw_corrections: list[float] = []
    applied_corrections: list[float] = []
    rollout_count = 0
    rewards: list[float] = []
    for step in result.steps:
        for candidate in step.candidates:
            weights = [rollout.log_weight for rollout in candidate.rollouts]
            ess.append(importance_effective_sample_size(weights))
            if len(weights) > 1:
                within_candidate_log_weight_dispersion.append(
                    statistics.pvariance(weights)
                )
            rollout_count += len(weights)
            rewards.extend(rollout.reward for rollout in candidate.rollouts)
            raw_corrections.extend(
                rollout.raw_log_importance_ratio
                for rollout in candidate.rollouts
                if rollout.raw_log_importance_ratio is not None
            )
            applied_corrections.extend(
                rollout.applied_log_importance_ratio
                for rollout in candidate.rollouts
                if rollout.applied_log_importance_ratio is not None
            )
    return {
        "guidance_steps": len(result.steps),
        "rollout_evaluations": rollout_count,
        "rollout_evaluations_planned": sum(
            int(
                getattr(
                    step,
                    "rollout_evaluations_planned",
                    sum(len(candidate.rollouts) for candidate in step.candidates),
                )
            )
            for step in result.steps
        ),
        "rollout_evaluations_performed": sum(
            int(
                getattr(
                    step,
                    "rollout_evaluations_performed",
                    sum(len(candidate.rollouts) for candidate in step.candidates),
                )
            )
            for step in result.steps
        ),
        "mean_rollout_ess": statistics.fmean(ess) if ess else 0.0,
        "mean_within_candidate_log_weight_dispersion": (
            statistics.fmean(within_candidate_log_weight_dispersion)
            if within_candidate_log_weight_dispersion
            else 0.0
        ),
        "candidate_log_weight_estimates_by_step": [
            [candidate.log_weight for candidate in step.candidates]
            for step in result.steps
        ],
        "candidate_token_ids_by_step": [
            [candidate.token_ids for candidate in step.candidates]
            for step in result.steps
        ],
        "selected_candidate_indices": [step.selected_index for step in result.steps],
        "mean_rollout_reward": statistics.fmean(rewards) if rewards else 0.0,
        "minimum_rollout_reward": min(rewards) if rewards else 0.0,
        "maximum_rollout_reward": max(rewards) if rewards else 0.0,
        "mean_absolute_raw_log_importance_correction": (
            statistics.fmean(abs(value) for value in raw_corrections)
            if raw_corrections
            else 0.0
        ),
        "mean_absolute_applied_log_importance_correction": (
            statistics.fmean(abs(value) for value in applied_corrections)
            if applied_corrections
            else 0.0
        ),
        "clipped_rollout_corrections": sum(
            raw != applied
            for raw, applied in zip(
                raw_corrections,
                applied_corrections,
                strict=True,
            )
        ),
        "importance_corrected_rollout_evaluations": len(raw_corrections),
        "uncorrected_rollout_evaluations": rollout_count - len(raw_corrections),
    }


def reference_mh_backend(backend: Any, prompt: TokenSequence, temperature: float) -> Any:
    reference: Any = ScoreCachingBackend(backend)
    if temperature != 1.0:
        reference = ReferencePolicyBackend(reference, temperature=temperature)
    return AbsorbingEOSBackend(reference, backend.tokenizer.eos_token_id, absorbing_after=len(prompt))


# TOML table -> algorithm config ----------------------------------------------------

def mh_config(table: dict[str, Any], maximum: int) -> MHConfig:
    return MHConfig(
        alpha=float(table["alpha"]),
        total_length=maximum,
        block_size=int(table["block_size"]),
        steps_per_block=int(table["steps_per_block"]),
        iterations=table.get("iterations"),
        suffix_schedule=str(table.get("suffix_schedule", "uniform")),
    )


def reward_mh_config(table: dict[str, Any], maximum: int, reward_temperature: float) -> RewardMHConfig:
    return RewardMHConfig(
        total_length=maximum,
        block_size=int(table["block_size"]),
        steps_per_block=int(table["steps_per_block"]),
        iterations=table.get("iterations"),
        reward_temperature=reward_temperature,
        suffix_schedule=str(table.get("suffix_schedule", "uniform")),
    )


def importance_log_ratio_clip(table: dict[str, Any], method: str) -> float | None:
    """Clipping applies only to corrected off-policy (small-proposal) rollouts."""
    if (
        method.endswith("small_proposal")
        and bool(table.get("apply_importance_correction", True))
        and table.get("importance_log_ratio_clip") is not None
    ):
        return float(table["importance_log_ratio_clip"])
    return None


def conditional_is_config(
    table: dict[str, Any], *, maximum: int, reward_temperature: float,
) -> ConditionalISConfig:
    return ConditionalISConfig(
        candidate_count=int(table["candidate_count"]),
        rollout_count=int(table["rollout_count"]),
        block_size=int(table["block_size"]),
        total_length=maximum,
        reward_temperature=reward_temperature,
    )


def block_conditional_is_config(
    table: dict[str, Any], *, method: str, maximum: int, reward_temperature: float,
) -> BlockConditionalISConfig:
    return BlockConditionalISConfig(
        candidate_count=int(table["candidate_count"]),
        rollout_count=int(table["rollout_count"]),
        block_size=int(table["block_size"]),
        total_length=maximum,
        reward_temperature=reward_temperature,
        importance_log_ratio_clip=importance_log_ratio_clip(table, method),
        apply_importance_correction=bool(table.get("apply_importance_correction", True)),
    )


def iterated_is_config(
    conditional: dict[str, Any], iterated: dict[str, Any], *, maximum: int, reward_temperature: float,
) -> IteratedConditionalISConfig:
    return IteratedConditionalISConfig(
        pool_size=int(iterated.get("pool_size", 3)),
        updates=int(iterated.get("updates", 4)),
        rollout_count=int(conditional["rollout_count"]),
        block_size=int(conditional["block_size"]),
        total_length=maximum,
        reward_temperature=reward_temperature,
    )


# Runners ---------------------------------------------------------------------------

def run_sampling(run: MethodRun) -> tuple[TokenSequence, Diagnostics]:
    temperature = 1.0 if run.method == "rl_sample" else run.sampling_temperature
    tokens = sample_one(
        run.backend,
        run.prompt,
        max_new_tokens=run.maximum,
        temperature=temperature,
        seed=run.seed,
        request_id=f"{run.method}:{run.problem.index}",
    )
    return tokens, {"sampling_temperature": temperature}


def run_direct(run: MethodRun) -> tuple[TokenSequence, Diagnostics]:
    beams = int(run.config["beam"]["num_beams"]) if run.method == "beam" else 1
    tokens = direct_generate(run.backend, run.prompt, max_new_tokens=run.maximum, num_beams=beams)
    forward_token_slots = beams * (len(run.prompt) + max(0, len(tokens) - 1))
    return tokens, {
        "num_beams": beams,
        "direct_generation_forward_token_slots": forward_token_slots,
        "direct_estimated_dense_forward_flops": (
            2 * run.backend.parameter_count * forward_token_slots
        ),
        "direct_compute_is_estimated": True,
        "direct_beam_compute_is_upper_bound": beams > 1,
    }


def run_best_of_n(run: MethodRun) -> tuple[TokenSequence, Diagnostics]:
    return run_best_of_n_selection(
        run.backend,
        run.problem,
        run.prompt,
        max_new_tokens=run.maximum,
        samples=int(run.config["best_of_n"]["samples"]),
        temperature=run.sampling_temperature,
        seeds=run.seeds,
        problem_index=run.problem.index,
        reward_source=method_reward_source(run.config, run.method),
        config=run.config,
    )


def run_power_mh(run: MethodRun) -> tuple[TokenSequence, Diagnostics]:
    mh = run.config["mh"]
    absorbing = reference_mh_backend(run.backend, run.prompt, run.sampling_temperature)
    result = run_mh_chain(
        absorbing,
        run.prompt,
        mh_config(mh, run.maximum),
        SamplingConfig(temperature=1.0 / float(mh["alpha"])),
        SeedStream(run.seed),
    )
    return trim_eos(result.token_ids, run.backend.tokenizer.eos_token_id), {
        "alpha": float(mh["alpha"]),
        "target_sampling_temperature": run.sampling_temperature,
        "block_size": int(mh["block_size"]),
        "steps_per_block": int(mh["steps_per_block"]),
        "iterations": mh.get("iterations"),
        "suffix_schedule": str(mh.get("suffix_schedule", "uniform")),
        "attempts": result.attempts,
        "accepted": result.accepted,
        "acceptance_rate": result.acceptance_rate,
        "mean_proposed_suffix_length": result.mean_proposed_suffix_length,
        "mean_proposed_token_changes": result.mean_proposed_token_changes,
        "mean_accepted_token_changes": result.mean_accepted_token_changes,
    }


def run_reward_mh(run: MethodRun) -> tuple[TokenSequence, Diagnostics]:
    backend, config = run.backend, run.config
    mh = config["mh"]
    is_verifier = run.method == "verifier_mh"
    source = method_reward_source(config, run.method)
    target_temperature = 1.0 if is_verifier else run.sampling_temperature
    absorbing = reference_mh_backend(backend, run.prompt, target_temperature)
    reward_temperature = (
        float(config["matched_target"]["reward_temperature"])
        if is_verifier else reward_temperature_from_config(config, source=source)
    )
    if source in BATCH_DEPENDENT_SOURCES:
        raise ValueError(f"reward MH needs a fixed per-sequence reward; {source!r} depends on other sequences")
    reward = problem_reward(
        source, backend=absorbing if source == "sequence_log_probability" else backend,
        problem=run.problem, prompt=run.prompt, config=config,
        sampling=None if source in MODEL_REWARD_SOURCES else SamplingConfig(
            temperature=target_temperature, eos_token_id=backend.tokenizer.eos_token_id,
        ),
        maximum=run.maximum, seeds=run.seeds,
    )
    reward_info: Diagnostics = {
        **reward.diagnostics,
        "uses_test_gold_oracle": bool(reward.verifier and reward.verifier.verifier.spec.requires_reference),
    }
    selected_reward = reward.pointwise

    result = run_reward_mh_chain(
        absorbing,
        run.prompt,
        reward_mh_config(mh, run.maximum, reward_temperature),
        SamplingConfig(),
        selected_reward,
        SeedStream(run.seed),
    )
    if isinstance(reward.model_reward, ConsilienceReward):
        reward_info["reward_scope_counts"] = reward.model_reward.scope_statistics()
    return trim_eos(result.token_ids, backend.tokenizer.eos_token_id), {
        "target": "base_probability_times_exp_reward_over_temperature",
        "reward_source": source,
        "target_sampling_temperature": target_temperature,
        **reward_info,
        "reward_temperature": reward_temperature,
        "block_size": int(mh["block_size"]),
        "steps_per_block": int(mh["steps_per_block"]),
        "iterations": mh.get("iterations"),
        "suffix_schedule": str(mh.get("suffix_schedule", "uniform")),
        "updates": result.attempts,
        "accepted": result.accepted,
        "acceptance_rate": result.acceptance_rate,
        "final_reward": result.reward,
        "mean_proposed_suffix_length": result.mean_proposed_suffix_length,
        "mean_proposed_token_changes": result.mean_proposed_token_changes,
        "mean_accepted_token_changes": result.mean_accepted_token_changes,
    }


def run_conditional(run: MethodRun) -> tuple[TokenSequence, Diagnostics]:
    method, backend, config = run.method, run.backend, run.config
    conditional = config["conditional_is"]
    iterated = config.get("iterated_is", {})
    small_proposal = method.endswith("small_proposal")
    rollout_backend = backend
    if small_proposal:
        if run.proposal_backend is None:
            raise ValueError("small-proposal method requires a proposal model")
        rollout_backend = run.proposal_backend
    use_matched_target = method.startswith("verifier_")
    reward_source = method_reward_source(config, method)
    if reward_source not in REWARD_SOURCES:
        raise ValueError(f"unknown reward source {reward_source!r}")
    mainline = method != "iterated_conditional_is" and method not in ARCHIVED_CONDITIONAL_METHODS
    if method not in ARCHIVED_CONDITIONAL_METHODS and reward_source in BATCH_DEPENDENT_SOURCES:
        raise ValueError(
            f"{method} reuses rewards across steps, so it needs a fixed per-sequence reward; "
            f"{reward_source!r} depends on the other scored sequences. Use frozen_consensus, "
            "pilot_agreement, verifier, consilience or sequence_log_probability."
        )
    target_sampling_temperature = 1.0 if use_matched_target else run.sampling_temperature
    reward_temperature = (
        float(config["matched_target"]["reward_temperature"])
        if use_matched_target
        else reward_temperature_from_config(config, source=reward_source)
    )
    base_sampling = SamplingConfig(
        temperature=target_sampling_temperature,
        eos_token_id=backend.tokenizer.eos_token_id,
    )
    reward = problem_reward(
        reward_source, backend=backend, problem=run.problem, prompt=run.prompt, config=config,
        sampling=base_sampling, maximum=run.maximum, seeds=run.seeds,
    )
    # Algorithms take one call; iterated IS compares stored values, so it keeps the pointwise one.
    pointwise, batch = (
        (reward.pointwise, None)
        if method == "iterated_conditional_is" or reward.batch is None
        else (None, reward.batch)
    )
    cached_base = ScoreCachingBackend(backend)
    cached_rollout = ScoreCachingBackend(rollout_backend)
    if method == "iterated_conditional_is":
        result = run_iterated_conditional_is(
            cached_base,
            run.prompt,
            iterated_is_config(
                conditional, iterated, maximum=run.maximum, reward_temperature=reward_temperature,
            ),
            pointwise,
            SeedStream(run.seed),
            base_sampling=base_sampling,
            rollout_backend=cached_rollout,
            rollout_sampling=base_sampling,
            reward_batch=batch,
        )
    elif method in ARCHIVED_CONDITIONAL_METHODS:
        result = run_block_conditional_is(
            cached_base,
            run.prompt,
            block_conditional_is_config(
                conditional, method=method, maximum=run.maximum, reward_temperature=reward_temperature,
            ),
            pointwise,
            SeedStream(run.seed),
            base_sampling=base_sampling,
            rollout_backend=cached_rollout,
            rollout_sampling=base_sampling,
            reward_batch=batch,
        )
    else:
        result = run_conditional_is(
            cached_base,
            run.prompt,
            conditional_is_config(conditional, maximum=run.maximum, reward_temperature=reward_temperature),
            pointwise,
            SeedStream(run.seed),
            sampling=base_sampling,
            reward_batch=batch,
        )
    diagnostics = conditional_diagnostics(result)
    if reward_source == "consilience":
        assert isinstance(reward.model_reward, ConsilienceReward)
        reward.diagnostics["reward_scope_counts"] = reward.model_reward.scope_statistics()
    diagnostics.update(reward.diagnostics)
    diagnostics["configured_candidate_count"] = int(conditional["candidate_count"])
    diagnostics["configured_rollout_count"] = int(conditional["rollout_count"])
    diagnostics["configured_block_size"] = int(conditional["block_size"])
    if method == "iterated_conditional_is":
        diagnostics.update(
            {
                "pool_size": int(iterated.get("pool_size", 3)),
                "updates_per_block": int(iterated.get("updates", 4)),
                "fresh_candidate_evaluations": result.fresh_candidate_evaluations,
                "reused_pool_entries": result.reused_pool_entries,
                "finite_pool_target_invariant": True,
            }
        )
    diagnostics["proposal_model"] = rollout_backend.model_id
    diagnostics["candidate_source"] = "base_model"
    target_description = (
        f"base_probability_times_exp_{REWARD_TARGET_NAMES[reward_source]}_over_temperature"
    )
    if small_proposal and not bool(conditional.get("apply_importance_correction", True)):
        target_description = (
            "base_candidates_reweighted_by_proposal_expected_exp_"
            f"{REWARD_TARGET_NAMES[reward_source]}_over_temperature"
        )
    elif small_proposal and conditional.get("importance_log_ratio_clip") is not None:
        target_description = "clipped_finite_rollout_approximation_to_" + target_description
    diagnostics["target"] = target_description
    diagnostics["reward_temperature"] = reward_temperature
    diagnostics["reward_source"] = reward_source
    diagnostics["reward_normalization"] = (
        "per-guidance-step min-max over all candidate rollouts"
        if reward_source in NORMALIZED_CONFIDENCE_SOURCES
        else "mean_per_effective_token" if reward_source == "sequence_log_probability"
        else None
    )
    diagnostics["importance_log_ratio_clip"] = importance_log_ratio_clip(conditional, method)
    diagnostics["apply_importance_correction"] = bool(
        conditional.get("apply_importance_correction", True)
    )
    diagnostics["sampling_temperature"] = target_sampling_temperature
    diagnostics["uses_test_gold_oracle"] = bool(
        reward.verifier is not None and reward.verifier.verifier.spec.requires_reference
    )
    diagnostics["uses_matched_reward_temperature"] = use_matched_target
    if mainline:
        diagnostics["retained_block_kept_steps"] = sum(
            step.retained_candidate and step.selected_index == 0 for step in result.steps
        )
    return result.token_ids, diagnostics


Runner = Callable[[MethodRun], tuple[TokenSequence, Diagnostics]]
RUNNERS: dict[str, Runner] = {
    "base": run_sampling,
    "rl_sample": run_sampling,
    "beam": run_direct,
    "rl_greedy": run_direct,
    "best_of_n": run_best_of_n,
    "mh": run_power_mh,
    "reward_mh": run_reward_mh,
    "verifier_mh": run_reward_mh,
    **dict.fromkeys(sorted(CONDITIONAL_METHODS), run_conditional),
}


def run_method_impl(
    method: str, backend: Any, problem: GSM8KProblem, prompt: TokenSequence,
    config: dict[str, Any], seeds: SeedStream, proposal_backend: Any | None,
) -> tuple[TokenSequence, Diagnostics]:
    run = MethodRun(
        method, backend, problem, prompt, config, seeds, proposal_backend,
        maximum=int(config["generation"]["max_new_tokens"]),
        sampling_temperature=float(config.get("sampling", {}).get("temperature", 1.0)),
        seed=seeds.derive(REPORTED_NAMES.get(method, method), problem.index),
    )
    runner = RUNNERS.get(method)
    if runner is None:
        raise ValueError(f"unknown method {method!r}")
    return runner(run)


def run_method(
    method: str,
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    config: dict[str, Any],
    seeds: SeedStream,
    proposal_backend: Any | None,
) -> tuple[TokenSequence, Diagnostics]:
    """Run one method inside its sampling scope and finish the final content."""

    method, config = resolve_method(method, config)
    config, length_budget = generation_config_for_prompt(config, len(prompt), [backend, proposal_backend])
    source = method_reward_source(config, method)
    scoped_algorithm = method in SCOPED_METHODS or "conditional_is" in method
    scope = SamplingScope.from_config(backend, config, active=scoped_algorithm).for_prompt(prompt)
    if scope.scope == "thinking" and method != "mh":
        if source not in MODEL_REWARD_SOURCES and config.get("reward", {}).get("input_scope") != "thinking":
            scope = scope.full_fallback("reward_uses_full_sequence")
        elif source == "consilience" and config.get("reward", {}).get("consilience", {}).get("scope") == "full":
            scope = scope.full_fallback("reward_uses_full_sequence")
    algorithm_backend = scope.wrap(backend, prompt)
    rollout_backend = None if proposal_backend is None else scope.wrap(proposal_backend, prompt)
    tokens, diagnostics = run_method_impl(
        method, algorithm_backend, problem, prompt, config, seeds, rollout_backend
    )
    maximum = int(config["generation"]["max_new_tokens"])
    temperature = 1.0 if method.startswith("verifier_") else float(config.get("sampling", {}).get("temperature", 1.0))
    tokens, output = scope.finish(
        backend, prompt, tokens, max_new_tokens=maximum,
        sampling=SamplingConfig(temperature=temperature, eos_token_id=backend.tokenizer.eos_token_id),
        seed=seeds.derive(REPORTED_NAMES.get(method, method), problem.index, "final-content"),
    )
    diagnostics["generation_budget"] = length_budget
    diagnostics["output_segments"] = output
    if diagnostics.get("reward_source") == "consilience" or (
        source == "consilience" and method not in {"mh", "base", "rl_sample", "beam", "rl_greedy"}
    ):
        selected_reward = model_reward_from_config(backend, config, source="consilience")
        assert isinstance(selected_reward, ConsilienceReward)
        output.update(selected_reward.describe_completion(prompt, tokens))
    return tokens, diagnostics


__all__ = [
    "CONDITIONAL_METHODS",
    "MethodRun",
    "RUNNERS",
    "conditional_diagnostics",
    "conditional_is_config",
    "iterated_is_config",
    "mh_config",
    "reference_mh_backend",
    "reward_mh_config",
    "run_best_of_n_selection",
    "run_method",
    "run_method_impl",
]

"""Reward reweighting and conditional importance sampling for masked dLLMs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, log

import numpy as np

from inference_scaling.dllm.config import (
    DiffusionISConfig,
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTrajectoryScoreRequest,
)
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence

DiffusionRewardFunction = Callable[[TokenSequence, TokenSequence], float]
DiffusionRewardBatchFunction = Callable[
    [TokenSequence, Sequence[TokenSequence]], Sequence[float]
]


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


def _normalized_probabilities(log_weights: Sequence[float]) -> np.ndarray:
    if not log_weights:
        raise ValueError("at least one log weight is required")
    values = np.asarray(log_weights, dtype=np.float64)
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise ValueError("all importance weights are zero or non-finite")
    weights = np.exp(values - maximum)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("importance weights cannot be normalized")
    return weights / total


def _clip_ratio(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    return min(max(value, -limit), limit)


@dataclass(frozen=True, slots=True)
class DiffusionSIRItem:
    sample: DiffusionSample
    reward: float
    target_trajectory_logprob: float | None
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float
    log_weight: float


@dataclass(frozen=True, slots=True)
class DiffusionSIRResult:
    items: tuple[DiffusionSIRItem, ...]
    probabilities: tuple[float, ...]
    selected_index: int
    effective_sample_size: float

    @property
    def selected(self) -> DiffusionSIRItem:
        return self.items[self.selected_index]


def resample_diffusion_candidates(
    *,
    samples: Sequence[DiffusionSample],
    rewards: Sequence[float],
    reward_temperature: float,
    rng: np.random.Generator,
    target_trajectory_logprobs: Sequence[float] | None = None,
    importance_log_ratio_clip: float | None = None,
) -> DiffusionSIRResult:
    """Sample once from reward-reweighted on-policy or trajectory-IS weights."""

    if len(samples) != len(rewards) or not samples:
        raise ValueError("samples and rewards must have the same positive length")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if target_trajectory_logprobs is not None and len(target_trajectory_logprobs) != len(samples):
        raise ValueError("one target trajectory score is required per sample")

    items: list[DiffusionSIRItem] = []
    for index, (sample, reward_value) in enumerate(zip(samples, rewards, strict=True)):
        target_logprob: float | None = None
        raw_ratio: float | None = None
        applied_ratio = 0.0
        if target_trajectory_logprobs is not None:
            if sample.trajectory_logprob is None:
                raise ValueError("off-policy IS requires an exact proposal trajectory density")
            target_logprob = float(target_trajectory_logprobs[index])
            raw_ratio = target_logprob - sample.trajectory_logprob
            applied_ratio = _clip_ratio(raw_ratio, importance_log_ratio_clip)
        log_weight = float(reward_value) / reward_temperature + applied_ratio
        items.append(
            DiffusionSIRItem(
                sample=sample,
                reward=float(reward_value),
                target_trajectory_logprob=target_logprob,
                raw_log_importance_ratio=raw_ratio,
                applied_log_importance_ratio=applied_ratio,
                log_weight=log_weight,
            )
        )

    log_weights = [item.log_weight for item in items]
    probabilities = _normalized_probabilities(log_weights)
    selected_index = int(rng.choice(len(items), p=probabilities))
    return DiffusionSIRResult(
        items=tuple(items),
        probabilities=tuple(float(value) for value in probabilities),
        selected_index=selected_index,
        effective_sample_size=importance_effective_sample_size(log_weights),
    )


@dataclass(frozen=True, slots=True)
class DiffusionRolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    proposal_trajectory_logprob: float | None
    target_trajectory_logprob: float | None
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float
    log_weight: float
    proposal_model_id: str
    proposal_policy_id: str


@dataclass(frozen=True, slots=True)
class DiffusionConditionalCandidate:
    token_ids: TokenSequence
    rollouts: tuple[DiffusionRolloutEvaluation, ...]
    log_energy: float


@dataclass(frozen=True, slots=True)
class DiffusionConditionalISStep:
    generated_length_before: int
    candidates: tuple[DiffusionConditionalCandidate, ...]
    probabilities: tuple[float, ...]
    selected_index: int

    @property
    def selected(self) -> DiffusionConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class DiffusionConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[DiffusionConditionalISStep, ...]


def _evaluate_rewards(
    *,
    prompt: TokenSequence,
    continuations: Sequence[TokenSequence],
    reward: DiffusionRewardFunction | None,
    reward_batch: DiffusionRewardBatchFunction | None,
) -> list[float]:
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    if reward_batch is not None:
        values = list(reward_batch(prompt, continuations))
    else:
        assert reward is not None
        values = [reward(prompt, continuation) for continuation in continuations]
    if len(values) != len(continuations):
        raise RuntimeError("reward evaluator returned an invalid number of values")
    return [float(value) for value in values]


def _needs_trajectory_correction(
    *,
    rollout_backend: DiffusionBackend,
    rollout_sampling: DiffusionSamplingConfig,
    target_backend: DiffusionBackend | None,
    target_sampling: DiffusionSamplingConfig | None,
    apply_importance_correction: bool,
) -> bool:
    if not apply_importance_correction:
        return False
    if target_backend is None or target_sampling is None:
        return False
    return not (
        rollout_backend.model_id == target_backend.model_id
        and rollout_sampling.policy_id == target_sampling.policy_id
    )


def run_conditional_diffusion_is(
    *,
    base_backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionISConfig,
    base_sampling: DiffusionSamplingConfig,
    reward: DiffusionRewardFunction | None = None,
    seed: int = 0,
    rollout_backend: DiffusionBackend | None = None,
    rollout_sampling: DiffusionSamplingConfig | None = None,
    target_rollout_backend: DiffusionBackend | None = None,
    target_rollout_sampling: DiffusionSamplingConfig | None = None,
    apply_importance_correction: bool = True,
    reward_batch: DiffusionRewardBatchFunction | None = None,
) -> DiffusionConditionalISResult:
    """Blockwise conditional-energy IS using complete dLLM rollouts.

    Base candidates always come from ``base_backend``.  When rollout generation
    uses another model or transition policy, the optional correction is the
    likelihood ratio of the *same random-remasking trajectory* under the target
    and proposal kernels.
    """

    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    target_rollout_backend = target_rollout_backend or base_backend
    target_rollout_sampling = target_rollout_sampling or base_sampling
    stage_lengths = diffusion_decision_stage_lengths(
        prompt_length=len(prompt),
        total_length=config.total_length,
        decision_block_size=config.block_size,
        sampling=base_sampling,
    )
    if rollout_sampling.block_alignment != base_sampling.block_alignment:
        raise ValueError("candidate and rollout policies must use the same block alignment")

    needs_correction = _needs_trajectory_correction(
        rollout_backend=rollout_backend,
        rollout_sampling=rollout_sampling,
        target_backend=target_rollout_backend,
        target_sampling=target_rollout_sampling,
        apply_importance_correction=apply_importance_correction,
    )
    if needs_correction:
        if not rollout_sampling.has_exact_trajectory_density:
            raise ValueError("off-policy dLLM IS requires random remasking and positive temperature")
        if not target_rollout_sampling.has_exact_trajectory_density:
            raise ValueError("the target dLLM policy must define an exact trajectory density")
        if (
            rollout_sampling.block_length != target_rollout_sampling.block_length
            or rollout_sampling.steps_per_block != target_rollout_sampling.steps_per_block
            or rollout_sampling.block_alignment
            != target_rollout_sampling.block_alignment
        ):
            raise ValueError("proposal and target trajectory schedules must match")

    seeds = SeedStream(seed)
    generated: TokenSequence = ()
    steps: list[DiffusionConditionalISStep] = []
    offset = 0
    for step_index, candidate_length in enumerate(stage_lengths):
        prefix = prompt + generated
        candidate_requests = [
            DiffusionGenerationRequest(
                prefix=prefix,
                generation_length=candidate_length,
                sampling=base_sampling,
                seed=seeds.derive("dllm-is", step_index, "candidate", candidate_index),
                request_id=f"dllm-is:step:{step_index}:candidate:{candidate_index}",
            )
            for candidate_index in range(config.candidate_count)
        ]
        candidate_samples = base_backend.sample_batch(candidate_requests)
        if len(candidate_samples) != config.candidate_count:
            raise RuntimeError("backend returned an invalid number of dLLM candidates")

        remaining = config.total_length - offset - candidate_length
        candidate_evaluations: list[list[DiffusionRolloutEvaluation]] = [
            [] for _ in candidate_samples
        ]
        if remaining == 0:
            continuations = [generated + sample.token_ids for sample in candidate_samples]
            rewards = _evaluate_rewards(
                prompt=prompt,
                continuations=continuations,
                reward=reward,
                reward_batch=reward_batch,
            )
            for candidate_index, reward_value in enumerate(rewards):
                candidate_evaluations[candidate_index].append(
                    DiffusionRolloutEvaluation(
                        token_ids=(),
                        reward=reward_value,
                        proposal_trajectory_logprob=0.0,
                        target_trajectory_logprob=0.0,
                        raw_log_importance_ratio=0.0,
                        applied_log_importance_ratio=0.0,
                        log_weight=reward_value / config.reward_temperature,
                        proposal_model_id=base_backend.model_id,
                        proposal_policy_id=base_sampling.policy_id,
                    )
                )
        else:
            rollout_requests: list[DiffusionGenerationRequest] = []
            rollout_owner: list[int] = []
            for candidate_index, candidate in enumerate(candidate_samples):
                rollout_prefix = prefix + candidate.token_ids
                for rollout_index in range(config.rollout_count):
                    rollout_requests.append(
                        DiffusionGenerationRequest(
                            prefix=rollout_prefix,
                            generation_length=remaining,
                            sampling=rollout_sampling,
                            seed=seeds.derive(
                                "dllm-is", step_index, "rollout", candidate_index, rollout_index
                            ),
                            request_id=(
                                f"dllm-is:step:{step_index}:candidate:{candidate_index}:"
                                f"rollout:{rollout_index}"
                            ),
                        )
                    )
                    rollout_owner.append(candidate_index)
            rollout_samples = rollout_backend.sample_batch(rollout_requests)
            if len(rollout_samples) != len(rollout_requests):
                raise RuntimeError("backend returned an invalid number of dLLM rollouts")

            target_scores: list[float] | None = None
            if needs_correction:
                target_scores = target_rollout_backend.score_trajectories(
                    [
                        DiffusionTrajectoryScoreRequest(sample, target_rollout_sampling)
                        for sample in rollout_samples
                    ]
                )
                if len(target_scores) != len(rollout_samples):
                    raise RuntimeError("backend returned an invalid number of trajectory scores")

            continuations = [
                generated
                + candidate_samples[owner].token_ids
                + sample.token_ids
                for owner, sample in zip(rollout_owner, rollout_samples, strict=True)
            ]
            rewards = _evaluate_rewards(
                prompt=prompt,
                continuations=continuations,
                reward=reward,
                reward_batch=reward_batch,
            )
            for rollout_index, (owner, sample, reward_value) in enumerate(
                zip(rollout_owner, rollout_samples, rewards, strict=True)
            ):
                target_logprob: float | None = None
                raw_ratio: float | None = None
                applied_ratio = 0.0
                if target_scores is not None:
                    if sample.trajectory_logprob is None:
                        raise RuntimeError("proposal omitted an exact trajectory score")
                    target_logprob = float(target_scores[rollout_index])
                    raw_ratio = target_logprob - sample.trajectory_logprob
                    applied_ratio = _clip_ratio(raw_ratio, config.importance_log_ratio_clip)
                candidate_evaluations[owner].append(
                    DiffusionRolloutEvaluation(
                        token_ids=sample.token_ids,
                        reward=reward_value,
                        proposal_trajectory_logprob=sample.trajectory_logprob,
                        target_trajectory_logprob=target_logprob,
                        raw_log_importance_ratio=raw_ratio,
                        applied_log_importance_ratio=applied_ratio,
                        log_weight=reward_value / config.reward_temperature + applied_ratio,
                        proposal_model_id=sample.model_id,
                        proposal_policy_id=sample.policy_id,
                    )
                )

        candidates = tuple(
            DiffusionConditionalCandidate(
                token_ids=sample.token_ids,
                rollouts=tuple(evaluations),
                log_energy=_logmeanexp([evaluation.log_weight for evaluation in evaluations]),
            )
            for sample, evaluations in zip(candidate_samples, candidate_evaluations, strict=True)
        )
        log_energies = [candidate.log_energy for candidate in candidates]
        probabilities = _normalized_probabilities(log_energies)
        selected_index = int(
            seeds.generator("dllm-is", step_index, "select").choice(
                len(candidates), p=probabilities
            )
        )
        steps.append(
            DiffusionConditionalISStep(
                generated_length_before=len(generated),
                candidates=candidates,
                probabilities=tuple(float(value) for value in probabilities),
                selected_index=selected_index,
            )
        )
        generated += candidates[selected_index].token_ids
        offset += candidate_length

    return DiffusionConditionalISResult(prompt=prompt, token_ids=generated, steps=tuple(steps))


__all__ = [
    "DiffusionConditionalCandidate",
    "DiffusionConditionalISResult",
    "DiffusionConditionalISStep",
    "DiffusionRewardBatchFunction",
    "DiffusionRewardFunction",
    "DiffusionRolloutEvaluation",
    "DiffusionSIRItem",
    "DiffusionSIRResult",
    "resample_diffusion_candidates",
    "run_conditional_diffusion_is",
]

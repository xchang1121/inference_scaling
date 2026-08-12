"""Conditional-energy importance sampling.

Candidate blocks are always sampled from the base model in this module.  A
completion may be sampled on-policy or from a full-support off-policy proposal.
Only the completion suffix receives the ``p_base / q`` correction.  This is the
finite-candidate, finite-rollout sampling-importance-resampling algorithm used as
the foundation for the replay extensions.  Optional symmetric clipping of the
sequence log-ratio is recorded explicitly; it is a biased variance-control
setting, while the default ``None`` retains the exact importance ratio.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np

from inference_scaling.config import ConditionalEnergyConfig, SamplingConfig
from inference_scaling.rng import SeedStream
from inference_scaling.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

RewardFunction = Callable[[TokenSequence, TokenSequence], float]
RewardBatchFunction = Callable[
    [TokenSequence, Sequence[TokenSequence]], Sequence[float]
]


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    base_logprob: float
    proposal_logprob: float
    raw_log_importance_ratio: float
    applied_log_importance_ratio: float
    log_weight: float
    proposal_model_id: str
    proposal_policy_id: str


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    rollouts: tuple[RolloutEvaluation, ...]
    log_energy: float


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


def _validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def _validate_rollout_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "off-policy IS requires proposal support wherever the base weighted target is positive; "
            "hard top-k/top-p truncation is not accepted"
        )


def _score_samples(
    base_backend: AutoregressiveBackend,
    prefixes: Sequence[TokenSequence],
    samples: Sequence[SequenceSample],
    base_sampling: SamplingConfig,
) -> list[float]:
    requests = [
        ScoreRequest(prefix, (sample.token_ids,), base_sampling)
        for prefix, sample in zip(prefixes, samples, strict=True)
    ]
    token_scores = base_backend.score_batch(requests)
    if len(token_scores) != len(samples):
        raise RuntimeError("backend returned an invalid number of base scores")
    totals: list[float] = []
    for sample, scores in zip(samples, token_scores, strict=True):
        if len(scores) != len(sample.token_ids):
            raise RuntimeError("backend returned an invalid base token score shape")
        total = float(sum(scores))
        if not isfinite(total):
            raise ValueError("rollout proposal generated a completion outside base-model support")
        totals.append(total)
    return totals


def _sample_candidates(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
) -> list[SequenceSample]:
    requests = [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive("conditional_is", step_index, "candidate", candidate_index),
            request_id=f"conditional-is:step:{step_index}:candidate:{candidate_index}",
        )
        for candidate_index in range(count)
    ]
    candidates = base_backend.sample_batch(requests)
    if len(candidates) != count:
        raise RuntimeError("backend returned an invalid number of candidates")
    for candidate in candidates:
        if not candidate.token_ids:
            raise RuntimeError("a candidate block must contain at least one token")
        if candidate.model_id != base_backend.model_id or candidate.policy_id != sampling.policy_id:
            raise RuntimeError("candidate was not sampled and scored by the requested base policy")
    return candidates


def estimate_conditional_energies(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    rollout_length: int,
    rollout_count: int,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward_temperature: float,
    importance_log_ratio_clip: float | None,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional energy with on/off-policy rollouts."""

    _validate_rollout_sampling(rollout_sampling)
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")

    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    rollout_prefixes: list[TokenSequence] = []
    terminal_candidates: set[int] = set()
    eos = rollout_sampling.eos_token_id

    for candidate_index, candidate in enumerate(candidates):
        full_generated_candidate = generated_prefix + candidate.token_ids
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        if terminal:
            terminal_candidates.add(candidate_index)
            continue
        rollout_prefix = prompt + full_generated_candidate
        for rollout_index in range(rollout_count):
            requests.append(
                GenerationRequest(
                    prefix=rollout_prefix,
                    max_new_tokens=rollout_length,
                    sampling=rollout_sampling,
                    seed=seeds.derive(
                        "conditional_is",
                        step_index,
                        "candidate",
                        candidate_index,
                        "rollout",
                        rollout_index,
                    ),
                    request_id=(
                        "conditional-is:"
                        f"step:{step_index}:candidate:{candidate_index}:rollout:{rollout_index}"
                    ),
                )
            )
            request_candidates.append(candidate_index)
            rollout_prefixes.append(rollout_prefix)

    samples = rollout_backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of rollouts")
    if rollout_backend is not base_backend:
        observe = getattr(base_backend, "observe_draft_samples", None)
        if callable(observe):
            observe(samples)
    rollout_is_base_policy = (
        rollout_backend.model_id == base_backend.model_id
        and rollout_sampling == base_sampling
    )
    if rollout_is_base_policy:
        base_totals = [sample.logprob for sample in samples]
    else:
        base_totals = (
            _score_samples(
                base_backend,
                rollout_prefixes,
                samples,
                base_sampling,
            )
            if samples
            else []
        )

    pending_by_candidate: list[
        list[tuple[TokenSequence, float, float, str, str, TokenSequence]]
    ] = [[] for _ in candidates]
    for candidate_index in terminal_candidates:
        generated = generated_prefix + candidates[candidate_index].token_ids
        pending_by_candidate[candidate_index].append(
            (
                (),
                0.0,
                0.0,
                rollout_backend.model_id,
                rollout_sampling.policy_id,
                generated,
            )
        )
    for candidate_index, sample, base_logprob in zip(
        request_candidates, samples, base_totals, strict=True
    ):
        generated = generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        proposal_logprob = sample.logprob
        pending_by_candidate[candidate_index].append(
            (
                sample.token_ids,
                base_logprob,
                proposal_logprob,
                sample.model_id,
                sample.policy_id,
                generated,
            )
        )

    pending = [item for group in pending_by_candidate for item in group]
    generated_sequences = [item[-1] for item in pending]
    if reward_batch is not None:
        rewards = tuple(float(value) for value in reward_batch(prompt, generated_sequences))
        if len(rewards) != len(pending):
            raise ValueError("reward_batch returned an invalid number of rewards")
    else:
        assert reward is not None
        rewards = tuple(float(reward(prompt, generated)) for generated in generated_sequences)
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")

    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    reward_index = 0
    for candidate_index, group in enumerate(pending_by_candidate):
        for token_ids, base_logprob, proposal_logprob, model_id, policy_id, _ in group:
            reward_value = rewards[reward_index]
            reward_index += 1
            raw_log_ratio = base_logprob - proposal_logprob
            applied_log_ratio = raw_log_ratio
            if importance_log_ratio_clip is not None:
                applied_log_ratio = max(
                    -importance_log_ratio_clip,
                    min(importance_log_ratio_clip, raw_log_ratio),
                )
            by_candidate[candidate_index].append(
                RolloutEvaluation(
                    token_ids=token_ids,
                    reward=reward_value,
                    base_logprob=base_logprob,
                    proposal_logprob=proposal_logprob,
                    raw_log_importance_ratio=raw_log_ratio,
                    applied_log_importance_ratio=applied_log_ratio,
                    log_weight=(
                        reward_value / reward_temperature
                        + applied_log_ratio
                    ),
                    proposal_model_id=model_id,
                    proposal_policy_id=policy_id,
                )
            )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if not evaluations:
            raise RuntimeError("each candidate must have at least one energy contribution")
        evaluated.append(
            ConditionalCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                rollouts=tuple(evaluations),
                log_energy=_logmeanexp([item.log_weight for item in evaluations]),
            )
        )
    return tuple(evaluated)


def conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ConditionalEnergyConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISStep:
    _validate_base_sampling(base_sampling)
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0:
        raise ValueError("generated prefix has already reached total_length")
    block_length = min(config.block_size, remaining)
    candidates = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.candidate_count,
        block_length,
        base_sampling,
        seeds,
        step_index,
    )
    evaluated = estimate_conditional_energies(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        generated_prefix=generated_prefix,
        candidates=candidates,
        rollout_length=max(0, remaining - block_length),
        rollout_count=config.rollout_count,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward_temperature=config.reward_temperature,
        importance_log_ratio_clip=config.importance_log_ratio_clip,
        reward=reward,
        seeds=seeds,
        step_index=step_index,
        reward_batch=reward_batch,
    )
    log_energies = np.asarray([candidate.log_energy for candidate in evaluated], dtype=np.float64)
    shifted = np.exp(log_energies - float(np.max(log_energies)))
    probabilities = shifted / shifted.sum()
    selected_index = int(
        seeds.generator("conditional_is", step_index, "select").choice(
            len(evaluated), p=probabilities
        )
    )
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=evaluated,
        selected_index=selected_index,
    )


def run_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalEnergyConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISResult:
    """Generate a sequence by repeatedly applying finite conditional-IS steps."""

    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")

    generated: list[int] = []
    steps: list[ConditionalISStep] = []
    step_index = 0
    while len(generated) < config.total_length:
        step = conditional_is_step(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=tuple(generated),
            config=config,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=reward_batch,
        )
        generated.extend(step.selected.token_ids)
        steps.append(step)
        eos = base_sampling.eos_token_id
        if eos is not None and eos in step.selected.token_ids:
            eos_index = generated.index(eos)
            generated = generated[: eos_index + 1]
            break
        step_index += 1
    return ConditionalISResult(prompt=prompt, token_ids=tuple(generated), steps=tuple(steps))

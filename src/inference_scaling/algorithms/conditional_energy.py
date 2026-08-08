"""Conditional-energy importance sampling.

Candidate blocks are always sampled from the base model in this module.  A
completion may be sampled on-policy or from a full-support off-policy proposal.
Only the completion suffix receives the ``p_base / q`` correction.  This is the
finite-candidate, finite-rollout sampling-importance-resampling algorithm used as
the foundation for the replay extensions.
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


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    base_logprob: float
    proposal_logprob: float
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
    if sampling.temperature != 1 or sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use the unmodified base distribution: "
            "temperature=1, top_p=1, top_k=None"
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
) -> list[float]:
    requests = [
        ScoreRequest(prefix, (sample.token_ids,), None)
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
    rollout_sampling: SamplingConfig,
    reward_temperature: float,
    reward: RewardFunction,
    seeds: SeedStream,
    step_index: int,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional energy with on/off-policy rollouts."""

    _validate_rollout_sampling(rollout_sampling)
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")

    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    rollout_prefixes: list[TokenSequence] = []
    terminal_evaluations: dict[int, RolloutEvaluation] = {}
    eos = rollout_sampling.eos_token_id

    for candidate_index, candidate in enumerate(candidates):
        full_generated_candidate = generated_prefix + candidate.token_ids
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        if terminal:
            reward_value = float(reward(prompt, full_generated_candidate))
            terminal_evaluations[candidate_index] = RolloutEvaluation(
                token_ids=(),
                reward=reward_value,
                base_logprob=0.0,
                proposal_logprob=0.0,
                log_weight=reward_value / reward_temperature,
                proposal_model_id=rollout_backend.model_id,
                proposal_policy_id=rollout_sampling.policy_id,
            )
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
    base_totals = _score_samples(base_backend, rollout_prefixes, samples) if samples else []

    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    for candidate_index, sample, base_logprob in zip(
        request_candidates, samples, base_totals, strict=True
    ):
        generated = generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        reward_value = float(reward(prompt, generated))
        if not isfinite(reward_value):
            raise ValueError("reward must be finite")
        proposal_logprob = sample.logprob
        log_weight = (
            reward_value / reward_temperature + base_logprob - proposal_logprob
        )
        by_candidate[candidate_index].append(
            RolloutEvaluation(
                token_ids=sample.token_ids,
                reward=reward_value,
                base_logprob=base_logprob,
                proposal_logprob=proposal_logprob,
                log_weight=log_weight,
                proposal_model_id=sample.model_id,
                proposal_policy_id=sample.policy_id,
            )
        )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if candidate_index in terminal_evaluations:
            evaluations = [terminal_evaluations[candidate_index]]
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
    reward: RewardFunction,
    seeds: SeedStream,
    step_index: int,
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
        rollout_sampling=rollout_sampling,
        reward_temperature=config.reward_temperature,
        reward=reward,
        seeds=seeds,
        step_index=step_index,
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
    reward: RewardFunction,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
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

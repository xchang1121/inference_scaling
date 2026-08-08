"""Base-candidate off-policy rollout replay with exact fresh tail correction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log1p

import numpy as np

from inference_scaling.algorithms.conditional_energy import (
    RewardFunction,
    _logmeanexp,
    _sample_candidates,
    _validate_base_sampling,
)
from inference_scaling.config import BaseReplayConfig, SamplingConfig
from inference_scaling.metrics import importance_effective_sample_size
from inference_scaling.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    FrozenReplayClaim,
    InMemoryReplayStore,
    ReplayKey,
    ReplayRecord,
    ReplaySampleRequest,
    mixture_logprobabilities,
    sample_replay_records,
    validate_record_probabilities,
)
from inference_scaling.rng import SeedStream
from inference_scaling.types import AutoregressiveBackend, ScoreRequest, SequenceSample, TokenSequence


@dataclass(frozen=True, slots=True)
class ProbabilityObservation:
    base_logprob: float
    mixture_logprob: float
    reward: float


@dataclass(frozen=True, slots=True)
class ReplayEnergyEstimate:
    log_energy: float
    history_log_terms: tuple[float, ...]
    fresh_log_terms: tuple[float, ...]
    history_record_ids: tuple[str, ...]
    behavior_counts: tuple[tuple[str, int], ...]

    @property
    def history_count(self) -> int:
        return len(self.history_log_terms)

    @property
    def fresh_count(self) -> int:
        return len(self.fresh_log_terms)

    @property
    def history_ess(self) -> float:
        return importance_effective_sample_size(self.history_log_terms)

    @property
    def fresh_ess(self) -> float:
        finite = [value for value in self.fresh_log_terms if value != float("-inf")]
        return importance_effective_sample_size(finite)


@dataclass(frozen=True, slots=True)
class BaseReplayCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    estimate: ReplayEnergyEstimate


@dataclass(frozen=True, slots=True)
class BaseReplayStep:
    generated_length_before: int
    candidates: tuple[BaseReplayCandidate, ...]
    selected_index: int

    @property
    def selected(self) -> BaseReplayCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class BaseReplayResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[BaseReplayStep, ...]
    reserve_records_written: int


def _logsum_pair(left: float, right: float) -> float:
    return float(np.logaddexp(left, right))


def corrected_replay_log_energy(
    history: Sequence[ProbabilityObservation],
    fresh: Sequence[ProbabilityObservation],
    *,
    truncation: float,
    reward_temperature: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Evaluate the document's truncated-history plus fresh-tail identity."""

    if not history:
        raise ValueError("corrected replay requires at least one history observation")
    if not fresh:
        raise ValueError("corrected replay requires at least one fresh observation")
    if truncation <= 0 or reward_temperature <= 0:
        raise ValueError("truncation and reward_temperature must be positive")
    log_tau = log(truncation)
    history_terms: list[float] = []
    for observation in history:
        log_ratio = observation.base_logprob - observation.mixture_logprob
        if not isfinite(log_ratio):
            raise ValueError("selected history behavior must assign mass only inside base support")
        history_terms.append(
            min(log_tau, log_ratio) + observation.reward / reward_temperature
        )

    fresh_terms: list[float] = []
    for observation in fresh:
        if observation.mixture_logprob == float("-inf"):
            log_tail = 0.0
        else:
            log_ratio = observation.base_logprob - observation.mixture_logprob
            if log_ratio <= log_tau:
                log_tail = float("-inf")
            else:
                log_tail = log1p(-exp(log_tau - log_ratio))
        fresh_terms.append(log_tail + observation.reward / reward_temperature)

    history_mean = _logmeanexp(history_terms)
    finite_fresh = [value for value in fresh_terms if value != float("-inf")]
    fresh_mean = (
        _logmeanexp(finite_fresh) + log(len(finite_fresh) / len(fresh_terms))
        if finite_fresh
        else float("-inf")
    )
    return _logsum_pair(history_mean, fresh_mean), tuple(history_terms), tuple(fresh_terms)


def _score_base(
    backend: AutoregressiveBackend,
    key: ReplayKey,
    completions: Sequence[TokenSequence],
    sampling: SamplingConfig,
) -> tuple[float, ...]:
    if not completions:
        return ()
    scored = backend.score_batch(
        [ScoreRequest(key.rollout_prefix, tuple(completions), sampling)]
    )
    if len(scored) != len(completions):
        raise RuntimeError("base backend returned an invalid number of scores")
    totals: list[float] = []
    for completion, token_scores in zip(completions, scored, strict=True):
        if len(completion) != len(token_scores):
            raise RuntimeError("base backend returned an invalid token score shape")
        total = float(sum(token_scores))
        if not isfinite(total):
            raise ValueError("replay behavior generated a completion outside base support")
        totals.append(total)
    return tuple(totals)


def build_fresh_replay_requests(
    *,
    key: ReplayKey,
    count: int,
    rollout_length: int,
    seeds: SeedStream,
    step_index: int,
    candidate_index: int,
) -> tuple[ReplaySampleRequest, ...]:
    return tuple(
        ReplaySampleRequest(
            key=key,
            max_new_tokens=rollout_length,
            seed=seeds.derive(
                "base_replay", step_index, "candidate", candidate_index, "fresh", fresh_index
            ),
            record_id=(
                f"fresh:{step_index}:{candidate_index}:{fresh_index}:"
                f"{seeds.derive('fresh-id', step_index, candidate_index, fresh_index)}"
            ),
        )
        for fresh_index in range(count)
    )


def _fresh_records(
    *,
    base_policy: BehaviorPolicy,
    key: ReplayKey,
    count: int,
    rollout_length: int,
    reward: RewardFunction,
    seeds: SeedStream,
    step_index: int,
    candidate_index: int,
) -> tuple[ReplayRecord, ...]:
    return sample_replay_records(
        base_policy,
        build_fresh_replay_requests(
            key=key,
            count=count,
            rollout_length=rollout_length,
            seeds=seeds,
            step_index=step_index,
            candidate_index=candidate_index,
        ),
        reward,
    )


def estimate_replay_energy(
    *,
    base_backend: AutoregressiveBackend,
    base_policy: BehaviorPolicy,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    claim: FrozenReplayClaim,
    fresh_count: int,
    rollout_length: int,
    reward: RewardFunction,
    reward_temperature: float,
    truncation: float,
    seeds: SeedStream,
    step_index: int,
    candidate_index: int,
    precomputed_fresh_records: Sequence[ReplayRecord] | None = None,
) -> ReplayEnergyEstimate:
    if precomputed_fresh_records is None:
        fresh_records = _fresh_records(
            base_policy=base_policy,
            key=claim.key,
            count=fresh_count,
            rollout_length=rollout_length,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            candidate_index=candidate_index,
        )
    else:
        fresh_records = tuple(precomputed_fresh_records)
        if len(fresh_records) != fresh_count:
            raise ValueError("precomputed fresh record count does not match the frozen design")
        if any(
            record.key != claim.key or record.behavior_id != base_policy.behavior_id
            for record in fresh_records
        ):
            raise ValueError("precomputed fresh records do not match the candidate and base policy")

    if claim.count == 0:
        if store.reveal_and_consume(claim):
            raise RuntimeError("an empty replay claim unexpectedly contained records")
        for record in fresh_records:
            store.add_design(record)
        log_terms = tuple(record.reward / reward_temperature for record in fresh_records)
        return ReplayEnergyEstimate(
            log_energy=_logmeanexp(log_terms),
            history_log_terms=(),
            fresh_log_terms=log_terms,
            history_record_ids=(),
            behavior_counts=(),
        )

    history_records = store.reveal_and_consume(claim)
    validate_record_probabilities(history_records, registry)
    behavior_counts = dict(claim.behavior_counts)
    history_completions = [record.completion for record in history_records]
    fresh_completions = [record.completion for record in fresh_records]
    history_base = _score_base(
        base_backend,
        claim.key,
        history_completions,
        base_policy.sampling,
    )
    fresh_base = tuple(record.behavior_logprob for record in fresh_records)
    history_mixture = mixture_logprobabilities(
        registry, claim.key, behavior_counts, history_completions
    )
    fresh_mixture = mixture_logprobabilities(
        registry, claim.key, behavior_counts, fresh_completions
    )
    history_observations = tuple(
        ProbabilityObservation(log_p, log_b, record.reward)
        for record, log_p, log_b in zip(
            history_records, history_base, history_mixture, strict=True
        )
    )
    fresh_observations = tuple(
        ProbabilityObservation(log_p, log_b, record.reward)
        for record, log_p, log_b in zip(fresh_records, fresh_base, fresh_mixture, strict=True)
    )
    log_energy, history_terms, fresh_terms = corrected_replay_log_energy(
        history_observations,
        fresh_observations,
        truncation=truncation,
        reward_temperature=reward_temperature,
    )
    for record in fresh_records:
        store.add_design(record)
    return ReplayEnergyEstimate(
        log_energy=log_energy,
        history_log_terms=history_terms,
        fresh_log_terms=fresh_terms,
        history_record_ids=tuple(record.record_id for record in history_records),
        behavior_counts=claim.behavior_counts,
    )


def base_replay_step(
    *,
    base_backend: AutoregressiveBackend,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: BaseReplayConfig,
    base_sampling: SamplingConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    step_index: int,
) -> BaseReplayStep:
    """Run one base-candidate guidance step with a frozen replay design."""

    _validate_base_sampling(base_sampling)
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0:
        raise ValueError("generated prefix has already reached total_length")
    block_length = min(config.block_size, remaining)
    candidate_samples = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.candidate_count,
        block_length,
        base_sampling,
        seeds,
        step_index,
    )
    base_policy = BehaviorPolicy.for_backend(base_backend, base_sampling, label="base")
    registry.register(base_policy)
    rollout_length = max(0, remaining - block_length)
    eos = base_sampling.eos_token_id

    keys = [
        ReplayKey(prompt, generated_prefix, candidate.token_ids, reward_version)
        for candidate in candidate_samples
    ]
    claims: list[FrozenReplayClaim | None] = []
    for key, candidate in zip(keys, candidate_samples, strict=True):
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        claims.append(
            None
            if terminal
            else store.freeze_claims([key], config.max_history_per_candidate)[0]
        )

    fresh_requests: list[ReplaySampleRequest] = []
    fresh_ranges: list[tuple[int, int] | None] = []
    for candidate_index, (key, claim) in enumerate(zip(keys, claims, strict=True)):
        if claim is None:
            fresh_ranges.append(None)
            continue
        start = len(fresh_requests)
        fresh_requests.extend(
            build_fresh_replay_requests(
                key=key,
                count=config.fresh_rollouts,
                rollout_length=rollout_length,
                seeds=seeds,
                step_index=step_index,
                candidate_index=candidate_index,
            )
        )
        fresh_ranges.append((start, len(fresh_requests)))
    batched_fresh = sample_replay_records(base_policy, fresh_requests, reward)

    candidates: list[BaseReplayCandidate] = []
    for candidate_index, (candidate, key, claim, fresh_range) in enumerate(
        zip(candidate_samples, keys, claims, fresh_ranges, strict=True)
    ):
        if claim is None:
            terminal_reward = float(reward(prompt, generated_prefix + candidate.token_ids))
            estimate = ReplayEnergyEstimate(
                log_energy=terminal_reward / config.reward_temperature,
                history_log_terms=(),
                fresh_log_terms=(terminal_reward / config.reward_temperature,),
                history_record_ids=(),
                behavior_counts=(),
            )
        else:
            assert fresh_range is not None
            estimate = estimate_replay_energy(
                base_backend=base_backend,
                base_policy=base_policy,
                registry=registry,
                store=store,
                claim=claim,
                fresh_count=config.fresh_rollouts,
                rollout_length=rollout_length,
                reward=reward,
                reward_temperature=config.reward_temperature,
                truncation=config.truncation,
                seeds=seeds,
                step_index=step_index,
                candidate_index=candidate_index,
                precomputed_fresh_records=batched_fresh[
                    fresh_range[0] : fresh_range[1]
                ],
            )
        candidates.append(
            BaseReplayCandidate(candidate.token_ids, candidate.token_logprobs, estimate)
        )

    log_energies = np.asarray(
        [candidate.estimate.log_energy for candidate in candidates], dtype=np.float64
    )
    weights = np.exp(log_energies - float(np.max(log_energies)))
    probabilities = weights / weights.sum()
    selected_index = int(
        seeds.generator("base_replay", step_index, "select").choice(
            len(candidates), p=probabilities
        )
    )
    return BaseReplayStep(len(generated_prefix), tuple(candidates), selected_index)


def write_reserve_records(
    *,
    base_backend: AutoregressiveBackend,
    base_sampling: SamplingConfig,
    reserve_policy: BehaviorPolicy,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: BaseReplayConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    step_index: int,
) -> int:
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0 or config.reserve_rollouts == 0:
        return 0
    block_length = min(config.block_size, remaining)
    reserve_candidates = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.reserve_rollouts,
        block_length,
        base_sampling,
        seeds,
        step_index + 1_000_000,
    )
    reserve_requests: list[ReplaySampleRequest] = []
    for reserve_index, candidate in enumerate(reserve_candidates):
        rollout_length = remaining - len(candidate.token_ids)
        eos = base_sampling.eos_token_id
        terminal = rollout_length <= 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        if terminal:
            continue
        key = ReplayKey(prompt, generated_prefix, candidate.token_ids, reward_version)
        reserve_requests.append(
            ReplaySampleRequest(
                key=key,
                max_new_tokens=rollout_length,
                seed=seeds.derive("base_replay", step_index, "reserve", reserve_index),
                record_id=(
                f"reserve:{step_index}:{reserve_index}:"
                f"{seeds.derive('reserve-id', step_index, reserve_index)}"
                ),
            )
        )
    records = sample_replay_records(reserve_policy, reserve_requests, reward)
    for record in records:
        store.add_evaluation(record)
    return len(records)


def run_base_replay(
    base_backend: AutoregressiveBackend,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    config: BaseReplayConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    reserve_policy: BehaviorPolicy | None = None,
) -> BaseReplayResult:
    base_sampling = base_sampling or SamplingConfig()
    _validate_base_sampling(base_sampling)
    base_policy = BehaviorPolicy.for_backend(base_backend, base_sampling, label="base")
    registry.register(base_policy)
    reserve_policy = reserve_policy or base_policy
    registry.register(reserve_policy)
    generated: list[int] = []
    steps: list[BaseReplayStep] = []
    reserve_written = 0
    step_index = 0
    while len(generated) < config.total_length:
        step = base_replay_step(
            base_backend=base_backend,
            registry=registry,
            store=store,
            prompt=prompt,
            generated_prefix=tuple(generated),
            config=config,
            base_sampling=base_sampling,
            reward=reward,
            reward_version=reward_version,
            seeds=seeds,
            step_index=step_index,
        )
        generated.extend(step.selected.token_ids)
        steps.append(step)
        eos = base_sampling.eos_token_id
        if eos is not None and eos in step.selected.token_ids:
            generated = generated[: generated.index(eos) + 1]
            break
        reserve_written += write_reserve_records(
            base_backend=base_backend,
            base_sampling=base_sampling,
            reserve_policy=reserve_policy,
            store=store,
            prompt=prompt,
            generated_prefix=tuple(generated),
            config=config,
            reward=reward,
            reward_version=reward_version,
            seeds=seeds,
            step_index=step_index,
        )
        step_index += 1
    return BaseReplayResult(
        prompt=prompt,
        token_ids=tuple(generated),
        steps=tuple(steps),
        reserve_records_written=reserve_written,
    )

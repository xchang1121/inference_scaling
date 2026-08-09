"""Dynamic candidate proposals with replay-corrected conditional energies.

This module adds three independent mechanisms to ``base_replay``: a defensive
candidate mixture, an outer base/proposal probability ratio, and a frozen
variance--cost allocation.  Completion-level replay correction is reused
without changing its estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

import numpy as np

from inference_scaling.algorithms.base_replay import (
    ReplayEnergyEstimate,
    build_fresh_replay_requests,
    estimate_replay_energy,
    write_reserve_records,
)
from inference_scaling.algorithms.conditional_energy import (
    RewardFunction,
    _validate_base_sampling,
)
from inference_scaling.config import DynamicISConfig, SamplingConfig
from inference_scaling.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplaySampleRequest,
    mixture_logprobabilities,
    score_continuations,
    sample_replay_records,
    validate_record_probabilities,
)
from inference_scaling.rng import SeedStream
from inference_scaling.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    proposal_id: str
    backend: AutoregressiveBackend
    sampling: SamplingConfig

    @classmethod
    def for_backend(
        cls,
        backend: AutoregressiveBackend,
        sampling: SamplingConfig,
        *,
        label: str | None = None,
    ) -> "CandidateProposal":
        return cls(label or f"{backend.model_id}|{sampling.policy_id}", backend, sampling)


@dataclass(frozen=True, slots=True)
class DynamicCandidateDraw:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    base_logprob: float
    auxiliary_logprob: float
    proposal_logprob: float
    outer_log_ratio: float
    source: str
    auxiliary_proposal_id: str | None


CandidateProposalFactory = Callable[
    [int, int, tuple[DynamicCandidateDraw, ...]], CandidateProposal
]
CandidateProposalSource = CandidateProposal | CandidateProposalFactory | None


@dataclass(frozen=True, slots=True)
class VarianceCostEstimate:
    history_std: float
    fresh_std: float
    history_cost: float = 1.0
    fresh_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.history_std < 0 or self.fresh_std < 0:
            raise ValueError("standard deviations must be non-negative")
        if not all(
            isfinite(value)
            for value in (
                self.history_std,
                self.fresh_std,
                self.history_cost,
                self.fresh_cost,
            )
        ):
            raise ValueError("variance and cost estimates must be finite")
        if self.history_cost <= 0 or self.fresh_cost <= 0:
            raise ValueError("per-sample costs must be positive")


@dataclass(frozen=True, slots=True)
class DesignStatisticsContext:
    key: ReplayKey
    evaluation_inventory: tuple[tuple[str, int], ...]
    store: InMemoryReplayStore
    registry: BehaviorRegistry
    base_policy: BehaviorPolicy
    reward_temperature: float
    truncation: float

    @property
    def available_history(self) -> int:
        return sum(count for _, count in self.evaluation_inventory)


DesignStatisticsProvider = Callable[[DesignStatisticsContext], VarianceCostEstimate]
DesignPreparation = Callable[[tuple[DesignStatisticsContext, ...]], None]


@dataclass(frozen=True, slots=True)
class RolloutBudgetContext:
    """Metadata available when freezing one step's rollout budget.

    The context intentionally exposes replay inventory counts but not evaluation
    completions or rewards.  This permits a cost-matched budget to depend on the
    candidates that were actually drawn and on cache hits without leaking the
    values later used by the estimator.
    """

    draws: tuple[DynamicCandidateDraw, ...]
    keys: tuple[ReplayKey, ...]
    terminal: tuple[bool, ...]
    history_capacities: tuple[int, ...]


RolloutBudgetProvider = Callable[[RolloutBudgetContext], float]


def constant_design_statistics(context: DesignStatisticsContext) -> VarianceCostEstimate:
    """Safe cold-start statistics; callers can replace this with design-pool estimates."""

    return VarianceCostEstimate(
        history_std=1.0 if context.available_history else 0.0,
        fresh_std=1.0,
    )


def _empirical_standard_deviation(values: Sequence[float], fallback: float) -> float:
    if len(values) < 2:
        return fallback
    result = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    return result if isfinite(result) else fallback


def empirical_design_statistics(context: DesignStatisticsContext) -> VarianceCostEstimate:
    """Estimate allocation inputs using only records already in the design pool."""

    inventory = dict(context.evaluation_inventory)
    design_records = context.store.design_records(context.key)
    behavior_ids = set(inventory)
    history_records = [
        record for record in design_records if record.behavior_id in behavior_ids
    ]
    base_records = [
        record
        for record in design_records
        if record.behavior_id == context.base_policy.behavior_id
    ]
    log_tau = log(context.truncation)

    history_values: list[float] = []
    if inventory and history_records:
        validate_record_probabilities(history_records, context.registry)
        completions = [record.completion for record in history_records]
        base_scores = score_continuations(context.base_policy, context.key, completions)
        mixture_scores = mixture_logprobabilities(
            context.registry, context.key, inventory, completions
        )
        history_values = [
            exp(
                min(log_tau, base_logprob - mixture_logprob)
                + record.reward / context.reward_temperature
            )
            for record, base_logprob, mixture_logprob in zip(
                history_records, base_scores, mixture_scores, strict=True
            )
        ]

    fresh_values: list[float] = []
    if base_records:
        if inventory:
            mixture_scores = mixture_logprobabilities(
                context.registry,
                context.key,
                inventory,
                [record.completion for record in base_records],
            )
            for record, mixture_logprob in zip(base_records, mixture_scores, strict=True):
                log_ratio = record.behavior_logprob - mixture_logprob
                tail = 0.0 if log_ratio <= log_tau else 1.0 - exp(log_tau - log_ratio)
                fresh_values.append(
                    tail * exp(record.reward / context.reward_temperature)
                )
        else:
            fresh_values = [
                exp(record.reward / context.reward_temperature) for record in base_records
            ]

    behavior_score_multiplier = 1 + len(inventory)
    history_cost = (
        max(
            1.0,
            behavior_score_multiplier
            * sum(len(record.completion) for record in history_records)
            / len(history_records),
        )
        if history_records
        else 1.0
    )
    fresh_cost = (
        max(
            1.0,
            behavior_score_multiplier
            * sum(len(record.completion) for record in base_records)
            / len(base_records),
        )
        if base_records
        else 1.0
    )
    return VarianceCostEstimate(
        history_std=(
            _empirical_standard_deviation(history_values, 1.0)
            if context.available_history
            else 0.0
        ),
        fresh_std=_empirical_standard_deviation(fresh_values, 1.0),
        history_cost=history_cost,
        fresh_cost=fresh_cost,
    )


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    history_count: int
    fresh_count: int
    continuous_history: float
    continuous_fresh: float
    estimated_cost: float


def _proportional_capped_counts(
    total: float,
    coefficients: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(coefficients, dtype=np.float64)
    active = {index for index, coefficient in enumerate(coefficients) if coefficient > 0}
    remaining = min(float(total), float(caps.sum()))
    while active and remaining > 1e-12:
        coefficient_sum = sum(float(coefficients[index]) for index in active)
        if coefficient_sum <= 0:
            break
        proposed = {
            index: remaining * float(coefficients[index]) / coefficient_sum
            for index in active
        }
        saturated = [
            index
            for index in active
            if proposed[index] >= float(caps[index]) - result[index] - 1e-12
        ]
        if not saturated:
            for index, value in proposed.items():
                result[index] += value
            break
        for index in saturated:
            addition = max(0.0, float(caps[index]) - result[index])
            result[index] += addition
            remaining -= addition
            active.remove(index)
    return result


def allocate_variance_cost_budget(
    *,
    outer_ratios: Sequence[float],
    statistics: Sequence[VarianceCostEstimate],
    history_capacities: Sequence[int],
    history_groups: Sequence[Hashable],
    group_capacities: Mapping[Hashable, int],
    rollout_budget: float,
    minimum_fresh: int | Sequence[int] = 1,
) -> tuple[BudgetAllocation, ...]:
    """Continuous optimum followed by deterministic largest-remainder rounding."""

    count = len(outer_ratios)
    if not (
        len(statistics) == count
        and len(history_capacities) == count
        and len(history_groups) == count
    ):
        raise ValueError("all candidate-level allocation inputs must have the same length")
    if count == 0:
        raise ValueError("at least one candidate is required")
    if rollout_budget <= 0 or not isfinite(rollout_budget):
        raise ValueError("rollout_budget must be positive and finite")
    if isinstance(minimum_fresh, int):
        minimum = np.full(count, minimum_fresh, dtype=np.int64)
    else:
        if len(minimum_fresh) != count:
            raise ValueError("minimum_fresh must have one value per candidate")
        minimum = np.asarray(minimum_fresh, dtype=np.int64)
    if np.any(minimum < 0):
        raise ValueError("minimum fresh counts must be non-negative")

    ratios = np.asarray(outer_ratios, dtype=np.float64)
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("outer probability ratios must be positive and finite")
    history_caps = np.asarray(history_capacities, dtype=np.int64)
    if np.any(history_caps < 0):
        raise ValueError("history capacities must be non-negative")
    history_costs = np.asarray([item.history_cost for item in statistics], dtype=np.float64)
    fresh_costs = np.asarray([item.fresh_cost for item in statistics], dtype=np.float64)
    history_coefficients = np.asarray(
        [
            ratio * item.history_std / sqrt(item.history_cost)
            for ratio, item in zip(ratios, statistics, strict=True)
        ],
        dtype=np.float64,
    )
    fresh_coefficients = np.asarray(
        [
            ratio * item.fresh_std / sqrt(item.fresh_cost)
            for ratio, item in zip(ratios, statistics, strict=True)
        ],
        dtype=np.float64,
    )
    history_coefficients[history_caps == 0] = 0.0

    mandatory_cost = float(np.dot(minimum, fresh_costs))
    if mandatory_cost > rollout_budget + 1e-9:
        raise ValueError(
            "rollout budget cannot cover the required fresh sample for every candidate"
        )
    remaining_budget = rollout_budget - mandatory_cost
    continuous_history = np.zeros(count, dtype=np.float64)
    continuous_fresh_extra = np.zeros(count, dtype=np.float64)
    active_history = {
        index
        for index in range(count)
        if history_caps[index] > 0 and history_coefficients[index] > 0
    }
    group_remaining = {
        group: float(capacity) for group, capacity in group_capacities.items()
    }
    for group, capacity in group_remaining.items():
        if capacity < 0:
            raise ValueError(f"history group {group!r} has a negative capacity")

    while remaining_budget > 1e-12:
        active_fresh = {index for index in range(count) if fresh_coefficients[index] > 0}
        denominator = sum(
            float(history_coefficients[index] * history_costs[index])
            for index in active_history
        ) + sum(
            float(fresh_coefficients[index] * fresh_costs[index])
            for index in active_fresh
        )
        if denominator <= 0:
            active_fresh = {index for index in range(count) if minimum[index] > 0}
            if not active_fresh:
                break
            temporary_fresh_coefficients = 1.0 / fresh_costs
            denominator = float(
                sum(
                    temporary_fresh_coefficients[index] * fresh_costs[index]
                    for index in active_fresh
                )
            )
        else:
            temporary_fresh_coefficients = fresh_coefficients

        proposed_history = {
            index: remaining_budget * history_coefficients[index] / denominator
            for index in active_history
        }
        proposed_fresh = {
            index: remaining_budget * temporary_fresh_coefficients[index] / denominator
            for index in active_fresh
        }

        violated_group: Hashable | None = None
        for group in dict.fromkeys(history_groups):
            members = [
                index
                for index in active_history
                if history_groups[index] == group
            ]
            if members and sum(proposed_history[index] for index in members) > (
                group_remaining.get(group, 0.0) + 1e-12
            ):
                violated_group = group
                break
        if violated_group is not None:
            members = [
                index
                for index in active_history
                if history_groups[index] == violated_group
            ]
            coefficients = np.asarray(
                [history_coefficients[index] for index in members], dtype=np.float64
            )
            caps = np.asarray(
                [history_caps[index] - continuous_history[index] for index in members],
                dtype=np.float64,
            )
            fixed = _proportional_capped_counts(
                group_remaining.get(violated_group, 0.0), coefficients, caps
            )
            fixed_cost = sum(
                value * history_costs[member]
                for member, value in zip(members, fixed, strict=True)
            )
            if fixed_cost > remaining_budget + 1e-12:
                fixed *= remaining_budget / fixed_cost
            for member, value in zip(members, fixed, strict=True):
                continuous_history[member] += value
                remaining_budget -= value * history_costs[member]
                active_history.discard(member)
            group_remaining[violated_group] = 0.0
            continue

        violated_index = next(
            (
                index
                for index in sorted(active_history)
                if proposed_history[index]
                > history_caps[index] - continuous_history[index] + 1e-12
            ),
            None,
        )
        if violated_index is not None:
            addition = float(history_caps[violated_index] - continuous_history[violated_index])
            continuous_history[violated_index] += addition
            remaining_budget -= addition * history_costs[violated_index]
            group = history_groups[violated_index]
            group_remaining[group] = max(0.0, group_remaining.get(group, 0.0) - addition)
            active_history.remove(violated_index)
            continue

        for index, value in proposed_history.items():
            continuous_history[index] += value
        for index, value in proposed_fresh.items():
            continuous_fresh_extra[index] += value
        remaining_budget = 0.0

    continuous_fresh = minimum.astype(np.float64) + continuous_fresh_extra
    history_integer = np.floor(continuous_history + 1e-12).astype(np.int64)
    fresh_integer = minimum + np.floor(continuous_fresh_extra + 1e-12).astype(np.int64)
    integer_cost = float(
        np.dot(history_integer, history_costs) + np.dot(fresh_integer, fresh_costs)
    )
    budget_left = rollout_budget - integer_cost
    group_used: dict[Hashable, int] = {}
    for index, group in enumerate(history_groups):
        group_used[group] = group_used.get(group, 0) + int(history_integer[index])

    remainders: list[tuple[float, int, str]] = []
    for index in range(count):
        history_fraction = float(continuous_history[index] - history_integer[index])
        fresh_fraction = float(continuous_fresh[index] - fresh_integer[index])
        if history_fraction > 1e-12:
            remainders.append((history_fraction, index, "history"))
        if fresh_fraction > 1e-12:
            remainders.append((fresh_fraction, index, "fresh"))
    remainders.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, index, source in remainders:
        cost = history_costs[index] if source == "history" else fresh_costs[index]
        if cost > budget_left + 1e-9:
            continue
        if source == "history":
            group = history_groups[index]
            if history_integer[index] >= history_caps[index]:
                continue
            if group_used.get(group, 0) >= group_capacities.get(group, 0):
                continue
            history_integer[index] += 1
            group_used[group] = group_used.get(group, 0) + 1
        else:
            fresh_integer[index] += 1
        budget_left -= float(cost)

    allocations = []
    for index in range(count):
        cost = (
            history_integer[index] * history_costs[index]
            + fresh_integer[index] * fresh_costs[index]
        )
        allocations.append(
            BudgetAllocation(
                history_count=int(history_integer[index]),
                fresh_count=int(fresh_integer[index]),
                continuous_history=float(continuous_history[index]),
                continuous_fresh=float(continuous_fresh[index]),
                estimated_cost=float(cost),
            )
        )
    return tuple(allocations)


@dataclass(frozen=True, slots=True)
class DynamicCandidate:
    draw: DynamicCandidateDraw
    allocation: BudgetAllocation
    estimate: ReplayEnergyEstimate
    log_weight: float

    @property
    def token_ids(self) -> TokenSequence:
        return self.draw.token_ids


@dataclass(frozen=True, slots=True)
class DynamicISStep:
    generated_length_before: int
    candidates: tuple[DynamicCandidate, ...]
    selected_index: int

    @property
    def selected(self) -> DynamicCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class DynamicISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[DynamicISStep, ...]
    reserve_records_written: int


def _score_candidate(
    backend: AutoregressiveBackend,
    sampling: SamplingConfig | None,
    prefix: TokenSequence,
    candidate: TokenSequence,
) -> tuple[float, ...]:
    return _score_candidates(backend, sampling, prefix, (candidate,))[0]


def _score_candidates(
    backend: AutoregressiveBackend,
    sampling: SamplingConfig | None,
    prefix: TokenSequence,
    candidates: Sequence[TokenSequence],
) -> tuple[tuple[float, ...], ...]:
    if not candidates:
        return ()
    scored = backend.score_batch([ScoreRequest(prefix, tuple(candidates), sampling)])
    if len(scored) != len(candidates) or any(
        len(token_scores) != len(candidate)
        for token_scores, candidate in zip(scored, candidates, strict=True)
    ):
        raise RuntimeError("candidate proposal returned an invalid score shape")
    return tuple(scored)


def _resolve_auxiliary(
    source: CandidateProposalSource,
    step_index: int,
    candidate_index: int,
    previous: tuple[DynamicCandidateDraw, ...],
) -> CandidateProposal | None:
    if source is None or isinstance(source, CandidateProposal):
        return source
    proposal = source(step_index, candidate_index, previous)
    if not isinstance(proposal, CandidateProposal):
        raise TypeError("candidate proposal factory must return CandidateProposal")
    return proposal


def _draw_candidates(
    *,
    base_backend: AutoregressiveBackend,
    base_sampling: SamplingConfig,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    mixture: float,
    auxiliary_source: CandidateProposalSource,
    seeds: SeedStream,
    step_index: int,
) -> tuple[DynamicCandidateDraw, ...]:
    if auxiliary_source is None or isinstance(auxiliary_source, CandidateProposal):
        auxiliary = auxiliary_source
        if mixture > 0 and auxiliary is None:
            raise ValueError("a positive auxiliary mixture requires a candidate proposal")
        if (
            auxiliary is not None
            and auxiliary.sampling.eos_token_id != base_sampling.eos_token_id
        ):
            raise ValueError(
                "base and auxiliary candidate policies must agree on eos_token_id"
            )

        request_groups: dict[
            tuple[int, SamplingConfig],
            tuple[AutoregressiveBackend, list[tuple[int, GenerationRequest]]],
        ] = {}
        used_auxiliary: list[bool] = []
        for candidate_index in range(count):
            use_auxiliary = bool(
                mixture > 0
                and seeds.generator(
                    "dynamic_is", step_index, candidate_index, "component"
                ).random()
                < mixture
            )
            used_auxiliary.append(use_auxiliary)
            backend = (
                auxiliary.backend
                if use_auxiliary and auxiliary is not None
                else base_backend
            )
            sampling = (
                auxiliary.sampling
                if use_auxiliary and auxiliary is not None
                else base_sampling
            )
            request = GenerationRequest(
                prefix=prefix,
                max_new_tokens=block_length,
                sampling=sampling,
                seed=seeds.derive("dynamic_is", step_index, candidate_index, "sample"),
                request_id=f"dynamic-is:step:{step_index}:candidate:{candidate_index}",
            )
            key = (id(backend), sampling)
            if key not in request_groups:
                request_groups[key] = (backend, [])
            request_groups[key][1].append((candidate_index, request))

        ordered_samples: list[SequenceSample | None] = [None] * count
        for backend, indexed_requests in request_groups.values():
            sampled = backend.sample_batch([request for _, request in indexed_requests])
            if len(sampled) != len(indexed_requests):
                raise RuntimeError("candidate proposal returned an invalid sample count")
            for (candidate_index, request), sample in zip(
                indexed_requests, sampled, strict=True
            ):
                if not sample.token_ids:
                    raise RuntimeError("candidate proposal must return a non-empty candidate")
                if (
                    sample.model_id != backend.model_id
                    or sample.policy_id != request.sampling.policy_id
                ):
                    raise RuntimeError("candidate sample does not match its declared proposal")
                ordered_samples[candidate_index] = sample
        if any(sample is None for sample in ordered_samples):
            raise RuntimeError("candidate proposal omitted a sample")
        samples = tuple(sample for sample in ordered_samples if sample is not None)
        token_sequences = tuple(sample.token_ids for sample in samples)
        base_scores = _score_candidates(base_backend, None, prefix, token_sequences)
        auxiliary_scores = (
            _score_candidates(
                auxiliary.backend, auxiliary.sampling, prefix, token_sequences
            )
            if auxiliary is not None
            else None
        )

        draws: list[DynamicCandidateDraw] = []
        for candidate_index, (sample, base_token_scores) in enumerate(
            zip(samples, base_scores, strict=True)
        ):
            base_logprob = float(sum(base_token_scores))
            if not isfinite(base_logprob):
                raise ValueError("auxiliary candidate proposal generated outside base support")
            auxiliary_logprob = float("-inf")
            auxiliary_id: str | None = None
            if auxiliary is not None:
                assert auxiliary_scores is not None
                auxiliary_logprob = float(sum(auxiliary_scores[candidate_index]))
                auxiliary_id = auxiliary.proposal_id
                if used_auxiliary[candidate_index] and not isfinite(auxiliary_logprob):
                    raise RuntimeError(
                        "auxiliary sampler generated a zero-probability candidate"
                    )
            proposal_logprob = (
                base_logprob
                if mixture == 0
                else float(
                    np.logaddexp(
                        log(1.0 - mixture) + base_logprob,
                        log(mixture) + auxiliary_logprob,
                    )
                )
            )
            draws.append(
                DynamicCandidateDraw(
                    token_ids=sample.token_ids,
                    base_token_logprobs=tuple(base_token_scores),
                    base_logprob=base_logprob,
                    auxiliary_logprob=auxiliary_logprob,
                    proposal_logprob=proposal_logprob,
                    outer_log_ratio=base_logprob - proposal_logprob,
                    source="auxiliary" if used_auxiliary[candidate_index] else "base",
                    auxiliary_proposal_id=auxiliary_id,
                )
            )
        return tuple(draws)

    draws: list[DynamicCandidateDraw] = []
    for candidate_index in range(count):
        auxiliary = _resolve_auxiliary(
            auxiliary_source, step_index, candidate_index, tuple(draws)
        )
        if mixture > 0 and auxiliary is None:
            raise ValueError("a positive auxiliary mixture requires a candidate proposal")
        if auxiliary is not None and auxiliary.sampling.eos_token_id != base_sampling.eos_token_id:
            raise ValueError("base and auxiliary candidate policies must agree on eos_token_id")
        use_auxiliary = bool(
            mixture > 0
            and seeds.generator("dynamic_is", step_index, candidate_index, "component").random()
            < mixture
        )
        backend = auxiliary.backend if use_auxiliary and auxiliary is not None else base_backend
        sampling = auxiliary.sampling if use_auxiliary and auxiliary is not None else base_sampling
        request = GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive("dynamic_is", step_index, candidate_index, "sample"),
            request_id=f"dynamic-is:step:{step_index}:candidate:{candidate_index}",
        )
        samples = backend.sample_batch([request])
        if len(samples) != 1 or not samples[0].token_ids:
            raise RuntimeError("candidate proposal must return one non-empty candidate")
        sample = samples[0]
        if sample.model_id != backend.model_id or sample.policy_id != sampling.policy_id:
            raise RuntimeError("candidate sample does not match its declared proposal")
        base_token_scores = _score_candidate(base_backend, None, prefix, sample.token_ids)
        base_logprob = float(sum(base_token_scores))
        if not isfinite(base_logprob):
            raise ValueError("auxiliary candidate proposal generated outside base support")
        auxiliary_logprob = float("-inf")
        auxiliary_id: str | None = None
        if auxiliary is not None:
            auxiliary_scores = _score_candidate(
                auxiliary.backend, auxiliary.sampling, prefix, sample.token_ids
            )
            auxiliary_logprob = float(sum(auxiliary_scores))
            auxiliary_id = auxiliary.proposal_id
            if use_auxiliary and not isfinite(auxiliary_logprob):
                raise RuntimeError("auxiliary sampler generated a zero-probability candidate")
        if mixture == 0:
            proposal_logprob = base_logprob
        else:
            proposal_logprob = float(
                np.logaddexp(
                    log(1.0 - mixture) + base_logprob,
                    log(mixture) + auxiliary_logprob,
                )
            )
        draws.append(
            DynamicCandidateDraw(
                token_ids=sample.token_ids,
                base_token_logprobs=tuple(base_token_scores),
                base_logprob=base_logprob,
                auxiliary_logprob=auxiliary_logprob,
                proposal_logprob=proposal_logprob,
                outer_log_ratio=base_logprob - proposal_logprob,
                source="auxiliary" if use_auxiliary else "base",
                auxiliary_proposal_id=auxiliary_id,
            )
        )
    return tuple(draws)


def dynamic_is_step(
    *,
    base_backend: AutoregressiveBackend,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: DynamicISConfig,
    base_sampling: SamplingConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    step_index: int,
    auxiliary_proposal: CandidateProposalSource = None,
    statistics_provider: DesignStatisticsProvider = constant_design_statistics,
    design_prepare: DesignPreparation | None = None,
    rollout_budget_provider: RolloutBudgetProvider | None = None,
) -> DynamicISStep:
    """Run one two-phase dynamic-candidate decision."""

    _validate_base_sampling(base_sampling)
    remaining = config.total_length - len(generated_prefix)
    if remaining <= 0:
        raise ValueError("generated prefix has already reached total_length")
    block_length = min(config.block_size, remaining)
    prefix = prompt + generated_prefix
    draws = _draw_candidates(
        base_backend=base_backend,
        base_sampling=base_sampling,
        prefix=prefix,
        count=config.candidate_count,
        block_length=block_length,
        mixture=config.auxiliary_mixture,
        auxiliary_source=auxiliary_proposal,
        seeds=seeds,
        step_index=step_index,
    )
    base_policy = BehaviorPolicy.for_backend(base_backend, base_sampling, label="base")
    registry.register(base_policy)
    rollout_length = max(0, remaining - block_length)
    eos = base_sampling.eos_token_id
    keys = [
        ReplayKey(prompt, generated_prefix, draw.token_ids, reward_version) for draw in draws
    ]
    terminal = [
        rollout_length == 0 or (eos is not None and draw.token_ids[-1] == eos)
        for draw in draws
    ]

    inventory_by_key = {key: store.inventory(key) for key in dict.fromkeys(keys)}
    group_capacities = {
        key: sum(inventory.values()) for key, inventory in inventory_by_key.items()
    }
    contexts: list[DesignStatisticsContext | None] = []
    history_capacities: list[int] = []
    minimum_fresh: list[int] = []
    for key, is_terminal in zip(keys, terminal, strict=True):
        inventory = inventory_by_key[key]
        if is_terminal:
            contexts.append(None)
            history_capacities.append(0)
            minimum_fresh.append(0)
            continue
        context = DesignStatisticsContext(
            key=key,
            evaluation_inventory=tuple(sorted(inventory.items())),
            store=store,
            registry=registry,
            base_policy=base_policy,
            reward_temperature=config.reward_temperature,
            truncation=config.truncation,
        )
        contexts.append(context)
        history_capacities.append(
            min(config.max_history_per_candidate, context.available_history)
        )
        minimum_fresh.append(config.minimum_fresh_per_candidate)

    if design_prepare is not None:
        design_prepare(tuple(context for context in contexts if context is not None))
    statistics = [
        VarianceCostEstimate(0.0, 0.0)
        if context is None
        else statistics_provider(context)
        for context in contexts
    ]

    outer_log_ratios = np.asarray([draw.outer_log_ratio for draw in draws], dtype=np.float64)
    relative_outer_ratios = np.exp(
        np.maximum(
            outer_log_ratios - float(np.max(outer_log_ratios)),
            log(np.finfo(np.float64).tiny),
        )
    )
    rollout_budget = (
        config.rollout_budget
        if rollout_budget_provider is None
        else float(
            rollout_budget_provider(
                RolloutBudgetContext(
                    draws=draws,
                    keys=tuple(keys),
                    terminal=tuple(terminal),
                    history_capacities=tuple(history_capacities),
                )
            )
        )
    )
    if rollout_budget <= 0 or not isfinite(rollout_budget):
        raise ValueError("rollout budget provider must return a positive finite value")
    allocations = allocate_variance_cost_budget(
        # A common scale cancels from every continuous allocation formula.
        outer_ratios=relative_outer_ratios,
        statistics=statistics,
        history_capacities=history_capacities,
        history_groups=keys,
        group_capacities=group_capacities,
        rollout_budget=rollout_budget,
        minimum_fresh=minimum_fresh,
    )

    claims = []
    for key, allocation, is_terminal in zip(keys, allocations, terminal, strict=True):
        if is_terminal:
            claims.append(None)
            continue
        claim = store.freeze_claims([key], allocation.history_count)[0]
        if claim.count != allocation.history_count:
            raise RuntimeError("frozen replay inventory changed during allocation")
        claims.append(claim)

    fresh_requests: list[ReplaySampleRequest] = []
    fresh_ranges: list[tuple[int, int] | None] = []
    for candidate_index, (key, allocation, is_terminal) in enumerate(
        zip(keys, allocations, terminal, strict=True)
    ):
        if is_terminal:
            fresh_ranges.append(None)
            continue
        start = len(fresh_requests)
        fresh_requests.extend(
            build_fresh_replay_requests(
                key=key,
                count=allocation.fresh_count,
                rollout_length=rollout_length,
                seeds=seeds,
                step_index=step_index,
                candidate_index=candidate_index,
            )
        )
        fresh_ranges.append((start, len(fresh_requests)))
    batched_fresh = sample_replay_records(base_policy, fresh_requests, reward)

    candidates: list[DynamicCandidate] = []
    for candidate_index, (
        draw,
        key,
        allocation,
        claim,
        is_terminal,
        fresh_range,
    ) in enumerate(
        zip(draws, keys, allocations, claims, terminal, fresh_ranges, strict=True)
    ):
        if is_terminal:
            reward_value = float(reward(prompt, generated_prefix + draw.token_ids))
            estimate = ReplayEnergyEstimate(
                log_energy=reward_value / config.reward_temperature,
                history_log_terms=(),
                fresh_log_terms=(reward_value / config.reward_temperature,),
                history_record_ids=(),
                behavior_counts=(),
            )
        else:
            assert claim is not None
            assert fresh_range is not None
            estimate = estimate_replay_energy(
                base_backend=base_backend,
                base_policy=base_policy,
                registry=registry,
                store=store,
                claim=claim,
                fresh_count=allocation.fresh_count,
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
            DynamicCandidate(
                draw=draw,
                allocation=allocation,
                estimate=estimate,
                log_weight=draw.outer_log_ratio + estimate.log_energy,
            )
        )

    log_weights = np.asarray([candidate.log_weight for candidate in candidates])
    weights = np.exp(log_weights - float(np.max(log_weights)))
    probabilities = weights / weights.sum()
    selected_index = int(
        seeds.generator("dynamic_is", step_index, "select").choice(
            len(candidates), p=probabilities
        )
    )
    return DynamicISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(candidates),
        selected_index=selected_index,
    )


def run_dynamic_is(
    base_backend: AutoregressiveBackend,
    registry: BehaviorRegistry,
    store: InMemoryReplayStore,
    prompt: TokenSequence,
    config: DynamicISConfig,
    reward: RewardFunction,
    reward_version: str,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    auxiliary_proposal: CandidateProposalSource = None,
    statistics_provider: DesignStatisticsProvider = constant_design_statistics,
    design_prepare: DesignPreparation | None = None,
    rollout_budget_provider: RolloutBudgetProvider | None = None,
    reserve_policy: BehaviorPolicy | None = None,
) -> DynamicISResult:
    base_sampling = base_sampling or SamplingConfig()
    _validate_base_sampling(base_sampling)
    base_policy = BehaviorPolicy.for_backend(base_backend, base_sampling, label="base")
    registry.register(base_policy)
    reserve_policy = reserve_policy or base_policy
    registry.register(reserve_policy)
    generated: list[int] = []
    steps: list[DynamicISStep] = []
    reserve_written = 0
    step_index = 0
    while len(generated) < config.total_length:
        step = dynamic_is_step(
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
            auxiliary_proposal=auxiliary_proposal,
            statistics_provider=statistics_provider,
            design_prepare=design_prepare,
            rollout_budget_provider=rollout_budget_provider,
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
    return DynamicISResult(
        prompt=prompt,
        token_ids=tuple(generated),
        steps=tuple(steps),
        reserve_records_written=reserve_written,
    )

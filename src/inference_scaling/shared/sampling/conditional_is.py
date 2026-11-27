"""Conditional IS weights, block-Metropolized selection, and SMC particle resampling.

Independent of the generation backend. Each candidate block is weighted by the
mean ``exp(reward / temperature)`` of its completions; selection uses the
block-Metropolized kernel (merging identical block mass before transferring),
and multi-particle stages resample systematically to avoid premature mode
collapse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from math import fsum, log

import numpy as np
from numpy.typing import NDArray

from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.importance import categorical_index_from_uniform, logmeanexp, normalize_log_weights
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    """One completed rollout, its reward and log weight including any sampling correction."""

    token_ids: TokenSequence
    reward: float
    log_weight: float


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    rollouts: tuple[RolloutEvaluation, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    token_count: int
    reward: float
    log_weight: float


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    token_count: int
    rollouts: tuple[RolloutSummary, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate | CandidateSummary, ...]
    selected_index: int
    # The completion of the selected candidate kept in the sequence; None when the sampler keeps none.
    completion_index: int | None
    # Whether candidate 0 continues the kept sequence.
    retained_candidate: bool
    # Completions generated and scored in this step; a reused one is excluded.
    rollout_evaluations_performed: int
    # Forward token slots this step cost, measured per request; None when the backend cannot separate it.
    # Measurement never enters the step's identity: two execution paths are compared on what they sampled.
    forward_token_slots: int | None = field(default=None, compare=False)
    # Planned (block size, candidates, rollouts), before terminal groups collapse to one leaf.
    sampling_plan: tuple[int, int, int] | None = None
    # Original candidate slots when stopping or auxiliary constraints remove entire groups.
    candidate_indices: tuple[int, ...] | None = None

    @property
    def selected(self) -> ConditionalCandidate | CandidateSummary:
        return self.candidates[self.selected_index]

    def compact(self) -> ConditionalISStep:
        """Only lengths and scalar statistics survive a completed step."""
        return replace(self, candidates=tuple(
            CandidateSummary(len(candidate.token_ids), tuple(
                RolloutSummary(len(rollout.token_ids), rollout.reward, rollout.log_weight)
                for rollout in candidate.rollouts), candidate.log_weight)
            if isinstance(candidate, ConditionalCandidate) else candidate for candidate in self.candidates))


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]
    # Forward token slots of all steps, measured per request; None when any step could not be attributed.
    forward_token_slots: int | None = field(default=None, compare=False)
    stopping_reason: str = "stop"


def weigh(blocks: Sequence[TokenSequence],
          completions: Sequence[Sequence[RolloutEvaluation]]) -> tuple[ConditionalCandidate, ...]:
    """Each candidate block with its completions, weighted by the mean of their weights."""

    return tuple(ConditionalCandidate(block, tuple(group), logmeanexp([item.log_weight for item in group]))
                 for block, group in zip(blocks, completions, strict=True))


def _metropolized_probabilities(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    # Sum complements directly: total - weight loses small off-diagonal masses.
    leave_out = np.array([fsum(float(w) for j, w in enumerate(weights) if i != j)
                          for i in range(len(weights))])
    denominator = np.maximum(leave_out[0], leave_out)
    probabilities = np.zeros_like(weights)
    np.divide(weights[1:], denominator[1:], out=probabilities[1:], where=denominator[1:] > 0)
    stay = 1 - fsum(probabilities)
    if stay < -1e-12:
        raise ArithmeticError("invalid Metropolized index probabilities")
    probabilities[0] = max(0., stay)
    return probabilities / probabilities.sum()


def select_metropolized(candidates: Sequence[ConditionalCandidate], seeds: SeedStream, *key: object,
                        group_blocks: bool = True) -> tuple[int, int]:
    """One block-Metropolized index update from candidate zero's first complete rollout.

    Leaf mass is ``exp(log_weight) / group_size``, including terminal one-leaf
    groups. Grouping sums identical blocks' masses, applies the Metropolized
    transfer across distinct blocks, then samples a leaf conditionally inside
    the chosen block.
    """
    if not candidates or any(not candidate.rollouts for candidate in candidates):
        raise ValueError("Metropolized selection requires non-empty candidate groups")
    labels = [(m, k) for m, candidate in enumerate(candidates) for k in range(len(candidate.rollouts))]
    logs = np.array([candidates[m].rollouts[k].log_weight - log(len(candidates[m].rollouts)) for m, k in labels])
    if not np.isfinite(logs).all():
        raise ValueError("Metropolized selection requires finite complete rollout weights")
    weights = np.exp(logs - logs.max())
    if group_blocks:
        groups: dict[TokenSequence, int] = {}
        membership = np.array([groups.setdefault(candidates[m].token_ids, len(groups)) for m, _ in labels])
        mass = np.asarray(np.bincount(membership, weights=weights), dtype=np.float64)
        conditional = np.divide(weights, mass[membership], out=np.zeros_like(weights), where=mass[membership] > 0)
        probabilities = _metropolized_probabilities(mass)[membership] * conditional
        probabilities /= probabilities.sum()
    else:
        probabilities = _metropolized_probabilities(weights)
    chosen = categorical_index_from_uniform(probabilities.tolist(), float(seeds.generator(*key, "metropolized").random()))
    return labels[chosen]


def calibrate_reward_temperature(
    reward_groups: Sequence[Sequence[float]],
    base_temperature: float,
    *,
    min_scale: float = 0.5,
    max_scale: float = 4.0,
    min_ess: float = 1.35,
    max_ess_fraction: float = 0.85,
) -> float:
    """Calibrate an effective reward temperature keeping candidate ESS in a healthy band.

    Symmetric (permutation-invariant) over candidate groups. Returns
    ``base_temperature`` unchanged whenever its effective sample size is already
    inside ``[min(min_ess, 0.7 * M), max_ess_fraction * M]``.
    """
    m = len(reward_groups)
    if m < 2 or any(not group for group in reward_groups):
        return base_temperature

    def ess(tau: float) -> float:
        group_logs = [logmeanexp([value / tau for value in group]) for group in reward_groups]
        weights = normalize_log_weights(group_logs)
        denom = fsum(w * w for w in weights)
        return 1.0 / denom if denom > 0 else float(m)

    low_target = min(min_ess, 0.7 * m)
    high_target = max(low_target, max_ess_fraction * m)
    ess_base = ess(base_temperature)
    if low_target <= ess_base <= high_target:
        return base_temperature
    if ess_base < low_target:
        lo, hi = base_temperature, base_temperature * max_scale
        if ess(hi) <= low_target:
            return hi
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if ess(mid) < low_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    lo, hi = base_temperature * min_scale, base_temperature
    if ess(lo) >= high_target:
        return lo
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if ess(mid) > high_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def resample_particles(
    candidates: Sequence[ConditionalCandidate],
    count: int,
    seeds: SeedStream,
    *key: object,
) -> list[tuple[int, int]]:
    """Systematic resampling of ``count`` (candidate_idx, rollout_idx) pairs.

    Preserves exact marginal importance weights while minimizing resampling
    variance and avoiding premature mode collapse relative to i.i.d. draws.
    When the same candidate group is selected multiple times, its completions
    are also drawn systematically to retain suffix diversity.
    """
    if count <= 0:
        raise ValueError("particle count must be positive")
    if not candidates or any(not candidate.rollouts for candidate in candidates):
        raise ValueError("particle resampling requires non-empty candidate groups")
    probs = normalize_log_weights([candidate.log_weight for candidate in candidates])
    base_u = float(seeds.generator(*key, "systematic_groups").random())
    selected_groups: list[int] = []
    for slot in range(count):
        u = min(1.0 - 1e-15, (base_u + slot) / count)
        selected_groups.append(categorical_index_from_uniform(probs, u))

    # Count multiplicity per group so replicated groups draw systematically across rollouts.
    multiplicity: dict[int, int] = {}
    for g in selected_groups:
        multiplicity[g] = multiplicity.get(g, 0) + 1
    group_completions: dict[int, list[int]] = {}
    for g, copies in multiplicity.items():
        r_probs = normalize_log_weights([rollout.log_weight for rollout in candidates[g].rollouts])
        u0 = float(seeds.generator(*key, "candidate", g, "systematic_completion").random())
        group_completions[g] = [
            categorical_index_from_uniform(r_probs, min(1.0 - 1e-15, (u0 + slot) / copies))
            for slot in range(copies)
        ]
    cursors: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for g in selected_groups:
        pos = cursors.get(g, 0)
        cursors[g] = pos + 1
        pairs.append((g, group_completions[g][pos]))
    return pairs


__all__ = [
    "CandidateSummary",
    "ConditionalCandidate",
    "ConditionalISResult",
    "ConditionalISStep",
    "RolloutEvaluation",
    "RolloutSummary",
    "calibrate_reward_temperature",
    "resample_particles",
    "select_metropolized",
    "weigh",
]

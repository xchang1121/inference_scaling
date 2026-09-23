"""Pilot statistics and model-independent joint M/K/block-size planning.

The score is a plug-in, stationary-horizon forecast, not a certified TV bound.
Production candidates and rollouts must be independent of the design sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite, sqrt

import numpy as np


def positive_integer(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True)
class WeightMoments:
    """Scale-invariant moments: variances divided by squared mean weight."""

    relative_between: float
    relative_within: float
    candidate_count: int = 0

    def __post_init__(self) -> None:
        for value in (self.relative_between, self.relative_within):
            if not isfinite(value) or value < 0:
                raise ValueError("relative variances must be finite and non-negative")
        positive_integer("candidate_count", self.candidate_count, minimum=0)


def estimate_weight_moments(
    log_weights: Sequence[Sequence[float]],
    *,
    deterministic: Sequence[bool] | None = None,
) -> WeightMoments:
    """Subtract within-group MC noise from the variance of candidate means.

    Use one common log shift, never a separate normalization per candidate.
    Singleton groups are allowed only for explicitly deterministic terminal weights.
    """
    if len(log_weights) < 2:
        raise ValueError("at least two independent pilot candidates are required")
    terminal = (
        tuple(deterministic)
        if deterministic is not None
        else (False,) * len(log_weights)
    )
    if len(terminal) != len(log_weights):
        raise ValueError("deterministic flags must match candidate groups")
    groups = [np.asarray(group, dtype=np.float64) for group in log_weights]
    for group, exact in zip(groups, terminal, strict=True):
        if group.ndim != 1 or len(group) < (1 if exact else 2):
            raise ValueError("nonterminal candidates need at least two pilot rollouts")
        if not np.all(np.isfinite(group)):
            raise ValueError("pilot log-weights must be finite")
        if exact and not np.all(group == group[0]):
            raise ValueError("deterministic weights must agree")
    shift = max(float(group.max()) for group in groups)
    weights = [np.exp(group - shift) for group in groups]
    means = np.asarray([float(group.mean()) for group in weights])
    variances = np.asarray(
        [
            0.0 if exact else float(group.var(ddof=1))
            for group, exact in zip(weights, terminal, strict=True)
        ]
    )
    noise = float(
        np.mean(
            [
                variance / len(group)
                for variance, group in zip(variances, weights, strict=True)
            ]
        )
    )
    mean_squared = float(means.mean()) ** 2
    return WeightMoments(
        relative_between=max(0.0, float(means.var(ddof=1)) - noise) / mean_squared,
        relative_within=float(variances.mean()) / mean_squared,
        candidate_count=len(groups),
    )


@dataclass(frozen=True, slots=True)
class BlockBudgetEstimate:
    block_size: int
    moments: WeightMoments
    candidate_cost: float
    rollout_cost: float

    def __post_init__(self) -> None:
        positive_integer("block_size", self.block_size)
        if not isfinite(self.candidate_cost) or self.candidate_cost <= 0:
            raise ValueError("candidate_cost must be finite and positive")
        if not isfinite(self.rollout_cost) or self.rollout_cost < 0:
            raise ValueError("rollout_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class JointBudgetPlan:
    block_size: int
    candidate_count: int
    rollout_count: int
    reserved_cost: float
    forecast_steps: int
    local_error_estimate: float
    forecast_error_score: float
    used_pilot: bool


def choose_joint_budget(
    estimates: Sequence[BlockBudgetEstimate],
    *,
    remaining_length: int,
    remaining_budget: float,
    candidate_counts: Sequence[int],
    rollout_counts: Sequence[int],
    finish_reserve: float = 0.0,
    relative_variance_floor: float = 1e-4,
    forecast_full_horizon: bool = True,
) -> JointBudgetPlan | None:
    """Enumerate integer plans under forecast and next-step reservation limits.

    ``finish_reserve`` is the cost of completing from any subsequent prefix. It
    prevents a cheap-looking first action from consuming the completion budget.
    Terminal blocks score full candidates directly and have rollout_count == 0.
    Disable the full-horizon forecast for next-step planning at a fixed block size.
    The remaining length still defines the hard output boundary, not a prediction.
    """
    if not isinstance(forecast_full_horizon, bool):
        raise ValueError("forecast_full_horizon must be a boolean")
    if not forecast_full_horizon and len(estimates) > 1:
        raise ValueError("next-chunk planning requires a single block estimate")
    positive_integer("remaining_length", remaining_length)
    for name, value in (
        ("remaining_budget", remaining_budget),
        ("finish_reserve", finish_reserve),
        ("relative_variance_floor", relative_variance_floor),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not candidate_counts or not rollout_counts:
        raise ValueError("candidate and rollout grids cannot be empty")
    for value in candidate_counts:
        positive_integer("candidate_count", value, minimum=2)
    for value in rollout_counts:
        positive_integer("rollout_count", value)
    if len({item.block_size for item in estimates}) != len(estimates):
        raise ValueError("duplicate block estimates")
    plans: list[JointBudgetPlan] = []
    for estimate in estimates:
        block = estimate.block_size
        if block > remaining_length:
            raise ValueError("block size exceeds remaining length")
        terminal = block == remaining_length
        if terminal and estimate.moments.relative_within != 0:
            raise ValueError("terminal block has no rollout variance")
        if not terminal and estimate.rollout_cost <= 0:
            raise ValueError("nonterminal block requires positive rollout cost")
        stages = ceil(remaining_length / block) if forecast_full_horizon else 1
        for candidates in sorted(set(candidate_counts)):
            for rollouts in (0,) if terminal else sorted(set(rollout_counts)):
                cost = candidates * (
                    estimate.candidate_cost + rollouts * estimate.rollout_cost
                )
                if (
                    max(stages * cost, cost + (0 if terminal else finish_reserve))
                    > remaining_budget
                ):
                    continue
                variance = max(
                    estimate.moments.relative_between, relative_variance_floor
                )
                if not terminal:
                    variance += (
                        max(estimate.moments.relative_within, relative_variance_floor)
                        / rollouts
                    )
                error = sqrt(variance / candidates)
                plans.append(
                    JointBudgetPlan(
                        block,
                        candidates,
                        rollouts,
                        cost,
                        stages,
                        error,
                        stages * error,
                        estimate.moments.candidate_count > 0,
                    )
                )
    # Leave the forecast unclipped: clipping to 1 destroys the ranking of noisy plans.
    return (
        min(
            plans,
            key=lambda plan: (
                plan.forecast_error_score,
                plan.reserved_cost,
                -plan.block_size,
                plan.candidate_count,
                plan.rollout_count,
            ),
        )
        if plans
        else None
    )

"""Next-block planners that choose the block size B, candidates M and rollouts K.

A planner only turns cost estimates and pilot moments into a plan. The caller
owns sampling: ``estimate_block(B)`` returns the planned costs of block size
``B`` with default moments, and ``measure(estimates)`` runs one independent
pilot per estimate, all in one batch, and returns their moments, ``None`` for
unusable pilots. The pilots at a prefix cut one shared pool of
``pilot_candidates`` complete outputs, so the first pilot there also pays for
the pool and a completion pilot costs nothing more.
:class:`PlanningState` describes the current prefix. Its output limit only
clips blocks; forecasts use the expected remaining length instead.

- :class:`FullHorizonPlanner` pilots every block size, adds the full remaining
  length as a completion option, and forecasts the remaining selections.
- :class:`AdaptiveBudgetController` keeps (B, M, K) between prefixes and adjusts
  them only on fresh pilot evidence, with an independent completion fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import ceil
from typing import Protocol

from inference_scaling.shared.budget.joint import (
    BlockBudgetEstimate,
    JointBudgetPlan,
    WeightMoments,
    choose_joint_budget,
)

EstimateBlock = Callable[[int], BlockBudgetEstimate]
MeasureBlocks = Callable[[Sequence[BlockBudgetEstimate]], Sequence[WeightMoments | None]]


class JointPlannerSettings(Protocol):
    """Grid, pilot and adjustment settings read by the planners."""

    @property
    def block_sizes(self) -> tuple[int, ...]: ...
    @property
    def candidate_counts(self) -> tuple[int, ...]: ...
    @property
    def rollout_counts(self) -> tuple[int, ...]: ...
    @property
    def pilot_candidates(self) -> int: ...
    @property
    def pilot_rollouts(self) -> int: ...
    @property
    def pilot_fraction(self) -> float: ...
    @property
    def relative_variance_floor(self) -> float: ...
    @property
    def initial_block_size(self) -> int | None: ...
    @property
    def initial_candidate_count(self) -> int | None: ...
    @property
    def initial_rollout_count(self) -> int | None: ...
    @property
    def adjustment_min_improvement(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PlanningState:
    """Budget and length information about the current prefix.

    ``remaining`` is the number of tokens left before the output limit; a block
    of that size completes the sequence. ``expected_remaining`` is the expected
    number of tokens until EOS, estimated from observed completions and capped
    by ``remaining``. ``finish_reserve`` is the planned cost of completing.
    """

    remaining: int
    budget: int
    finish_reserve: int
    expected_remaining: int


@dataclass(frozen=True, slots=True)
class PlanSelection:
    plan: JointBudgetPlan
    estimates: tuple[BlockBudgetEstimate, ...]
    pilot_reserved_cost: int
    remaining_budget: int
    adjustment: dict[str, object] | None = None


def pilot_cost(settings: JointPlannerSettings, estimate: BlockBudgetEstimate) -> float:
    """A pilot's planned cost beyond the shared pool: the extra completions of its cut candidates."""
    if estimate.rollout_cost == 0:
        return 0.0
    return settings.pilot_candidates * (
        estimate.branch_cost + (settings.pilot_rollouts - 1) * estimate.rollout_cost
    )


def pool_cost(settings: JointPlannerSettings, completion: BlockBudgetEstimate) -> float:
    """Planned cost of the pilots' shared pool: ``pilot_candidates`` scored complete outputs."""
    return completion.cost(settings.pilot_candidates, 0)


def _parameters(plan: JointBudgetPlan) -> tuple[int, int, int]:
    return plan.block_size, plan.candidate_count, plan.rollout_count


class FullHorizonPlanner:
    """Pilot each block size within ``pilot_fraction``, then forecast the horizon."""

    def __init__(self, settings: JointPlannerSettings):
        self.settings = settings

    def select(
        self,
        state: PlanningState,
        estimate_block: EstimateBlock,
        measure: MeasureBlocks,
    ) -> PlanSelection:
        settings = self.settings
        remaining, budget = state.remaining, state.budget
        blocks = sorted({min(value, remaining) for value in settings.block_sizes} | {remaining})
        pilot_limit = min(int(settings.pilot_fraction * budget), budget - state.finish_reserve)
        estimates = [estimate_block(block) for block in blocks]
        # The last block completes the sequence; its candidates are the pool itself.
        pool = int(pool_cost(settings, estimates[-1]))
        spent, piloted = 0, list[int]()
        for index, estimate in enumerate(estimates):
            cost = int(pilot_cost(settings, estimate)) + (0 if piloted else pool)
            if spent + cost <= pilot_limit:
                piloted.append(index)
                spent += cost
        measured = measure([estimates[index] for index in piloted]) if piloted else []
        for index, moments in zip(piloted, measured, strict=True):
            if moments is None:
                raise ValueError("pilot log-weights must be finite")
            estimates[index] = replace(estimates[index], moments=moments)
        plan = choose_joint_budget(
            estimates,
            remaining_length=remaining,
            remaining_budget=budget - spent,
            candidate_counts=settings.candidate_counts,
            rollout_counts=settings.rollout_counts,
            finish_reserve=state.finish_reserve,
            relative_variance_floor=settings.relative_variance_floor,
            forecast_length=state.expected_remaining,
        )
        if plan is None:
            raise RuntimeError("completion reservation invariant violated")
        return PlanSelection(plan, tuple(estimates), spent, budget - spent)


class AdaptiveBudgetController:
    """Evidence-gated next-chunk planning with an independent completion fallback."""

    def __init__(self, settings: JointPlannerSettings):
        block, candidates, rollouts = (
            settings.initial_block_size,
            settings.initial_candidate_count,
            settings.initial_rollout_count,
        )
        if block is None or candidates is None or rollouts is None:
            raise ValueError("adaptive planning requires initial block, candidate and rollout counts")
        self.config = settings
        self.parameters: tuple[int, int, int] = (block, candidates, rollouts)
        self.started = False
        self.neighbor_cursor = 0

    def select(
        self,
        state: PlanningState,
        estimate_block: EstimateBlock,
        measure: MeasureBlocks,
    ) -> PlanSelection:
        config = self.config
        remaining, budget, finish_reserve = state.remaining, state.budget, state.finish_reserve
        spent = 0
        # The first pilot at this prefix pays for the shared pool.
        pool = int(pool_cost(config, estimate_block(remaining)))
        decision: dict[str, object] = {"previous_parameters": self.parameters}

        def choose(
            estimate: BlockBudgetEstimate, parameters: tuple[int, int, int] | None = None
        ) -> JointBudgetPlan | None:
            return choose_joint_budget(
                [estimate], remaining_length=remaining, remaining_budget=budget,
                candidate_counts=(parameters[1],) if parameters else config.candidate_counts,
                rollout_counts=(max(1, parameters[2]),) if parameters else config.rollout_counts,
                finish_reserve=finish_reserve,
                relative_variance_floor=config.relative_variance_floor,
                forecast_full_horizon=False,
            )

        def result(
            plan: JointBudgetPlan | None, estimates: list[BlockBudgetEstimate], status: str
        ) -> PlanSelection:
            if plan is None:
                raise RuntimeError("adaptive completion reservation invariant violated")
            decision["status"] = status
            if status != "finish":
                self.parameters = _parameters(plan)
            self.started = True
            return PlanSelection(plan, tuple(estimates), spent, budget, decision)

        def measured(estimate: BlockBudgetEstimate) -> BlockBudgetEstimate | None:
            nonlocal spent, budget
            cost = int(pilot_cost(config, estimate)) + (0 if spent else pool)
            spent += cost
            budget -= cost
            moments = measure([estimate])[0]
            if moments is None or moments.candidate_count < 2:
                return None
            return replace(estimate, moments=moments)

        block = min(self.parameters[0], remaining)
        estimate = estimate_block(block)
        incumbent = choose(estimate, self.parameters) if block < remaining else None
        if incumbent is None:
            decision["finish_reason"] = (
                "remaining_within_chunk" if block == remaining else "insufficient_incumbent_budget"
            )
            completion = estimate_block(remaining)
            plan = choose(completion, (remaining, min(config.candidate_counts), 0))
            return result(plan, [completion], "finish")
        if not self.started:
            return result(incumbent, [estimate], "initial")
        if config.pilot_fraction == 0:
            return result(incumbent, [estimate], "kept_pilot_disabled")
        pilot_limit = int(config.pilot_fraction * budget)
        protected = incumbent.reserved_cost + finish_reserve
        current_cost = pilot_cost(config, estimate) + pool
        if current_cost > min(pilot_limit, budget - protected):
            return result(incumbent, [estimate], "kept_no_pilot_budget")

        blocks = sorted(set(config.block_sizes))
        position = blocks.index(block)
        neighbors = [blocks[index] for index in (position - 1, position + 1)
                     if 0 <= index < len(blocks) and blocks[index] < remaining]
        neighbor = None
        decision["neighbor_status"] = "no_neighbor"
        if neighbors:
            candidate = estimate_block(neighbors[self.neighbor_cursor % len(neighbors)])
            self.neighbor_cursor += 1
            minimum_neighbor = candidate.cost(min(config.candidate_counts), min(config.rollout_counts))
            pair_protected = max(incumbent.reserved_cost, minimum_neighbor) + finish_reserve
            decision.update(neighbor_block=candidate.block_size, neighbor_status="skipped_for_budget")
            if current_cost + pilot_cost(config, candidate) <= min(pilot_limit, budget - pair_protected):
                neighbor = candidate

        current = measured(estimate)
        if current is None:
            if neighbor is not None:
                decision["neighbor_status"] = "skipped_unscored_incumbent"
            return result(incumbent, [estimate], "kept_unscored_incumbent")
        incumbent = choose(current, self.parameters)
        if max(current.moments.relative_between, current.moments.relative_within) <= config.relative_variance_floor:
            if neighbor is not None:
                decision["neighbor_status"] = "skipped_no_variance_signal"
            return result(incumbent, [current], "kept_no_variance_signal")
        estimates = [current]
        if neighbor is not None:
            alternative = measured(neighbor)
            decision["neighbor_status"] = "measured" if alternative is not None else "unscored"
            if alternative is not None:
                estimates.append(alternative)
        incumbent = choose(current, self.parameters)
        if incumbent is None:
            raise RuntimeError("adaptive incumbent reservation invariant violated")
        horizon = max(item.block_size for item in estimates)

        def score(plan: JointBudgetPlan) -> float:
            return ceil(horizon / plan.block_size) * plan.local_error_estimate

        options = [
            choose(item, (item.block_size, candidate_count, rollout_count))
            for item in estimates
            for candidate_count in sorted(set(config.candidate_counts))
            for rollout_count in sorted(set(config.rollout_counts))
        ]
        candidates = [plan for plan in options if plan is not None]
        best = min(candidates, key=lambda plan: (score(plan), plan.reserved_cost))
        threshold = score(incumbent) * (1 - config.adjustment_min_improvement)
        eligible = [plan for plan in candidates if score(plan) < threshold]
        if eligible:
            best = min(eligible, key=lambda plan: (
                plan.reserved_cost, score(plan), -plan.block_size,
                plan.candidate_count, plan.rollout_count,
            ))
        decision.update(
            comparison_horizon=horizon, incumbent_score=score(incumbent), best_score=score(best),
            incumbent_reserved_cost=incumbent.reserved_cost,
            eligible_count=len(eligible),
            required_relative_improvement=config.adjustment_min_improvement,
            comparisons=[{"parameters": _parameters(plan), "score": score(plan),
                          "reserved_cost": plan.reserved_cost,
                          "eligible": score(plan) < threshold} for plan in candidates],
        )
        if _parameters(best) != self.parameters and score(best) < threshold:
            decision.update(
                selection_reason="cheapest_sufficient_improvement",
                selected_relative_improvement=1 - score(best) / score(incumbent),
                selected_reserved_cost=best.reserved_cost,
            )
            return result(best, estimates, "adjusted")
        decision.update(selection_reason="no_sufficient_improvement",
                        selected_relative_improvement=0.0,
                        selected_reserved_cost=incumbent.reserved_cost)
        return result(incumbent, estimates, "kept_no_improvement")


__all__ = [
    "AdaptiveBudgetController",
    "FullHorizonPlanner",
    "JointPlannerSettings",
    "PlanSelection",
    "PlanningState",
    "pilot_cost",
    "pool_cost",
]

"""Evidence-gated next-chunk planning with an independent completion fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING

from inference_scaling.shared.joint_budget import (
    BlockBudgetEstimate,
    JointBudgetPlan,
    WeightMoments,
    choose_joint_budget,
)

if TYPE_CHECKING:
    from inference_scaling.experimental.arllm.joint_budget_is import JointBudgetISConfig


@dataclass(frozen=True, slots=True)
class AdaptiveBudgetSelection:
    plan: JointBudgetPlan
    estimates: tuple[BlockBudgetEstimate, ...]
    pilot_reserved_cost: int
    remaining_budget: int
    adjustment: dict[str, object]


def _parameters(plan: JointBudgetPlan) -> tuple[int, int, int]:
    return plan.block_size, plan.candidate_count, plan.rollout_count


class AdaptiveBudgetController:
    def __init__(self, config: JointBudgetISConfig, finish_reserve: int):
        self.config = config
        self.finish_reserve = finish_reserve
        self.parameters = (
            config.initial_block_size,
            config.initial_candidate_count,
            config.initial_rollout_count,
        )
        self.started = False
        self.neighbor_cursor = 0

    def select(
        self,
        remaining: int,
        budget: int,
        estimate_block: Callable[[int], BlockBudgetEstimate],
        measure: Callable[[BlockBudgetEstimate], WeightMoments | None],
    ) -> AdaptiveBudgetSelection:
        config = self.config
        spent = 0
        decision: dict[str, object] = {"previous_parameters": self.parameters}

        def choose(estimate, parameters=None):
            return choose_joint_budget(
                [estimate], remaining_length=remaining, remaining_budget=budget,
                candidate_counts=(parameters[1],) if parameters else config.candidate_counts,
                rollout_counts=(max(1, parameters[2]),) if parameters else config.rollout_counts,
                finish_reserve=self.finish_reserve,
                relative_variance_floor=config.relative_variance_floor,
                forecast_full_horizon=False,
            )

        def result(plan, estimates, status):
            if plan is None:
                raise RuntimeError("adaptive completion reservation invariant violated")
            decision["status"] = status
            if status != "finish":
                self.parameters = _parameters(plan)
            self.started = True
            return AdaptiveBudgetSelection(plan, tuple(estimates), spent, budget, decision)

        def pilot_cost(estimate):
            return config.pilot_candidates * (
                estimate.candidate_cost + config.pilot_rollouts * estimate.rollout_cost
            )

        def measured(estimate):
            nonlocal spent, budget
            cost = int(pilot_cost(estimate))
            spent += cost
            budget -= cost
            moments = measure(estimate)
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
        protected = incumbent.reserved_cost + self.finish_reserve
        current_cost = pilot_cost(estimate)
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
            minimum_neighbor = min(config.candidate_counts) * (
                candidate.candidate_cost + min(config.rollout_counts) * candidate.rollout_cost
            )
            pair_protected = max(incumbent.reserved_cost, minimum_neighbor) + self.finish_reserve
            decision.update(neighbor_block=candidate.block_size, neighbor_status="skipped_for_budget")
            if current_cost + pilot_cost(candidate) <= min(pilot_limit, budget - pair_protected):
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

        def score(plan):
            return ceil(horizon / plan.block_size) * plan.local_error_estimate

        candidates = [choose(item) for item in estimates]
        candidates = [plan for plan in candidates if plan is not None]
        best = min(candidates, key=lambda plan: (score(plan), plan.reserved_cost))
        decision.update(
            comparison_horizon=horizon, incumbent_score=score(incumbent), best_score=score(best),
            required_relative_improvement=config.adjustment_min_improvement,
            comparisons=[{"parameters": _parameters(plan), "score": score(plan),
                          "reserved_cost": plan.reserved_cost} for plan in candidates],
        )
        if _parameters(best) != self.parameters and score(best) < score(incumbent) * (1 - config.adjustment_min_improvement):
            return result(best, estimates, "adjusted")
        return result(incumbent, estimates, "kept_no_improvement")

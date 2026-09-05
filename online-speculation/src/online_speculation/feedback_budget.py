"""R6A request-local residual correction using propensity-weighted side feedback."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections.abc import Mapping

from .tree_uno import CandidateTree, TreeBudgetController, TreeConfig


@dataclass
class _PendingDecision:
    budget: int
    probabilities: dict[int, float]
    surrogate: dict[int, float]
    limit: int
    actual_reward: int | None = None


class FeedbackBudgetController(TreeBudgetController):
    """No target mutation and no full-information or global-TPS guarantee.

    Reward updates precede the final cost timestamp, so their work is included
    in the cycle cost label. Only constant-size cost bookkeeping follows it.
    """

    def __init__(self, config: TreeConfig, *, seed: int) -> None:
        config.validate()
        if not config.feedback_budget:
            raise ValueError("feedback controller must be explicitly enabled")
        super().__init__(config)
        self.rng = random.Random(seed ^ 0x6A17B39D)
        self.residual = dict.fromkeys(self.budgets, 0.0)
        self.reward_updates = dict.fromkeys(self.budgets, 0)
        self.pending: _PendingDecision | None = None
        self.last_probabilities: dict[int, float] = {}
        self.max_update_fraction = 0.0

    def choose(self, full_tree: CandidateTree, *, remaining: int) -> tuple[int, str]:
        if self.pending is not None:
            raise RuntimeError("complete previous feedback and cost before next choice")
        if remaining < 1 or len(full_tree.nodes) < max(self.budgets):
            raise ValueError("positive remaining budget and full candidate tree required")
        limit = min(remaining, self.config.block_size + 1)
        surrogate = {n: min(float(limit), 2 + sum(x.weight for x in full_tree.nodes[1:n])) for n in self.budgets}
        initial = next((n for n in self.order if self.counts[n] < self.config.explore_each), None)
        if initial is not None:
            budget, reason = initial, "feedback_initial_probe"
            probabilities = {n: float(n == initial) for n in self.budgets}
        else:
            scores = {
                n: min(float(limit), max(1.0, surrogate[n] + self.residual[n])) / self.seconds[n]
                for n in self.budgets
            }
            exploit = max(self.order, key=lambda n: scores[n])
            if scores[exploit] < scores[self.config.nodes] * (1 + self.config.switch_margin):
                exploit = self.config.nodes
            epsilon = self.config.feedback_exploration
            probabilities = {
                n: epsilon / len(self.budgets) + (1 - epsilon) * (n == exploit)
                for n in self.budgets
            }
            uniform, cumulative = self.rng.random(), 0.0
            budget = self.order[-1]
            for n in self.order:
                cumulative += probabilities[n]
                if uniform < cumulative:
                    budget = n
                    break
            reason = "feedback_exploit" if budget == exploit else "feedback_explore"
        self.pending = _PendingDecision(budget, probabilities.copy(), surrogate, limit)
        self.last_probabilities = probabilities.copy()
        return budget, reason

    def observe_rewards(self, budget: int, visible_rewards: Mapping[int, int]) -> None:
        pending = self.pending
        if pending is None or budget != pending.budget or pending.actual_reward is not None:
            raise RuntimeError("reward feedback must match exactly one pending choice")
        visible = {n for n in self.budgets if n <= budget}
        if set(visible_rewards) != visible:
            raise ValueError("feedback must cover exactly the verified nested budgets")
        if any(not isinstance(r, int) or not 1 <= r <= pending.limit for r in visible_rewards.values()):
            raise ValueError("feedback must be a valid truncated committed length")
        ordered = [visible_rewards[n] for n in sorted(visible)]
        if ordered != sorted(ordered):
            raise ValueError("nested committed lengths must be monotone")
        # Validate all inputs before mutating any reward statistics.
        for n, reward in visible_rewards.items():
            propensity = sum(p for a, p in pending.probabilities.items() if a >= n)
            if propensity <= 0:
                raise RuntimeError("observed feedback has zero coverage probability")
            fraction = min(1.0, self.config.feedback_step_size / propensity)
            self.residual[n] += fraction * (reward - pending.surrogate[n] - self.residual[n])
            self.reward_updates[n] += 1
            self.max_update_fraction = max(self.max_update_fraction, fraction)
        pending.actual_reward = visible_rewards[budget]

    def observe(self, budget: int, *, tokens: int, seconds: float) -> None:
        pending = self.pending
        if pending is None or budget != pending.budget or pending.actual_reward != tokens:
            raise RuntimeError("complete matching reward feedback before recording cost")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("completed-cycle seconds must be finite and positive")
        super().observe(budget, tokens=tokens, seconds=seconds)
        self.pending = None

    def snapshot(self) -> dict[str, object]:
        result = super().snapshot()
        result.update({
            "selection": "current surrogate + propensity-corrected past residual / past cycle cost",
            "residual": self.residual.copy(), "reward_updates": self.reward_updates.copy(),
            "exploration_probability": self.config.feedback_exploration,
            "feedback_step_size": self.config.feedback_step_size,
            "last_action_probabilities": self.last_probabilities.copy(),
            "max_update_fraction": self.max_update_fraction,
            "pending_feedback": self.pending is not None,
            "separate_policy_rng": True, "unbiased_tps_or_regret_claim": False,
        })
        return result

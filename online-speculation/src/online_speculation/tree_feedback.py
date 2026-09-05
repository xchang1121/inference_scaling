"""Offline mathematical helpers for a future counterfactual budget experiment.

Not called by the frozen held-out runtime or the current budget controller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .tree_uno import CandidateTree, walk_target_draws


def nested_committed_lengths(
    verified_tree: CandidateTree, target_draws: Sequence[int], budgets: Sequence[int],
) -> dict[int, int]:
    if len(target_draws) != len(verified_tree.nodes):
        raise ValueError("one draw for each verified node is required")
    if any(n < 1 or n > len(verified_tree.nodes) for n in budgets):
        raise ValueError("cannot identify reward for an unverified larger tree")
    return {
        n: len(walk_target_draws(CandidateTree(verified_tree.nodes[:n]), target_draws[:n]).committed)
        for n in budgets
    }


def ips_visible_rewards(
    visible_rewards: Mapping[int, int], action_probabilities: Mapping[int, float],
) -> dict[int, float]:
    probabilities = list(action_probabilities.values())
    if (any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)):
        raise ValueError("action probabilities must form a probability distribution")
    result = {}
    for budget, reward in visible_rewards.items():
        coverage = sum(p for action, p in action_probabilities.items() if action >= budget)
        if coverage <= 0:
            raise ValueError("unobserved action has zero coverage propensity")
        result[budget] = reward / coverage
    return result

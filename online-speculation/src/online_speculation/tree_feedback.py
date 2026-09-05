"""Nested verifier feedback; only the separate R6A candidate enables learning.

The frozen R3E cost-only runtime does not call these helpers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .tree_uno import CandidateTree, TreeWalk, walk_target_draws


def nested_lengths_from_walk(
    walk: TreeWalk, budgets: Sequence[int], *, verified_nodes: int, output_limit: int,
) -> dict[int, int]:
    """Reuse one target walk; smaller prefix trees stop at the first missing edge."""
    path = walk.path_indices
    if (verified_nodes < 1 or not path or path[0] != 0
            or tuple(sorted(set(path))) != path or path[-1] >= verified_nodes):
        raise ValueError("expected a strictly ordered verified root path")
    if output_limit < 1 or len(set(budgets)) != len(budgets):
        raise ValueError("positive output limit and distinct budgets required")
    if any(n < 1 or n > verified_nodes for n in budgets):
        raise ValueError("cannot identify reward for an unverified larger tree")
    return {n: min(output_limit, 2 + sum(i < n for i in path[1:])) for n in budgets}


def nested_committed_lengths(
    verified_tree: CandidateTree, target_draws: Sequence[int], budgets: Sequence[int],
) -> dict[int, int]:
    if len(target_draws) != len(verified_tree.nodes):
        raise ValueError("one draw for each verified node is required")
    walk = walk_target_draws(verified_tree, target_draws)
    return nested_lengths_from_walk(
        walk, budgets, verified_nodes=len(verified_tree.nodes), output_limit=len(walk.committed),
    )


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

from __future__ import annotations

import itertools

import pytest

from online_speculation.tree_feedback import ips_visible_rewards, nested_committed_lengths, nested_lengths_from_walk
from online_speculation.tree_uno import CandidateTree, build_tree, walk_target_draws


def test_nested_rewards_are_monotone_under_shared_target_draws():
    tree = build_tree(2, [[0, 1], [0, 1]], [[0.5, 0.5]] * 2, nodes=7)
    for draws in itertools.product(range(2), repeat=7):
        lengths = nested_committed_lengths(tree, draws, [3, 5, 7])
        assert lengths[3] <= lengths[5] <= lengths[7]
    with pytest.raises(ValueError):
        nested_committed_lengths(tree, [0] * 7, [8])


def test_ips_coverage_estimator_is_unbiased_conditionally_on_potential_rewards():
    probability = {3: 0.2, 5: 0.3, 7: 0.5}
    potential = {3: 2, 5: 3, 7: 4}
    averages = dict.fromkeys(probability, 0.0)
    for selected, p in probability.items():
        estimates = ips_visible_rewards({n: r for n, r in potential.items() if n <= selected}, probability)
        for n, value in estimates.items():
            averages[n] += p * value
    assert averages == pytest.approx(potential)


def test_single_walk_rewards_equal_separate_walks_with_all_truncations_and_eos():
    tree = build_tree(2, [[0, 1], [0, 1]], [[0.5, 0.5]] * 2, nodes=7)
    for draws in itertools.product(range(2), repeat=7):
        walk = walk_target_draws(tree, draws)
        for remaining in range(1, 6):
            eos_limit = next((i + 1 for i, t in enumerate(walk.committed) if t == 0), len(walk.committed))
            for limit in (remaining, min(remaining, eos_limit)):
                actual = nested_lengths_from_walk(walk, [3, 5, 7], verified_nodes=7, output_limit=limit)
                for n in (3, 5, 7):
                    smaller = walk_target_draws(CandidateTree(tree.nodes[:n]), draws[:n])
                    assert actual[n] == len(smaller.committed[:limit])


def test_single_walk_refuses_unverified_feedback():
    tree = build_tree(2, [[0, 1]], [[0.5, 0.5]], nodes=3)
    walk = walk_target_draws(tree, [0, 0, 0])
    with pytest.raises(ValueError, match="unverified"):
        nested_lengths_from_walk(walk, [4], verified_nodes=3, output_limit=5)

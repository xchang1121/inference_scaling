from __future__ import annotations

import itertools

import pytest

from online_speculation.tree_feedback import ips_visible_rewards, nested_committed_lengths
from online_speculation.tree_uno import build_tree


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

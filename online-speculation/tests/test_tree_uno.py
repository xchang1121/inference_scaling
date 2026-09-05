from __future__ import annotations

import itertools
import math
from collections import defaultdict

import pytest

from online_speculation.tree_uno import (
    RankCalibrator, TreeConfig, build_tree, walk_target_draws,
)


def test_best_first_solves_finite_prefix_closed_surrogate() -> None:
    full = build_tree(9, [[0, 1], [2, 3]], [[0.6, 0.3], [0.7, 0.2]], nodes=7, include_spine=False)
    for budget in range(1, 8):
        tree = build_tree(9, [[0, 1], [2, 3]], [[0.6, 0.3], [0.7, 0.2]], nodes=budget, include_spine=False)
        scores = []
        for chosen in itertools.combinations(range(1, 7), budget - 1):
            subset = {0, *chosen}
            if all(full.nodes[i].parent in subset for i in chosen):
                scores.append(2 + sum(full.nodes[i].weight for i in chosen))
        assert tree.surrogate_committed == pytest.approx(max(scores))


def test_spine_and_parent_order_are_preserved() -> None:
    tree = build_tree(9, [[1, 2], [3, 4], [5, 6]], [[0.3, 0.2]] * 3, nodes=9)
    assert [n.token for n in tree.nodes[:4]] == [9, 1, 3, 5]
    assert tree.ancestor_indices(3) == (0, 1, 2, 3)
    for i, node in enumerate(tree.nodes[1:], 1):
        assert 0 <= node.parent < i
        assert node.weight <= tree.nodes[node.parent].weight


def _p(history):
    raw = [1 + (sum(history) + len(history) * (k + 1)) % 5 for k in range(3)]
    return [x / sum(raw) for x in raw]


@pytest.mark.parametrize("budget", [1, 2, 4, 7])
def test_tree_target_draws_match_complete_three_token_ar_law(budget) -> None:
    tree = build_tree(1, [[0, 1], [1, 2]], [[0.5, 0.3], [0.4, 0.2]], nodes=budget, include_spine=False)
    prefix = [2, 1]
    node_laws = [
        _p(prefix + [tree.nodes[a].token for a in tree.ancestor_indices(i)[1:]])
        for i in range(len(tree.nodes))
    ]
    actual = defaultdict(float)

    def continue_ar(tokens, mass):
        if len(tokens) >= 3:
            actual[tuple(tokens[:3])] += mass
            return
        for token, p in enumerate(_p(prefix + tokens)):
            continue_ar(tokens + [token], mass * p)

    for draws in itertools.product(range(3), repeat=len(tree.nodes)):
        mass = math.prod(node_laws[i][x] for i, x in enumerate(draws))
        walk = walk_target_draws(tree, draws)
        continue_ar(list(walk.committed[1:]), mass)
    for tokens in itertools.product(range(3), repeat=3):
        expected = math.prod(_p(prefix + list(tokens[:i]))[x] for i, x in enumerate(tokens))
        assert actual[tokens] == pytest.approx(expected, abs=1e-12)
    assert sum(actual.values()) == pytest.approx(1.0, abs=1e-12)


def test_missing_rank_is_not_renormalized_away_and_update_is_past_only() -> None:
    config = TreeConfig(block_size=3, top_k=2, online_rank=True, decay=1, prior_strength=2)
    state = RankCalibrator(config)
    prior = [[0.6, 0.3], [0.4, 0.2]]
    old = state.weights(prior)
    state.observe([(0, 8), (1, 4)], [[0, 1], [3, 4]])
    assert old == prior
    assert state.weights(prior)[0] == pytest.approx([0.4, 0.2])
    assert state.weights(prior)[1] == pytest.approx([0.8/3, 1.4/3])
    assert state.snapshot()["observations"] == 2


def test_invalid_sibling_labels_or_probabilities_rejected() -> None:
    with pytest.raises(ValueError):
        build_tree(0, [[1, 1]], [[0.4, 0.3]], nodes=2)
    with pytest.raises(ValueError):
        build_tree(0, [[1, 2]], [[0.8, 0.7]], nodes=2)
    with pytest.raises(ValueError):
        TreeConfig(nodes=4).validate()

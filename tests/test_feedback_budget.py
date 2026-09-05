from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from online_speculation.feedback_budget import FeedbackBudgetController
from online_speculation.tree_uno import TreeConfig, build_tree


def _config(**kwargs):
    return TreeConfig(
        block_size=3, nodes=7, top_k=2, node_budgets=(3, 5, 7),
        feedback_budget=True, explore_each=1, **kwargs,
    )


def _tree():
    return build_tree(2, [[0, 1], [0, 1]], [[0.5, 0.4]] * 2, nodes=7)


def _ready(uniform=0.1):
    controller = FeedbackBudgetController(_config(), seed=7)
    controller.counts = {n: 1 for n in controller.budgets}
    controller.seconds = {n: 1.0 for n in controller.budgets}
    controller.rng = SimpleNamespace(random=lambda: uniform)
    return controller


def test_probability_weighted_residual_innovation_matches_conditional_expectation():
    reward = {3: 2, 5: 3, 7: 4}
    expected_update = dict.fromkeys(reward, 0.0)
    surrogate = None
    for budget, mass, uniform in ((7, 0.9, 0.1), (3, 0.05, 0.925), (5, 0.05, 0.975)):
        controller = _ready(uniform)
        selected, _ = controller.choose(_tree(), remaining=20)
        assert selected == budget
        assert controller.pending.probabilities == pytest.approx({3: 0.05, 5: 0.05, 7: 0.9})
        surrogate = controller.pending.surrogate
        controller.observe_rewards(selected, {n: r for n, r in reward.items() if n <= selected})
        for n in reward:
            expected_update[n] += mass * controller.residual[n]
    assert expected_update == pytest.approx({n: 0.05 * (reward[n] - surrogate[n]) for n in reward})


def test_initial_largest_tree_provides_side_feedback_but_only_one_cost_label():
    controller = FeedbackBudgetController(_config(), seed=7)
    budget, reason = controller.choose(_tree(), remaining=30)
    assert (budget, reason) == (7, "feedback_initial_probe")
    controller.observe_rewards(7, {3: 2, 5: 3, 7: 4})
    controller.observe(7, tokens=4, seconds=1.0)
    assert controller.reward_updates == {3: 1, 5: 1, 7: 1}
    assert controller.counts == {3: 0, 5: 0, 7: 1}
    assert set(controller.seconds) == {7}
    assert not controller.snapshot()["pending_feedback"]


def test_bad_or_repeated_feedback_is_rejected_without_partial_reward_mutation():
    controller = _ready()
    budget, _ = controller.choose(_tree(), remaining=10)
    with pytest.raises(RuntimeError, match="previous"):
        controller.choose(_tree(), remaining=10)
    for feedback in ({3: 2}, {3: 2, 5: 3, 7: 5}, {3: 3, 5: 2, 7: 4}):
        with pytest.raises(ValueError):
            controller.observe_rewards(budget, feedback)
    assert controller.residual == {3: 0.0, 5: 0.0, 7: 0.0}
    with pytest.raises(RuntimeError, match="reward"):
        controller.observe(budget, tokens=4, seconds=1.0)
    controller.observe_rewards(budget, {3: 2, 5: 3, 7: 4})
    with pytest.raises(RuntimeError, match="exactly one"):
        controller.observe_rewards(budget, {3: 2, 5: 3, 7: 4})
    with pytest.raises(RuntimeError, match="matching"):
        controller.observe(budget, tokens=3, seconds=1.0)
    controller.observe(budget, tokens=4, seconds=1.0)


def test_full_support_exploration_is_bounded_and_separate_from_torch_rng():
    state = torch.random.get_rng_state().clone()
    controller = FeedbackBudgetController(_config(), seed=172)
    for i in range(500):
        budget, _ = controller.choose(_tree(), remaining=10)
        controller.observe_rewards(budget, {n: (2 if i % 2 else 4) for n in controller.budgets if n <= budget})
        controller.observe(budget, tokens=(2 if i % 2 else 4), seconds=1.0)
        assert all(abs(b) <= controller.config.block_size for b in controller.residual.values())
    assert controller.max_update_fraction <= 1.0
    assert all(p >= 0.05 - 1e-12 for p in controller.last_probabilities.values())
    assert torch.equal(state, torch.random.get_rng_state())


def test_low_propensity_configuration_cannot_hide_unstable_step_with_clipping():
    with pytest.raises(ValueError, match="propensity"):
        _config(feedback_step_size=0.06).validate()
    with pytest.raises(ValueError, match="exploration"):
        _config(feedback_exploration=0.0).validate()


def test_remaining_token_budget_caps_prediction_and_observation():
    controller = _ready()
    budget, _ = controller.choose(_tree(), remaining=1)
    assert set(controller.pending.surrogate.values()) == {1.0}
    controller.observe_rewards(budget, {n: 1 for n in controller.budgets if n <= budget})
    controller.observe(budget, tokens=1, seconds=1.0)

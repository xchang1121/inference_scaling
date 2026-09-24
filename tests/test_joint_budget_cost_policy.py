from math import ceil

import pytest

from inference_scaling.experimental.arllm.joint_budget_is import JointBudgetISConfig
from inference_scaling.shared.joint_budget import WeightMoments, choose_joint_budget
from test_joint_budget_adaptive import controller, parameters, select, settings


def test_default_enumerates_mk_and_prefers_cost_to_error_minimum():
    scheduler = controller(block_sizes=(4,))
    select(scheduler)
    choice, measured = select(scheduler, prefix=4, moments=WeightMoments(100, 0, 2))
    assert measured == [4]
    assert choice.adjustment["status"] == "adjusted"
    assert parameters(choice.plan) == (4, 8, 1)
    comparisons = choice.adjustment["comparisons"]
    lowest_error = min(comparisons, key=lambda item: item["score"])
    assert lowest_error["parameters"] == (4, 8, 4)
    assert choice.plan.reserved_cost < lowest_error["reserved_cost"]
    assert choice.plan.local_error_estimate > lowest_error["score"]
    assert len(comparisons) == 9
    assert choice.adjustment["selection_reason"] == "cheapest_sufficient_improvement"


def test_cost_policy_can_reduce_both_m_and_k_with_measured_neighbor():
    scheduler = controller()
    select(scheduler)
    choice, measured = select(scheduler, prefix=4, moments=lambda block: WeightMoments(
        100 if block == 4 else 0.001, 0, 2,
    ))
    assert measured == [4, 2]
    assert parameters(choice.plan) == (2, 2, 1)
    assert choice.adjustment["neighbor_status"] == "measured"
    assert choice.adjustment["selected_relative_improvement"] > 0.1
    assert choice.adjustment["selected_reserved_cost"] == choice.plan.reserved_cost


@pytest.mark.parametrize("budget", [600, 1500, 20000])
@pytest.mark.parametrize("between, within", [(100, 0), (0, 100), (1, 1), (0.01, 5)])
def test_cost_policy_matches_exhaustive_feasible_oracle(budget, between, within):
    scheduler = controller()
    select(scheduler)
    choice, _ = select(scheduler, prefix=4, budget=budget, moments=lambda block: WeightMoments(
        between * block / 4, within * 4 / block, 2,
    ))
    decision = choice.adjustment
    threshold = decision["incumbent_score"] * 0.9
    feasible = []
    for estimate in choice.estimates:
        for candidate_count in scheduler.config.candidate_counts:
            for rollout_count in scheduler.config.rollout_counts:
                plan = choose_joint_budget(
                    [estimate], remaining_length=28, remaining_budget=choice.remaining_budget,
                    candidate_counts=(candidate_count,), rollout_counts=(rollout_count,),
                    finish_reserve=scheduler.finish_reserve, forecast_full_horizon=False,
                )
                if plan is not None:
                    score = ceil(decision["comparison_horizon"] / plan.block_size) * plan.local_error_estimate
                    if score < threshold:
                        feasible.append((plan, score))
    assert decision["eligible_count"] == len(feasible)
    if feasible:
        expected, _ = min(feasible, key=lambda item: (
            item[0].reserved_cost, item[1], -item[0].block_size,
            item[0].candidate_count, item[0].rollout_count,
        ))
        assert choice.plan == expected
        assert decision["status"] == "adjusted"
    else:
        assert decision["status"] == "kept_no_improvement"
        assert parameters(choice.plan) == (4, 4, 2)


def test_exact_improvement_threshold_is_not_sufficient():
    scheduler = controller(
        block_sizes=(4,), candidate_counts=(4, 16),
        rollout_counts=(2,), adjustment_min_improvement=0.5,
    )
    select(scheduler)
    choice, _ = select(scheduler, prefix=4, moments=WeightMoments(4, 4, 2))
    assert choice.adjustment["best_score"] == choice.adjustment["incumbent_score"] * 0.5
    assert choice.adjustment["status"] == "kept_no_improvement"
    assert choice.adjustment["eligible_count"] == 0
    assert choice.adjustment["selected_relative_improvement"] == 0
    assert parameters(choice.plan) == (4, 4, 2)


@pytest.mark.parametrize("overrides, budget, moments, status", [
    ({"pilot_fraction": 0}, 20000, None, "kept_pilot_disabled"),
    ({"pilot_fraction": 0.001}, 20000, None, "kept_no_pilot_budget"),
    ({}, 20000, None, "kept_unscored_incumbent"),
    ({}, 20000, WeightMoments(0, 0, 2), "kept_no_variance_signal"),
    ({}, 68, None, "finish"),
])
def test_policy_preserves_initial_evidence_and_budget_guards(overrides, budget, moments, status):
    scheduler = controller(**overrides)
    initial, measured = select(scheduler)
    assert parameters(initial.plan) == (4, 4, 2)
    assert initial.adjustment["status"] == "initial" and not measured
    choice, _ = select(scheduler, prefix=4, budget=budget, moments=moments)
    assert choice.adjustment["status"] == status
    assert parameters(choice.plan) == ((28, 2, 0) if status == "finish" else (4, 4, 2))


def test_cost_policy_ignores_unscored_neighbor():
    scheduler = controller()
    select(scheduler)
    choice, measured = select(scheduler, prefix=4, moments=lambda block: (
        WeightMoments(1, 1, 2) if block == 4 else None
    ))
    assert measured == [4, 2]
    assert choice.adjustment["neighbor_status"] == "unscored"
    assert choice.plan.block_size == 4
    assert all(item["parameters"][0] == 4 for item in choice.adjustment["comparisons"])


def test_grid_order_and_duplicates_do_not_change_cost_selection():
    results = []
    for grids in ({}, {"candidate_counts": (8, 4, 2, 4), "rollout_counts": (4, 2, 1, 2)}):
        scheduler = controller(**grids)
        select(scheduler)
        results.append(select(scheduler, prefix=4, moments=WeightMoments(1, 1, 2))[0])
    assert results[0] == results[1]


def test_config_has_no_selection_policy_switch():
    assert not hasattr(JointBudgetISConfig(forward_token_budget=1000), "selection_policy")
    assert not hasattr(settings(), "selection_policy")
    with pytest.raises(TypeError, match="selection_policy"):
        settings(selection_policy="min_cost_improvement")

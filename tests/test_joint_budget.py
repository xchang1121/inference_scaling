from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import log, sqrt

import numpy as np
import pytest

from inference_scaling.shared.joint_budget import (
    BlockBudgetEstimate,
    WeightMoments,
    choose_joint_budget,
    estimate_weight_moments,
)


def test_moments_remove_inner_sampling_noise_and_are_shift_invariant():
    logs = [[log(1), log(3)], [log(3), log(5)]]
    moments = estimate_weight_moments(logs)
    assert moments.relative_between == pytest.approx(1 / 9)
    assert moments.relative_within == pytest.approx(2 / 9)
    shifted = estimate_weight_moments(
        [[value + 10000 for value in group] for group in logs]
    )
    assert shifted.relative_between == pytest.approx(moments.relative_between)
    assert shifted.relative_within == pytest.approx(moments.relative_within)


def test_terminal_and_noisy_groups():
    moments = estimate_weight_moments([[0], [log(2)]], deterministic=[True, True])
    assert moments.relative_within == 0
    assert moments.relative_between > 0
    noisy = estimate_weight_moments([[0, log(3)], [0, log(3)]])
    assert noisy.relative_between == 0
    assert noisy.relative_within > 0


@pytest.mark.parametrize(
    "groups,terminal",
    [
        ([[0, 1]], None),
        ([[0], [1]], None),
        ([[0, float("nan")], [1, 2]], None),
        ([[0, 1], [1, 2]], [True]),
        ([[0, 1], [1, 2]], [True, False]),
    ],
)
def test_invalid_pilot_data(groups, terminal):
    with pytest.raises(ValueError):
        estimate_weight_moments(groups, deterministic=terminal)


def plan_for(estimates, **kwargs):
    return choose_joint_budget(
        estimates,
        remaining_length=8,
        remaining_budget=400,
        candidate_counts=(2, 4, 8, 16),
        rollout_counts=(1, 2, 4, 8),
        **kwargs,
    )


def test_joint_choice_changes_width_replication_and_block():
    broad = BlockBudgetEstimate(4, WeightMoments(100, 0), 5, 5)
    noisy = replace(broad, moments=WeightMoments(0, 100))
    width = plan_for([broad])
    replication = plan_for([noisy])
    assert width is not None and replication is not None
    assert width.candidate_count > replication.candidate_count
    assert width.rollout_count < replication.rollout_count
    short = BlockBudgetEstimate(2, WeightMoments(0.001, 0.001), 2, 1)
    long = BlockBudgetEstimate(8, WeightMoments(100, 0), 8, 0)
    assert plan_for([short, long]).block_size == 2
    long = replace(long, moments=WeightMoments(0, 0))
    assert plan_for([short, long]).block_size == 8
    assert plan_for([short, long]).rollout_count == 0


def test_completion_reserve_and_infeasibility():
    estimate = BlockBudgetEstimate(4, WeightMoments(1, 1), 10, 1)
    plan = plan_for([estimate], finish_reserve=360)
    assert plan is not None and plan.reserved_cost + 360 <= 400
    assert plan_for([estimate], finish_reserve=400) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remaining_length": 0},
        {"remaining_budget": float("nan")},
        {"candidate_counts": (1,)},
        {"rollout_counts": (0,)},
        {"candidate_counts": (2.5,)},
        {"relative_variance_floor": -1},
        {"forecast_full_horizon": 1},
    ],
)
def test_invalid_planning_inputs(kwargs):
    inputs = dict(
        remaining_length=2,
        remaining_budget=100,
        candidate_counts=(2,),
        rollout_counts=(1,),
    )
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        choose_joint_budget(
            [BlockBudgetEstimate(1, WeightMoments(1, 1), 1, 1)], **inputs
        )


def test_next_step_planning_does_not_treat_output_cap_as_forecast_length():
    estimate = BlockBudgetEstimate(400, WeightMoments(1, 1), 2857, 132864)
    settings = dict(
        remaining_length=130407, remaining_budget=3463530,
        candidate_counts=(2, 4, 8), rollout_counts=(1, 2, 4), finish_reserve=265728,
    )
    assert choose_joint_budget([estimate], **settings) is None
    plan = choose_joint_budget([estimate], **settings, forecast_full_horizon=False)
    assert plan is not None
    assert plan.block_size == 400
    assert plan.rollout_count > 0
    assert plan.forecast_steps == 1
    assert plan.forecast_error_score == plan.local_error_estimate
    assert plan.reserved_cost + settings["finish_reserve"] <= settings["remaining_budget"]


def test_next_step_planning_still_requires_completion_reserve():
    estimate = BlockBudgetEstimate(100, WeightMoments(1, 1), 102, 1002)
    settings = dict(
        remaining_length=1000, candidate_counts=(2,), rollout_counts=(1,),
        finish_reserve=2004, forecast_full_horizon=False,
    )
    assert choose_joint_budget([estimate], remaining_budget=4211, **settings) is None
    assert choose_joint_budget([estimate], remaining_budget=4212, **settings) is not None


def test_next_step_planning_changes_m_and_k_with_measured_moments():
    settings = dict(
        remaining_length=1000, remaining_budget=900, candidate_counts=(2, 4, 8),
        rollout_counts=(1, 2, 4), forecast_full_horizon=False,
    )
    between = choose_joint_budget(
        [BlockBudgetEstimate(100, WeightMoments(100, 0, 2), 100, 10)], **settings,
    )
    within = choose_joint_budget(
        [BlockBudgetEstimate(100, WeightMoments(0, 100, 2), 100, 10)], **settings,
    )
    assert (between.candidate_count, between.rollout_count) == (8, 1)
    assert (within.candidate_count, within.rollout_count) == (4, 4)
    assert between.used_pilot and within.used_pilot


def test_next_step_forecast_does_not_change_terminal_definition():
    plan = choose_joint_budget(
        [BlockBudgetEstimate(1000, WeightMoments(1, 0), 1002, 0)],
        remaining_length=1000, remaining_budget=2004, candidate_counts=(2,),
        rollout_counts=(1, 2), finish_reserve=2004, forecast_full_horizon=False,
    )
    assert plan is not None and plan.rollout_count == 0


def test_next_step_planning_rejects_mixed_block_horizons():
    estimates = [
        BlockBudgetEstimate(100, WeightMoments(1, 1), 2557, 132864),
        BlockBudgetEstimate(130407, WeightMoments(1, 0), 132864, 0),
    ]
    with pytest.raises(ValueError, match="single block"):
        choose_joint_budget(
            estimates, remaining_length=130407, remaining_budget=3463530,
            candidate_counts=(2, 4, 8), rollout_counts=(1, 2, 4),
            finish_reserve=265728, forecast_full_horizon=False,
        )


def test_finite_sir_tv_bound_by_exact_enumeration():
    # p(z)=p(u|z)=1/2; G(z,u) is positive. Enumerate all candidate/rollout pools.
    weights = np.asarray([[1.0, 3.0], [2.0, 8.0]])
    conditional = weights.mean(axis=1)
    mean = weights.mean()
    between = conditional.var()
    within = weights.var(axis=1).mean()
    target = conditional / conditional.sum()
    for candidates, rollouts in ((2, 1), (2, 2), (3, 2)):
        output = np.zeros(2)
        for flat in product((0, 1), repeat=candidates * (rollouts + 1)):
            pool = np.asarray(flat).reshape(candidates, rollouts + 1)
            z = pool[:, 0]
            estimates = np.asarray([weights[row[0], row[1:]].mean() for row in pool])
            probabilities = estimates / estimates.sum()
            for index in (0, 1):
                output[index] += probabilities[z == index].sum() / 2 ** len(flat)
        tv = np.abs(output - target).sum() / 2
        assert tv <= sqrt((between + within / rollouts) / (candidates * mean**2))

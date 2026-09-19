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

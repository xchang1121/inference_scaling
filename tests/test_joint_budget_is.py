from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.experimental.arllm.joint_budget_is import (
    JointBudgetISConfig,
    block_costs,
    run_joint_budget_is,
)
from inference_scaling.shared.rng import SeedStream


class RecordingBackend(TabularAutoregressiveBackend):
    def __init__(self, probabilities=(0.65, 0.35)):
        super().__init__({}, fallback=probabilities)
        self.requests = []

    def sample_batch(self, requests):
        self.requests.extend(requests)
        return super().sample_batch(requests)


def test_budget_includes_pilots_and_independent_production_samples():
    backend = RecordingBackend()
    result = run_joint_budget_is(
        backend,
        (),
        JointBudgetISConfig(
            forward_token_budget=400,
            total_length=4,
            block_sizes=(1, 2),
            candidate_counts=(2, 4),
            rollout_counts=(1, 2),
            pilot_fraction=0.4,
            reward_forward_passes=0,
        ),
        lambda _p, y: float(sum(y)),
        SeedStream(42),
    )
    assert len(result.token_ids) == 4
    assert result.pilot_reserved_forward_tokens > 0
    assert result.reserved_forward_tokens <= 400
    assert result.reserved_forward_tokens == sum(
        step.pilot_reserved_cost + step.plan.reserved_cost for step in result.steps
    )
    assert len({request.seed for request in backend.requests}) == len(backend.requests)
    # Stored statistical observations are production-only, including terminal scoring.
    for step in result.steps:
        assert len(step.evaluation.candidates) == step.plan.candidate_count
        assert all(
            len(candidate.rollouts) == max(1, step.plan.rollout_count)
            for candidate in step.evaluation.candidates
        )
    actual_reserved_generation = sum(
        len(r.prefix) + r.max_new_tokens for r in backend.requests
    )
    assert actual_reserved_generation == result.reserved_forward_tokens


def test_multistep_plan_recomputes_budget_and_preserves_fixed_horizon(monkeypatch):
    from inference_scaling.experimental.arllm import joint_budget_is as driver

    original = driver.choose_joint_budget

    def short_blocks(estimates, **kwargs):
        return original(estimates[:1], **kwargs) or original(estimates, **kwargs)

    monkeypatch.setattr(driver, "choose_joint_budget", short_blocks)
    result = run_joint_budget_is(
        RecordingBackend(),
        (),
        JointBudgetISConfig(
            forward_token_budget=1000,
            total_length=8,
            block_sizes=(1, 2),
            candidate_counts=(2, 4),
            rollout_counts=(1, 2),
            pilot_fraction=0.8,
            reward_forward_passes=0,
        ),
        lambda _p, _y: 0.0,
        SeedStream(3),
    )
    assert len(result.steps) > 1
    before = 1000
    length = 0
    for step in result.steps:
        assert step.evaluation.generated_length_before == length
        length += len(step.evaluation.selected.token_ids)
        before -= step.pilot_reserved_cost + step.plan.reserved_cost
        assert step.remaining_budget == before >= 0


def test_initial_insufficient_budget_has_no_side_effects():
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="at least"):
        run_joint_budget_is(
            backend,
            (),
            JointBudgetISConfig(forward_token_budget=7, total_length=4),
            lambda _p, _y: pytest.fail("reward called"),
            SeedStream(1),
        )
    assert not backend.requests


def test_terminal_fallback_and_early_eos():
    config = JointBudgetISConfig(forward_token_budget=16, total_length=4)
    result = run_joint_budget_is(
        RecordingBackend(), (), config, lambda _p, _y: 0.0, SeedStream(8)
    )
    assert len(result.steps) == 1
    assert result.steps[0].plan.block_size == 4
    assert result.steps[0].plan.rollout_count == 0
    assert result.pilot_reserved_forward_tokens == 0
    assert not result.steps[0].plan.used_pilot
    eos = run_joint_budget_is(
        RecordingBackend((0, 1)),
        (),
        replace(config, forward_token_budget=300),
        lambda _p, _y: 0.0,
        SeedStream(8),
        sampling=SamplingConfig(eos_token_id=1),
    )
    assert eos.token_ids == (1,) and eos.stopping_reason == "eos"


def test_constant_reward_preserves_base_and_repeated_seed_is_identical():
    config = JointBudgetISConfig(
        forward_token_budget=100,
        total_length=2,
        block_sizes=(1,),
        candidate_counts=(2, 4),
        rollout_counts=(1, 2),
        pilot_fraction=0.5,
        reward_forward_passes=0,
    )
    backend = RecordingBackend()
    outputs = [
        run_joint_budget_is(
            backend, (), config, lambda _p, _y: 0.0, SeedStream(i)
        ).token_ids
        for i in range(500)
    ]
    assert np.mean([y[0] for y in outputs]) == pytest.approx(0.35, abs=0.07)
    assert np.mean([y[1] for y in outputs]) == pytest.approx(0.35, abs=0.07)
    assert (
        outputs[0]
        == run_joint_budget_is(
            backend, (), config, lambda _p, _y: 0.0, SeedStream(0)
        ).token_ids
    )


def test_full_sequence_sir_approaches_reward_target():
    config = JointBudgetISConfig(
        forward_token_budget=64,
        total_length=1,
        candidate_counts=(64,),
        pilot_fraction=0,
        reward_forward_passes=0,
    )
    outputs = [
        run_joint_budget_is(
            RecordingBackend((0.5, 0.5)),
            (),
            config,
            lambda _p, y: np.log(3) * y[0],
            SeedStream(i),
        ).token_ids[0]
        for i in range(500)
    ]
    assert np.mean(outputs) == pytest.approx(0.75, abs=0.06)


def test_reward_cost_and_support_checks():
    assert block_costs(
        prompt_length=3,
        generated_length=2,
        total_length=8,
        block_size=2,
        reward_forward_passes=1,
    ) == (7, 22)
    assert block_costs(
        prompt_length=3,
        generated_length=2,
        total_length=8,
        block_size=6,
        reward_forward_passes=1,
    ) == (22, 0)
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="full-support"):
        run_joint_budget_is(
            backend,
            (),
            JointBudgetISConfig(forward_token_budget=100, total_length=2),
            lambda _p, _y: 0.0,
            SeedStream(0),
            sampling=SamplingConfig(top_p=0.9),
        )
    assert not backend.requests


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pilot_fraction": float("nan")},
        {"pilot_fraction": 1},
        {"pilot_candidates": 1},
        {"pilot_rollouts": 1},
        {"reward_temperature": 0},
        {"reward_forward_passes": -1},
        {"block_sizes": ()},
        {"candidate_counts": (1,)},
        {"rollout_counts": (0,)},
        {"relative_variance_floor": 0},
        {"total_length": 2.5},
        {"forward_token_budget": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        JointBudgetISConfig(**({"forward_token_budget": 100} | kwargs))

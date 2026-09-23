from dataclasses import replace

import pytest

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.experimental.arllm.adaptive_budget import AdaptiveBudgetController
from inference_scaling.experimental.arllm.joint_budget_is import (
    JointBudgetISConfig,
    block_costs,
    run_joint_budget_is,
)
from inference_scaling.shared.joint_budget import BlockBudgetEstimate, WeightMoments
from inference_scaling.shared.rng import SeedStream
from test_joint_budget_is import RecordingBackend


def settings(**overrides):
    return JointBudgetISConfig(**({
        "forward_token_budget": 20000, "total_length": 32,
        "block_sizes": (2, 4, 8), "candidate_counts": (2, 4, 8), "rollout_counts": (1, 2, 4),
        "planning_mode": "chunk_adaptive", "initial_block_size": 4,
        "initial_candidate_count": 4, "initial_rollout_count": 2,
        "pilot_fraction": 0.6, "reward_forward_passes": 0,
    } | overrides))


def select(controller, prefix=0, budget=None, moments=None):
    config = controller.config
    measured = []

    def estimate(block):
        costs = block_costs(prompt_length=2, generated_length=prefix,
                            total_length=config.total_length, block_size=block,
                            reward_forward_passes=config.reward_forward_passes)
        return BlockBudgetEstimate(block, WeightMoments(1, 0 if costs[1] == 0 else 1), *costs)

    def measure(estimate):
        measured.append(estimate.block_size)
        return moments(estimate.block_size) if callable(moments) else moments

    selection = controller.select(
        config.total_length - prefix,
        config.forward_token_budget if budget is None else budget,
        estimate, measure,
    )
    assert selection.remaining_budget == (
        (config.forward_token_budget if budget is None else budget) - selection.pilot_reserved_cost
    )
    protected = controller.finish_reserve if selection.plan.rollout_count else 0
    assert selection.plan.reserved_cost + protected <= selection.remaining_budget
    return selection, measured


def controller(**overrides):
    config = settings(**overrides)
    reserve = min(config.candidate_counts) * (2 + config.total_length) * (1 + config.reward_forward_passes)
    return AdaptiveBudgetController(config, reserve)


def parameters(plan):
    return plan.block_size, plan.candidate_count, plan.rollout_count


def test_first_chunk_uses_presets_without_pilot_or_terminal_competition():
    scheduler = controller()
    selection, measured = select(scheduler)
    assert parameters(selection.plan) == (4, 4, 2)
    assert selection.adjustment["status"] == "initial"
    assert not selection.plan.used_pilot and not measured


@pytest.mark.parametrize("fraction, status", [(0, "kept_pilot_disabled"), (0.001, "kept_no_pilot_budget")])
def test_no_evidence_keeps_all_three_parameters(fraction, status):
    scheduler = controller(pilot_fraction=fraction)
    select(scheduler)
    selection, measured = select(scheduler, prefix=4)
    assert parameters(selection.plan) == (4, 4, 2)
    assert selection.adjustment["status"] == status
    assert not measured


@pytest.mark.parametrize("moments, status", [
    (None, "kept_unscored_incumbent"),
    (WeightMoments(1, 1), "kept_unscored_incumbent"),
    (WeightMoments(0, 0, 2), "kept_no_variance_signal"),
])
def test_missing_or_flat_pilot_does_not_tune(moments, status):
    scheduler = controller()
    select(scheduler)
    selection, measured = select(scheduler, prefix=4, moments=moments)
    assert parameters(selection.plan) == (4, 4, 2)
    assert selection.adjustment["status"] == status
    assert measured == [4]
    assert selection.pilot_reserved_cost > 0


def test_current_only_pilot_changes_mk_but_not_b():
    scheduler = controller()
    select(scheduler)
    selection, measured = select(scheduler, prefix=4, budget=600, moments=WeightMoments(100, 0, 2))
    assert measured == [4]
    assert selection.adjustment["neighbor_status"] == "skipped_for_budget"
    assert selection.adjustment["status"] == "adjusted"
    assert parameters(selection.plan) == (4, 8, 1)


def test_paired_pilot_can_change_b_to_an_adjacent_value():
    scheduler = controller()
    select(scheduler)
    selection, measured = select(
        scheduler, prefix=4,
        moments=lambda block: WeightMoments(10 if block == 4 else 0.01, 0, 2),
    )
    assert measured == [4, 2]
    assert selection.adjustment["status"] == "adjusted"
    assert parameters(selection.plan) == (2, 8, 4)
    assert selection.adjustment["comparison_horizon"] == 4


def test_unscored_neighbor_cannot_change_b():
    scheduler = controller()
    select(scheduler)
    selection, measured = select(
        scheduler, prefix=4, moments=lambda block: WeightMoments(1, 1, 2) if block == 4 else None,
    )
    assert measured == [4, 2]
    assert selection.plan.block_size == 4
    assert selection.adjustment["neighbor_status"] == "unscored"


def test_neighbor_probes_alternate_and_can_increase_b_without_skipping_grid_values():
    scheduler = controller(block_sizes=(2, 4, 8, 16), candidate_counts=(4,), rollout_counts=(2,))
    select(scheduler)
    selection, measured = select(scheduler, prefix=4, moments=WeightMoments(1, 1, 2))
    assert measured == [4, 2]
    assert selection.adjustment["status"] == "kept_no_improvement"
    selection, measured = select(
        scheduler, prefix=8,
        moments=lambda block: WeightMoments(1 if block == 4 else 0.001, 0, 2),
    )
    assert measured == [4, 8]
    assert selection.adjustment["status"] == "adjusted"
    assert parameters(selection.plan) == (8, 4, 2)
    assert selection.adjustment["comparison_horizon"] == 8


def test_old_pilot_is_not_reused_when_next_prefix_cannot_afford_new_evidence():
    scheduler = controller(block_sizes=(4,))
    select(scheduler)
    adjusted, _ = select(scheduler, prefix=4, moments=WeightMoments(1, 1, 2))
    assert parameters(adjusted.plan) == (4, 8, 4)
    selection, measured = select(scheduler, prefix=8, budget=1300)
    assert parameters(selection.plan) == (4, 8, 4)
    assert selection.adjustment["status"] == "kept_no_pilot_budget"
    assert not selection.plan.used_pilot and not measured


def test_threshold_keeps_parameters_when_improvement_is_too_small():
    scheduler = controller(adjustment_min_improvement=0.9)
    select(scheduler)
    selection, _ = select(scheduler, prefix=4, moments=WeightMoments(1, 1, 2))
    assert selection.adjustment["status"] == "kept_no_improvement"
    assert parameters(selection.plan) == (4, 4, 2)


@pytest.mark.parametrize("budget, status", [(379, "finish"), (380, "kept_no_pilot_budget")])
def test_incumbent_budget_boundary_is_exact(budget, status):
    scheduler = controller()
    select(scheduler)
    selection, measured = select(scheduler, prefix=4, budget=budget)
    assert selection.adjustment["status"] == status
    assert not measured
    if status == "finish":
        assert parameters(selection.plan) == (28, 2, 0)
        assert selection.adjustment["finish_reason"] == "insufficient_incumbent_budget"
        assert scheduler.parameters == (4, 4, 2)


def test_output_boundary_finishes_without_extra_pilot():
    scheduler = controller()
    selection, measured = select(scheduler, prefix=30, budget=68)
    assert parameters(selection.plan) == (2, 2, 0)
    assert selection.adjustment["finish_reason"] == "remaining_within_chunk"
    assert not measured


def test_real_driver_130k_cap_uses_initial_chunk_not_full_remaining():
    backend = RecordingBackend(probabilities=(1.0, 0.0))
    config = settings(
        total_length=130407, forward_token_budget=4000000, block_sizes=(100, 200, 400),
        initial_block_size=100, pilot_fraction=0.15,
    )
    result = run_joint_budget_is(
        backend, (1,) * 2457, config, lambda _prompt, _tokens: 0.0, SeedStream(42),
        sampling=SamplingConfig(eos_token_id=0),
    )
    assert parameters(result.steps[0].plan) == (100, 4, 2)
    assert all(request.max_new_tokens == 100 for request in backend.requests)
    assert result.pilot_reserved_forward_tokens == 0
    assert result.stopping_reason == "eos"


@pytest.mark.parametrize("reward_passes", [0, 1, 2])
def test_real_driver_multichunk_and_completion_accounting(reward_passes):
    backend = RecordingBackend()
    config = settings(
        forward_token_budget=750 * (1 + reward_passes), pilot_fraction=0,
        reward_forward_passes=reward_passes,
    )
    result = run_joint_budget_is(backend, (1, 1), config, lambda _prompt, _tokens: 0.0, SeedStream(4))
    assert len(result.token_ids) == config.total_length
    assert len(result.steps) > 1
    assert parameters(result.steps[0].plan) == (4, 4, 2)
    assert result.steps[-1].adjustment["status"] == "finish"
    before = config.forward_token_budget
    for step in result.steps:
        before -= step.pilot_reserved_cost + step.plan.reserved_cost
        assert step.remaining_budget == before >= 0
        assert step.plan.forecast_steps == 1
        if step.adjustment["status"] != "finish":
            assert parameters(step.plan) == (4, 4, 2)
    actual = sum(len(request.prefix) + request.max_new_tokens for request in backend.requests)
    actual += sum(
        len(candidate.rollouts) * (2 + config.total_length) * reward_passes
        for step in result.steps for candidate in step.evaluation.candidates
    )
    assert actual == result.reserved_forward_tokens <= config.forward_token_budget
    assert len({request.seed for request in backend.requests}) == len(backend.requests)


def test_pilots_and_parameters_reset_for_each_run():
    config = settings(total_length=12, block_sizes=(2,), initial_block_size=2)
    results = [run_joint_budget_is(
        RecordingBackend(), (), config, lambda _prompt, tokens: float(sum(tokens)), SeedStream(0),
    ) for _ in range(2)]
    assert results[0] == results[1]
    assert all(parameters(result.steps[0].plan) == (2, 4, 2) for result in results)
    assert results[0].pilot_reserved_forward_tokens > 0
    assert any(step.adjustment["status"] == "adjusted" for step in results[0].steps)
    assert any(step.plan.used_pilot for step in results[0].steps)


def test_real_pilots_are_charged_but_never_reused_as_production_samples():
    backend = RecordingBackend()
    config = settings(total_length=12, block_sizes=(2,), initial_block_size=2, reward_forward_passes=1)
    scored = []

    def reward(_prompt, tokens):
        scored.append(tokens)
        return 0.0

    result = run_joint_budget_is(backend, (1, 1), config, reward, SeedStream(8))
    actual = sum(len(request.prefix) + request.max_new_tokens for request in backend.requests)
    actual += len(scored) * (2 + config.total_length)
    assert result.reserved_forward_tokens == actual
    assert result.pilot_reserved_forward_tokens > 0
    assert result.pilot_reserved_forward_tokens == sum(step.pilot_reserved_cost for step in result.steps)
    assert all(len(step.evaluation.candidates) == step.plan.candidate_count for step in result.steps)
    assert len({request.seed for request in backend.requests}) == len(backend.requests)
    assert result.reserved_forward_tokens <= config.forward_token_budget


@pytest.mark.parametrize("overrides", [
    {"planning_mode": "unknown"}, {"initial_block_size": None}, {"initial_block_size": 3},
    {"initial_candidate_count": 1}, {"initial_rollout_count": 0},
    {"initial_rollout_count": True}, {"adjustment_min_improvement": 0},
    {"adjustment_min_improvement": 1}, {"adjustment_min_improvement": float("nan")},
])
def test_invalid_adaptive_config(overrides):
    with pytest.raises(ValueError):
        settings(**overrides)


def test_full_horizon_does_not_silently_ignore_adaptive_settings():
    with pytest.raises(ValueError, match="requires chunk_adaptive"):
        replace(settings(), planning_mode="full_horizon")
    with pytest.raises(ValueError, match="requires chunk_adaptive"):
        JointBudgetISConfig(forward_token_budget=1000, adjustment_min_improvement=0.2)

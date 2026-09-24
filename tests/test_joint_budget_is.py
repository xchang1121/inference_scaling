from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import SequenceSample
from inference_scaling.arllm.algorithms.joint_budget_is import (
    JointBudgetISConfig,
    run_joint_budget_is,
)
from inference_scaling.shared.budget.costs import block_costs
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
    # Without EOS every request decodes its limit, so the realized cost is the
    # planned cost plus the length probe that measured the output length.
    actual_generation = sum(len(r.prefix) + r.max_new_tokens for r in backend.requests)
    assert actual_generation == result.actual_forward_tokens <= 400
    assert result.length_probe_forward_tokens == 4
    assert result.actual_forward_tokens == result.reserved_forward_tokens + 4


def test_multistep_plan_recomputes_budget_and_preserves_fixed_horizon(monkeypatch):
    from inference_scaling.shared.budget import planners

    original = planners.choose_joint_budget

    def short_blocks(estimates, **kwargs):
        return original(estimates[:1], **kwargs) or original(estimates, **kwargs)

    monkeypatch.setattr(planners, "choose_joint_budget", short_blocks)
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
    before = 1000 - result.length_probe_forward_tokens
    length = 0
    for step in result.steps:
        assert step.evaluation.generated_length_before == length
        length += len(step.evaluation.selected.token_ids)
        before -= step.pilot_actual_cost + step.actual_cost
        assert step.remaining_budget == before >= 0


def test_initial_insufficient_budget_has_no_side_effects():
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="at least"):
        run_joint_budget_is(
            backend,
            (),
            JointBudgetISConfig(forward_token_budget=7, total_length=4, expected_output_tokens=4),
            lambda _p, _y: pytest.fail("reward called"),
            SeedStream(1),
        )
    assert not backend.requests


def test_length_probe_is_the_only_call_before_an_insufficient_budget_error():
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="at least"):
        run_joint_budget_is(
            backend,
            (),
            JointBudgetISConfig(forward_token_budget=7, total_length=4),
            lambda _p, _y: pytest.fail("reward called"),
            SeedStream(1),
        )
    assert [request.request_id for request in backend.requests] == ["joint-budget-is:length-probe"]


def test_terminal_fallback_and_early_eos():
    config = JointBudgetISConfig(forward_token_budget=16, total_length=4, expected_output_tokens=4)
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
        expected_output_tokens=1,
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
    def costs(block, *, total_length=8, expected=6):
        return block_costs(
            prompt_length=3, generated_length=2, total_length=total_length,
            block_size=block, reward_forward_passes=1, expected_remaining=expected,
        )

    # An expected completion reaching the output limit prices every rollout at it.
    assert costs(2) == (7, 22)
    assert costs(6) == (22, 0)
    assert costs(2, expected=100) == (7, 22)
    # Otherwise the limit does not enter the cost: rollouts end at the expected EOS.
    assert costs(2, expected=3) == costs(2, total_length=10_000, expected=3) == (7, 16)
    assert costs(4, expected=3) == (9, 18)
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
        {"expected_output_tokens": 0},
        {"expected_output_tokens": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        JointBudgetISConfig(**({"forward_token_budget": 100} | kwargs))


class BoundedLengthBackend:
    """Every request ends with EOS after 30-90 tokens, whatever its token limit."""

    model_id = "bounded"

    def sample_batch(self, requests):
        samples = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            tokens = tuple(rng.integers(1, 9, size=int(rng.integers(30, 91)) - 1).tolist()) + (0,)
            tokens = tokens[: request.max_new_tokens]
            samples.append(SequenceSample(
                request.prefix, tokens, (-1.0,) * len(tokens), request.sampling.policy_id,
                self.model_id, request.request_id, "eos" if tokens[-1] == 0 else "length",
            ))
        return samples

    def score_batch(self, requests):
        raise AssertionError("on-policy runs never rescore")


@pytest.mark.parametrize("options", [
    dict(planning_mode="chunk_adaptive", block_sizes=(4, 8, 16), candidate_counts=(2, 4),
         rollout_counts=(1, 2), initial_block_size=8, initial_candidate_count=2,
         initial_rollout_count=1, pilot_fraction=0.3),
    dict(block_sizes=(4, 8, 16), candidate_counts=(2, 4, 8), rollout_counts=(1, 2, 4)),
])
def test_plans_do_not_depend_on_an_output_limit_that_is_never_reached(options):
    def run(total_length):
        return run_joint_budget_is(
            BoundedLengthBackend(), (5,) * 10,
            JointBudgetISConfig(forward_token_budget=12_000, total_length=total_length, **options),
            lambda _p, y: float(np.mean(y[:20])) / 4, SeedStream(6),
            sampling=SamplingConfig(eos_token_id=0),
        )

    def trace(result):
        # A completion step has no chunk length: it runs to EOS under the limit.
        return [
            (step.plan.block_size if step.plan.rollout_count else "to_eos",
             step.plan.candidate_count, step.plan.rollout_count, step.adjustment,
             step.expected_remaining, step.pilot_actual_cost, step.actual_cost)
            for step in result.steps
        ], result.token_ids, result.actual_forward_tokens

    traces = [trace(run(total_length)) for total_length in (2_048, 16_384, 1_048_576)]
    assert traces[0] == traces[1] == traces[2]
    if options.get("planning_mode") == "chunk_adaptive":
        assert any(isinstance(block, int) for block, *_ in traces[0][0])

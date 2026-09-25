from __future__ import annotations

from math import log

import pytest

from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.stepwise import (
    StepwiseCandidate,
    normalize_log_weights,
    run_stepwise_generation,
)


class BinaryStepwiseBackend:
    initial_state: tuple[int, ...] = ()

    def is_terminal(self, state):
        return len(state) == 2

    def propose(self, state, step_index, seeds):
        del state, step_index, seeds
        return (0, 1)

    def evaluate(self, state, proposals, step_index, seeds):
        del step_index, seeds
        return tuple(
            StepwiseCandidate(state + (proposal,), float(proposal))
            for proposal in proposals
        )

    def advance(self, state, selected, step_index):
        del state, step_index
        return selected


def test_common_stepwise_driver_is_state_and_model_agnostic():
    first = run_stepwise_generation(
        BinaryStepwiseBackend(), SeedStream(7), selection_namespace=("test",)
    )
    second = run_stepwise_generation(
        BinaryStepwiseBackend(), SeedStream(7), selection_namespace=("test",)
    )

    assert first == second
    assert len(first.final_state) == 2
    assert len(first.steps) == 2
    assert all(len(step.candidates) == 2 for step in first.steps)
    assert all(sum(step.probabilities) == pytest.approx(1.0) for step in first.steps)


def test_log_weight_normalization_matches_softmax():
    probabilities = normalize_log_weights((0.0, log(3.0)))
    assert probabilities == pytest.approx((0.25, 0.75))

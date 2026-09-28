"""Tests for diffusion-language-model sampling configuration."""

from __future__ import annotations

import pytest

from inference_scaling.dllm.config import DiffusionSamplingConfig, diffusion_decision_stage_lengths
from inference_scaling.dllm.types import DiffusionGenerationRequest, DiffusionSample, DiffusionTraceStep

VALID = {"block_length": 4, "steps_per_block": 4, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "remasking": "random"}


def test_decision_stages_preserve_complete_llada_blocks():
    sampling = DiffusionSamplingConfig(**VALID | {"steps_per_block": 2})
    assert diffusion_decision_stage_lengths(total_length=96, decision_block_size=48, sampling=sampling) == (48, 48)


def test_generation_length_must_contain_complete_llada_blocks():
    sampling = DiffusionSamplingConfig(**VALID | {"temperature": 0.0, "remasking": "low_confidence"})
    DiffusionGenerationRequest((1, 2, 3, 4, 5), 8, sampling, 0, "aligned")
    with pytest.raises(ValueError, match="generation_length"):
        DiffusionGenerationRequest((1, 2, 3, 4, 5), 7, sampling, 0, "split")


def test_diffusion_policy_id_and_float_validation_are_exact() -> None:
    assert (DiffusionSamplingConfig(**VALID | {"temperature": 1.0000001}).policy_id
            != DiffusionSamplingConfig(**VALID | {"temperature": 1.0000002}).policy_id)
    for kwargs in ({"temperature": float("nan")}, {"top_p": float("inf")}):
        with pytest.raises(ValueError):
            DiffusionSamplingConfig(**(VALID | kwargs))


def test_exact_diffusion_trajectory_requires_a_complete_finite_trace() -> None:
    with pytest.raises(ValueError, match="complete trace"):
        DiffusionSample((), (1,), (), 0.0, "policy", "model", "request")
    with pytest.raises(ValueError, match="finite"):
        DiffusionTraceStep(0, 0, (0,), (1,), float("nan"))

"""Tests for diffusion-language-model sampling configuration."""

from __future__ import annotations

import pytest

from inference_scaling.dllm.config import (
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.dllm.types import (
    DiffusionSample,
    DiffusionTraceStep,
)


def test_decision_stages_preserve_complete_llada_blocks():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=2,
        temperature=1.0,
        remasking="random", top_k=0, top_p=1.0, cfg_scale=0.0,
    )

    lengths = diffusion_decision_stage_lengths(
        total_length=96,
        decision_block_size=48,
        sampling=sampling,
    )

    assert lengths == (48, 48)


def test_generation_length_must_contain_complete_llada_blocks():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, remasking="low_confidence",
    )

    DiffusionGenerationRequest((1, 2, 3, 4, 5), 8, sampling, 0, "aligned")
    with pytest.raises(ValueError, match="generation_length"):
        DiffusionGenerationRequest((1, 2, 3, 4, 5), 7, sampling, 0, "split")


def test_diffusion_policy_id_and_float_validation_are_exact() -> None:
    assert DiffusionSamplingConfig(temperature=1.0000001, block_length=32, steps_per_block=32, top_k=0, top_p=1.0, cfg_scale=0.0, remasking="low_confidence").policy_id != (
        DiffusionSamplingConfig(temperature=1.0000002, block_length=32, steps_per_block=32, top_k=0, top_p=1.0, cfg_scale=0.0, remasking="low_confidence").policy_id
    )
    valid = {"block_length": 4, "steps_per_block": 4, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "cfg_scale": 0.0,
             "remasking": "random"}
    for kwargs in (
        {"temperature": float("nan")},
        {"top_p": float("inf")},
        {"cfg_scale": float("nan")},
    ):
        with pytest.raises(ValueError):
            DiffusionSamplingConfig(**(valid | kwargs))


def test_exact_diffusion_trajectory_requires_a_complete_finite_trace() -> None:
    with pytest.raises(ValueError, match="complete trace"):
        DiffusionSample((), (1,), (), 0.0, "policy", "model", "request")
    with pytest.raises(ValueError, match="finite"):
        DiffusionTraceStep(0, 0, (0,), (1,), float("nan"))

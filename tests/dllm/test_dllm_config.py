"""Tests for diffusion-language-model sampling configuration."""

from __future__ import annotations

import pytest

from inference_scaling.dllm.config import (
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest


def test_decision_stages_preserve_complete_llada_blocks():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=2,
        temperature=1.0,
        remasking="random",
    )

    lengths = diffusion_decision_stage_lengths(
        prompt_length=5,
        total_length=96,
        decision_block_size=48,
        sampling=sampling,
    )

    assert lengths == (48, 48)
    assert sampling.total_steps(96, prefix_length=5) == 48


def test_generation_length_must_contain_complete_llada_blocks():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
    )

    DiffusionGenerationRequest((1, 2, 3, 4, 5), 8, sampling, 0, "aligned")
    with pytest.raises(ValueError, match="generation_length"):
        DiffusionGenerationRequest((1, 2, 3, 4, 5), 7, sampling, 0, "split")

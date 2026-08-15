from __future__ import annotations

import pytest

from inference_scaling.dllm.config import (
    DiffusionSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest


def test_absolute_sdar_decisions_finish_the_partial_prompt_block_once():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=1.0,
        remasking="random",
        block_alignment="absolute",
    )

    lengths = diffusion_decision_stage_lengths(
        prompt_length=5,
        total_length=95,
        decision_block_size=48,
        sampling=sampling,
    )

    assert lengths == (47, 48)
    assert sampling.total_steps(95, prefix_length=5) == 95


def test_absolute_alignment_is_checked_against_prompt_plus_continuation():
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        block_alignment="absolute",
    )

    DiffusionGenerationRequest((1, 2, 3, 4, 5), 7, sampling, 0, "aligned")
    with pytest.raises(ValueError, match=r"prefix_length \+ generation_length"):
        DiffusionGenerationRequest((1, 2, 3, 4, 5), 8, sampling, 0, "split")

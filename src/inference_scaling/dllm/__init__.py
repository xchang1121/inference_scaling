"""Diffusion language-model sampling, weighting, and alignment utilities."""

from inference_scaling.dllm.config import (
    BlockAlignment,
    DiffusionBlockBeamConfig,
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
    VRPOSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTraceStep,
    DiffusionTrajectoryScoreRequest,
)
from inference_scaling.dllm.vrpo import (
    VRPOMaskPlan,
    VRPOMaskSample,
    VRPOPreferenceEstimate,
    estimate_masked_elbo,
    estimate_vrpo_preference_loss,
    sample_vrpo_mask_plan,
)

__all__ = [
    "BlockAlignment",
    "DiffusionBackend",
    "DiffusionBlockBeamConfig",
    "DiffusionGenerationRequest",
    "DiffusionISConfig",
    "DiffusionMHConfig",
    "DiffusionPowerMHConfig",
    "DiffusionSample",
    "DiffusionSamplingConfig",
    "DiffusionTraceStep",
    "DiffusionTrajectoryScoreRequest",
    "VRPOSamplingConfig",
    "VRPOMaskPlan",
    "VRPOMaskSample",
    "VRPOPreferenceEstimate",
    "estimate_masked_elbo",
    "estimate_vrpo_preference_loss",
    "diffusion_decision_stage_lengths",
    "sample_vrpo_mask_plan",
]

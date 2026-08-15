"""Diffusion language-model sampling, weighting, and alignment utilities."""

from inference_scaling.dllm.config import (
    DiffusionBlockBeamConfig,
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
    VRPOSamplingConfig,
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
    "sample_vrpo_mask_plan",
]

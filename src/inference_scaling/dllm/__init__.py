"""Diffusion language-model sampling, weighting, and alignment utilities."""

from inference_scaling.dllm.config import (
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
from inference_scaling.dllm.replay import (
    DiffusionReplayCandidate,
    DiffusionReplayEnergyEstimate,
    DiffusionReplayHistory,
    DiffusionReplayRecord,
    DiffusionReplaySelection,
    build_diffusion_replay_history,
    select_diffusion_candidates_with_replay,
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
    "DiffusionReplayCandidate",
    "DiffusionReplayEnergyEstimate",
    "DiffusionReplayHistory",
    "DiffusionReplayRecord",
    "DiffusionReplaySelection",
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
    "build_diffusion_replay_history",
    "sample_vrpo_mask_plan",
    "select_diffusion_candidates_with_replay",
]

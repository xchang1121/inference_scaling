"""Validated configuration of the masked-diffusion sampling algorithms.

The reverse-diffusion sampling policy lives in :mod:`inference_scaling.dllm.config`.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.shared.config import require_positive


@dataclass(frozen=True, slots=True)
class DiffusionISConfig:
    candidate_count: int
    rollout_count: int
    block_size: int
    total_length: int
    reward_temperature: float

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


@dataclass(frozen=True, slots=True)
class DiffusionMHConfig:
    total_length: int
    updates: int
    reward_temperature: float

    def __post_init__(self) -> None:
        require_positive("total_length", self.total_length)
        require_positive("updates", self.updates)
        require_positive("reward_temperature", self.reward_temperature)


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHConfig:
    """Finite-step sharpening of an exact reverse-trajectory policy."""

    total_length: int
    decision_block_size: int
    updates_per_stage: int
    alpha: float

    def __post_init__(self) -> None:
        for name in ("total_length", "decision_block_size", "updates_per_stage"):
            require_positive(name, getattr(self, name))
        if self.decision_block_size > self.total_length:
            raise ValueError("decision_block_size cannot exceed total_length")
        require_positive("alpha", self.alpha)


@dataclass(frozen=True, slots=True)
class DiffusionBlockBeamConfig:
    """Sampled diffusion-block search used as the counterpart of token beam search."""

    total_length: int
    decision_block_size: int
    width: int
    branching_factor: int

    def __post_init__(self) -> None:
        for name in ("total_length", "decision_block_size", "width", "branching_factor"):
            require_positive(name, getattr(self, name))
        if self.decision_block_size > self.total_length:
            raise ValueError("decision_block_size cannot exceed total_length")


__all__ = [
    "DiffusionBlockBeamConfig",
    "DiffusionISConfig",
    "DiffusionMHConfig",
    "DiffusionPowerMHConfig",
]

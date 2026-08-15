"""Configuration for masked diffusion language-model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RemaskingStrategy = Literal[
    "low_confidence",
    "low_confidence_static",
    "low_confidence_dynamic",
    "random",
    "sequential",
]


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class DiffusionSamplingConfig:
    """One reverse-diffusion sampling policy.

    ``steps_per_block`` is used instead of a total step count so the same policy
    can generate a candidate block or a longer rollout without changing its
    per-block transition kernel.
    """

    block_length: int = 32
    steps_per_block: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    cfg_scale: float = 0.0
    remasking: RemaskingStrategy = "low_confidence"
    confidence_threshold: float = 0.85
    mask_token_id: int | None = None

    def __post_init__(self) -> None:
        _positive("block_length", self.block_length)
        _positive("steps_per_block", self.steps_per_block)
        if self.steps_per_block > self.block_length:
            raise ValueError("steps_per_block cannot exceed block_length")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")
        if self.remasking not in (
            "low_confidence",
            "low_confidence_static",
            "low_confidence_dynamic",
            "random",
            "sequential",
        ):
            raise ValueError(f"unsupported remasking strategy {self.remasking!r}")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must lie in (0, 1]")
        if self.mask_token_id is not None and self.mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")

    @property
    def policy_id(self) -> str:
        return (
            f"block={self.block_length};steps={self.steps_per_block};"
            f"temperature={self.temperature:g};top_k={self.top_k};top_p={self.top_p:g};"
            f"cfg={self.cfg_scale:g};remasking={self.remasking};"
            f"threshold={self.confidence_threshold:g};mask={self.mask_token_id}"
        )

    @property
    def has_exact_trajectory_density(self) -> bool:
        """Whether committed transitions have a tractable normalized density."""

        return self.temperature > 0 and self.remasking in {"random", "sequential"}

    def validate_generation_length(self, generation_length: int) -> None:
        _positive("generation_length", generation_length)
        if generation_length % self.block_length:
            raise ValueError("generation_length must be divisible by block_length")

    def total_steps(self, generation_length: int) -> int:
        self.validate_generation_length(generation_length)
        return generation_length // self.block_length * self.steps_per_block


@dataclass(frozen=True, slots=True)
class DiffusionISConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 32
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            _positive(name, getattr(self, name))
        _positive("reward_temperature", self.reward_temperature)
        if self.total_length % self.block_size:
            raise ValueError("total_length must be divisible by block_size")
        if self.importance_log_ratio_clip is not None:
            _positive("importance_log_ratio_clip", self.importance_log_ratio_clip)


@dataclass(frozen=True, slots=True)
class DiffusionMHConfig:
    total_length: int = 128
    updates: int = 8
    reward_temperature: float = 1.0

    def __post_init__(self) -> None:
        _positive("total_length", self.total_length)
        _positive("updates", self.updates)
        _positive("reward_temperature", self.reward_temperature)


@dataclass(frozen=True, slots=True)
class VRPOSamplingConfig:
    """Monte Carlo layout for the masked-diffusion ELBO estimator."""

    timestep_samples: int = 8
    masks_per_timestep: int = 1
    antithetic: bool = True

    def __post_init__(self) -> None:
        _positive("timestep_samples", self.timestep_samples)
        _positive("masks_per_timestep", self.masks_per_timestep)

    @property
    def forward_passes(self) -> int:
        return self.timestep_samples * self.masks_per_timestep


__all__ = [
    "DiffusionISConfig",
    "DiffusionMHConfig",
    "DiffusionSamplingConfig",
    "RemaskingStrategy",
    "VRPOSamplingConfig",
]

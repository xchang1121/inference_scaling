"""Masked-diffusion sampling policy and VRPO estimator layout.

Algorithm settings live in :mod:`inference_scaling.dllm.algorithms.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from inference_scaling.shared.config import (
    canonical_float,
    require_nonnegative,
    require_positive,
    require_probability,
)

RemaskingStrategy = Literal["low_confidence", "random"]

@dataclass(frozen=True, slots=True)
class DiffusionSamplingConfig:
    """One reverse-diffusion sampling policy.

    ``steps_per_block`` is used instead of a total step count so the same policy
    can generate a candidate block or a longer rollout without changing its
    per-block transition kernel.
    """

    block_length: int
    steps_per_block: int
    temperature: float
    top_k: int
    top_p: float
    cfg_scale: float
    remasking: RemaskingStrategy

    def __post_init__(self) -> None:
        require_positive("block_length", self.block_length)
        require_positive("steps_per_block", self.steps_per_block)
        if self.steps_per_block > self.block_length:
            raise ValueError("steps_per_block cannot exceed block_length")
        require_nonnegative("temperature", self.temperature)
        require_nonnegative("top_k", self.top_k)
        require_probability("top_p", self.top_p, include_zero=False)
        require_nonnegative("cfg_scale", self.cfg_scale)
        if self.remasking not in ("low_confidence", "random"):
            raise ValueError(f"unsupported remasking strategy {self.remasking!r}")

    @property
    def policy_id(self) -> str:
        return (
            f"block={self.block_length};steps={self.steps_per_block};"
            f"temperature={canonical_float(self.temperature)};top_k={self.top_k};"
            f"top_p={canonical_float(self.top_p)};"
            f"cfg={canonical_float(self.cfg_scale)};remasking={self.remasking}"
        )

    @property
    def has_exact_trajectory_density(self) -> bool:
        """Whether committed transitions have a tractable normalized density."""

        return self.temperature > 0 and self.remasking == "random"

    def validate_generation_length(self, generation_length: int) -> None:
        require_positive("generation_length", generation_length)
        if generation_length % self.block_length:
            raise ValueError("generation_length must be divisible by block_length")


def sampling_from_settings(section: Mapping[str, Any]) -> DiffusionSamplingConfig:
    """A sampling policy from a ``sampling`` / ``exact_sampling`` settings section."""

    return DiffusionSamplingConfig(
        block_length=int(section["block_length"]), steps_per_block=int(section["steps_per_block"]),
        temperature=float(section["temperature"]), top_k=int(section["top_k"]), top_p=float(section["top_p"]),
        cfg_scale=float(section["cfg_scale"]), remasking=section["remasking"],
    )


def diffusion_decision_stage_lengths(
    *, total_length: int, decision_block_size: int, sampling: DiffusionSamplingConfig,
) -> tuple[int, ...]:
    """Partition a continuation into decision blocks without splitting a diffusion block."""

    require_positive("decision_block_size", decision_block_size)
    if decision_block_size > total_length:
        raise ValueError("decision_block_size cannot exceed total_length")
    sampling.validate_generation_length(total_length)
    sampling.validate_generation_length(decision_block_size)
    lengths = [decision_block_size] * (total_length // decision_block_size)
    if total_length % decision_block_size:
        lengths.append(total_length % decision_block_size)
    return tuple(lengths)


@dataclass(frozen=True, slots=True)
class VRPOSamplingConfig:
    """Monte Carlo layout for the masked-diffusion ELBO estimator."""

    timestep_samples: int
    masks_per_timestep: int
    antithetic: bool

    def __post_init__(self) -> None:
        require_positive("timestep_samples", self.timestep_samples)
        require_positive("masks_per_timestep", self.masks_per_timestep)

    @property
    def forward_passes(self) -> int:
        return self.timestep_samples * self.masks_per_timestep


__all__ = [
    "diffusion_decision_stage_lengths",
    "DiffusionSamplingConfig",
    "RemaskingStrategy",
    "VRPOSamplingConfig",
    "sampling_from_settings",
]

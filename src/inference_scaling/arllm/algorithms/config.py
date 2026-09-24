"""Validated configuration of the autoregressive sampling algorithms.

The model-level sampling policy lives in :mod:`inference_scaling.arllm.config`;
these dataclasses only fix candidate/rollout counts, block sizes, horizons and
reward temperatures of each algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.shared.config import (
    require_finite,
    require_positive,
)
from inference_scaling.shared.model.generation import DEFAULT_MAX_NEW_TOKENS


@dataclass(frozen=True, slots=True)
class MHConfig:
    alpha: float = 4.0
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    block_size: int = 32
    steps_per_block: int = 10
    suffix_schedule: str = "uniform"
    iterations: int | None = None

    def __post_init__(self) -> None:
        require_finite("alpha", self.alpha)
        if self.alpha < 1:
            raise ValueError("alpha must be at least one")
        for name in ("total_length", "block_size", "steps_per_block"):
            require_positive(name, getattr(self, name))
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
            raise ValueError("unknown MH suffix_schedule")
        if self.iterations is not None:
            require_positive("iterations", self.iterations)

    @property
    def stages(self) -> tuple[int, ...]:
        if self.iterations is not None:
            return (self.total_length,)
        lengths = tuple(range(self.block_size, self.total_length + 1, self.block_size))
        return lengths if lengths and lengths[-1] == self.total_length else (*lengths, self.total_length)

    @property
    def stage_updates(self) -> int:
        return self.steps_per_block if self.iterations is None else self.iterations


@dataclass(frozen=True, slots=True)
class RewardMHConfig:
    """Full-sequence MH budget for a base-times-exponentiated-reward target."""

    total_length: int = DEFAULT_MAX_NEW_TOKENS
    block_size: int = 32
    steps_per_block: int = 10
    reward_temperature: float = 0.1
    suffix_schedule: str = "uniform"
    iterations: int | None = None

    def __post_init__(self) -> None:
        for name in ("total_length", "block_size", "steps_per_block"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
            raise ValueError("unknown MH suffix_schedule")
        if self.iterations is not None:
            require_positive("iterations", self.iterations)

    @property
    def updates(self) -> int:
        if self.iterations is not None:
            return self.iterations
        blocks = (self.total_length + self.block_size - 1) // self.block_size
        return blocks * self.steps_per_block


@dataclass(frozen=True, slots=True)
class ConditionalISConfig:
    """Conditional IS on a kept complete sequence (see ``conditional_is``).

    Completions are both weight estimates and candidate answers, so one or two
    per candidate usually suffice; the budget is better spent on candidates.
    """

    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


__all__ = ["ConditionalISConfig", "MHConfig", "RewardMHConfig"]

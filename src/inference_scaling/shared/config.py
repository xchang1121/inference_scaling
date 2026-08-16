"""Runtime configuration shared by every model family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int = 0
    max_batch_size: int = 32
    max_batch_tokens: int = 4096
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")


@dataclass(frozen=True, slots=True)
class SMCForestConfig:
    """Model-independent particle and rollout layout for an SMC forest."""

    particle_count: int = 8
    branch_factor: int = 2
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    reward_workers: int = 4
    reuse_rollout_forest: bool = True

    def __post_init__(self) -> None:
        for name in (
            "particle_count",
            "branch_factor",
            "rollout_count",
            "block_size",
            "total_length",
            "reward_temperature",
            "reward_workers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


__all__ = ["RuntimeConfig", "SMCForestConfig"]

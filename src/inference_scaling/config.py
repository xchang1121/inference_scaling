"""Validated configuration objects shared by all algorithms."""

from __future__ import annotations

from dataclasses import dataclass


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """The complete stochastic policy used for one autoregressive request."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        _positive("temperature", self.temperature)
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must lie in (0, 1], got {self.top_p!r}")
        if self.top_k is not None:
            _positive("top_k", self.top_k)
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")

    @property
    def policy_id(self) -> str:
        return (
            f"temperature={self.temperature:g};top_p={self.top_p:g};"
            f"top_k={self.top_k};eos={self.eos_token_id}"
        )


@dataclass(frozen=True, slots=True)
class MHConfig:
    alpha: float = 4.0
    total_length: int = 192
    block_size: int = 32
    steps_per_block: int = 10
    chains: int = 1

    def __post_init__(self) -> None:
        if self.alpha < 1:
            raise ValueError("alpha must be at least one")
        for name in ("total_length", "block_size", "steps_per_block", "chains"):
            _positive(name, getattr(self, name))
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


@dataclass(frozen=True, slots=True)
class ConditionalEnergyConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            _positive(name, getattr(self, name))
        _positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


@dataclass(frozen=True, slots=True)
class BaseReplayConfig:
    candidate_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    max_history_per_candidate: int = 8
    fresh_rollouts: int = 2
    truncation: float = 8.0
    reserve_rollouts: int = 0

    def __post_init__(self) -> None:
        for name in ("candidate_count", "block_size", "total_length"):
            _positive(name, getattr(self, name))
        _positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.max_history_per_candidate < 0:
            raise ValueError("max_history_per_candidate must be non-negative")
        _positive("fresh_rollouts", self.fresh_rollouts)
        _positive("truncation", self.truncation)
        if self.reserve_rollouts < 0:
            raise ValueError("reserve_rollouts must be non-negative")


@dataclass(frozen=True, slots=True)
class DynamicISConfig:
    candidate_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    max_history_per_candidate: int = 8
    truncation: float = 8.0
    reserve_rollouts: int = 0
    rollout_budget: float = 64.0
    auxiliary_mixture: float = 0.25
    minimum_fresh_per_candidate: int = 1

    def __post_init__(self) -> None:
        for name in ("candidate_count", "block_size", "total_length"):
            _positive(name, getattr(self, name))
        _positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.max_history_per_candidate < 0:
            raise ValueError("max_history_per_candidate must be non-negative")
        _positive("truncation", self.truncation)
        if self.reserve_rollouts < 0:
            raise ValueError("reserve_rollouts must be non-negative")
        _positive("rollout_budget", self.rollout_budget)
        if not 0 <= self.auxiliary_mixture < 1:
            raise ValueError("auxiliary_mixture must lie in [0, 1)")
        _positive("minimum_fresh_per_candidate", self.minimum_fresh_per_candidate)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int = 0
    max_batch_size: int = 32
    max_batch_tokens: int = 4096
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        _positive("max_batch_size", self.max_batch_size)
        _positive("max_batch_tokens", self.max_batch_tokens)

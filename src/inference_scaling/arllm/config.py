"""Validated configuration objects shared by all algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.shared.config import RuntimeConfig


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
class RewardMHConfig:
    """Full-sequence MH budget for a base-times-exponentiated-reward target."""

    total_length: int = 192
    block_size: int = 32
    steps_per_block: int = 10
    reward_temperature: float = 0.1

    def __post_init__(self) -> None:
        for name in ("total_length", "block_size", "steps_per_block"):
            _positive(name, getattr(self, name))
        _positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")

    @property
    def updates(self) -> int:
        blocks = (self.total_length + self.block_size - 1) // self.block_size
        return blocks * self.steps_per_block


@dataclass(frozen=True, slots=True)
class ConditionalEnergyConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    apply_importance_correction: bool = True

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            _positive(name, getattr(self, name))
        _positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            _positive(
                "importance_log_ratio_clip",
                self.importance_log_ratio_clip,
            )
        if not self.apply_importance_correction and self.importance_log_ratio_clip is not None:
            raise ValueError(
                "importance_log_ratio_clip requires apply_importance_correction=True"
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


@dataclass(frozen=True, slots=True)
class ProgressiveISConfig:
    """Pilot/evaluation split for cost-aware conditional-energy estimation."""

    candidate_count: int = 4
    pilot_rollouts_per_candidate: int = 2
    evaluation_cost_budget: float = 16.0
    minimum_evaluation_per_candidate: int = 1
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    reward_workers: int = 4
    run_ahead_rollouts_per_candidate: int = 0
    evaluation_reference_rollouts_per_candidate: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "pilot_rollouts_per_candidate",
            "minimum_evaluation_per_candidate",
            "block_size",
            "total_length",
            "reward_workers",
        ):
            _positive(name, getattr(self, name))
        _positive("evaluation_cost_budget", self.evaluation_cost_budget)
        _positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            _positive("importance_log_ratio_clip", self.importance_log_ratio_clip)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.run_ahead_rollouts_per_candidate < 0:
            raise ValueError("run_ahead_rollouts_per_candidate must be non-negative")
        if (
            self.evaluation_reference_rollouts_per_candidate is not None
            and self.evaluation_reference_rollouts_per_candidate <= 0
        ):
            raise ValueError(
                "evaluation_reference_rollouts_per_candidate must be positive"
            )


@dataclass(frozen=True, slots=True)
class SMCForestConfig:
    """Auxiliary particle filter with reusable conditional rollout suffixes."""

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
            "reward_workers",
        ):
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


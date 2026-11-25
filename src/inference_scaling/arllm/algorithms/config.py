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


@dataclass(frozen=True, slots=True)
class MHConfig:
    """Suffix MH toward ``p(y)**alpha * exp(r(y) / reward_temperature)`` (see ``mh``).

    Without a reward the outputs grow in stages of ``block_size`` tokens with
    ``steps_per_block`` updates each; a reward needs complete outputs, so a
    rewarded chain runs as many updates in one full-length stage.
    ``iterations`` runs that many updates in one full-length stage.
    """

    alpha: float
    # None: the target has no reward term.
    reward_temperature: float | None
    total_length: int
    block_size: int
    steps_per_block: int
    suffix_schedule: str
    iterations: int | None
    # Replay the current suffix as a draft of each base proposal (same proposals, fewer model calls).
    suffix_replay: bool
    # Stop generating a proposal once it can no longer be accepted (same chain, fewer tokens).
    early_rejection: bool

    def __post_init__(self) -> None:
        require_finite("alpha", self.alpha)
        if self.alpha < 1:
            raise ValueError("alpha must be at least one")
        if self.reward_temperature is not None:
            require_positive("reward_temperature", self.reward_temperature)
        for name in ("total_length", "block_size", "steps_per_block"):
            require_positive(name, getattr(self, name))
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale", "log_uniform"}:
            raise ValueError("unknown MH suffix_schedule")
        if self.iterations is not None:
            require_positive("iterations", self.iterations)

    @property
    def stages(self) -> tuple[tuple[int, int], ...]:
        """The length limit and the update count of each stage."""

        if self.iterations is not None:
            return ((self.total_length, self.iterations),)
        limits = (*range(self.block_size, self.total_length, self.block_size), self.total_length)
        if self.reward_temperature is not None:
            return ((self.total_length, len(limits) * self.steps_per_block),)
        return tuple((limit, self.steps_per_block) for limit in limits)


@dataclass(frozen=True, slots=True)
class ConditionalISConfig:
    """Conditional IS on a kept complete sequence (see ``conditional_is``).

    External inputs ``candidate_count`` (M), ``rollout_count`` (K), and
    ``block_size`` (B) fix the scaling budget. ``exploration_schedule="adaptive"``
    adds dynamic scheduling on top of (M, K, B): B-anchored pilot escalation,
    unbiased selective K-lookahead, stagnation-only straggler pruning, ESS-bounded
    reward-temperature calibration, and bounded Stage-2 exploration.
    """

    candidate_count: int
    rollout_count: int
    block_size: int
    total_length: int
    reward_temperature: float
    # Draw fresh candidates as blocks and their first completions with the other completions.
    block_first: bool
    # Fixed blocks are the CPU reference; two_stage bounds exploration to two pools;
    # adaptive adds dynamic escalation, selective K-lookahead, straggler pruning, and ESS calibration.
    exploration_schedule: str = "fixed"
    # Fresh-suffix capacity; None retains weight-based allocation and the auxiliary length interval.
    final_pool_limit: int | None = None
    # Approximate stopping from cached prefix weights in either two-stage pool.
    selection_ready: bool = False
    # Complete-pool index kernel; experimental alternatives leave generation unchanged.
    selection: str = "sir"
    # Adaptive schedule dynamic controls (ignored unless exploration_schedule == "adaptive").
    pilot_gate: bool = True
    selective_lookahead: bool = True
    straggler_pruning: bool = True
    adaptive_temperature: bool = True
    # Optional escalation horizon for pilot_gate; None anchors directly to block_size (B).
    pilot_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        if type(self.selection_ready) is not bool:
            raise ValueError("selection_ready must be a boolean")
        if self.exploration_schedule not in {"fixed", "two_stage", "adaptive"}:
            raise ValueError("unknown IS exploration_schedule")
        if self.selection_ready and (self.exploration_schedule != "two_stage" or self.block_first):
            raise ValueError("selection_ready requires two_stage without block_first")
        if self.selection not in {"sir", "metropolized", "block_metropolized"}:
            raise ValueError("unknown IS selection")
        if self.selection_ready and self.selection != "sir":
            raise ValueError("Metropolized selection requires complete rollout weights; disable selection_ready")
        if self.exploration_schedule == "adaptive" and self.block_first:
            raise ValueError("adaptive exploration_schedule requires block_first=false")
        if self.final_pool_limit is not None and (type(self.final_pool_limit) is not int or self.final_pool_limit < 1):
            raise ValueError("final_pool_limit must be a positive integer or None")
        if self.pilot_tokens is not None:
            require_positive("pilot_tokens", self.pilot_tokens)
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


__all__ = ["ConditionalISConfig", "MHConfig"]

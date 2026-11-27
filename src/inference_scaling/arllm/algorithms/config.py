"""Validated configuration of the autoregressive sampling algorithms.

The model-level sampling policy lives in :mod:`inference_scaling.arllm.config`;
these dataclasses fix the external scaling dimensions and horizons of each
algorithm.
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
    """Converged multi-stage Block-SMC Conditional IS parameterized by ``(M, K, S)``.

    External dimensions:
    - ``candidate_count`` (M): candidate width per stage.
    - ``rollout_count`` (K): Stage-0 lookahead depth (decays geometrically across stages).
    - ``stages`` (S): maximum number of progressive refinement stages.

    Internal scheduling automatically binds:
    - hardware KV-page-aligned Stage-0 checkpoint ``stage0_block_size`` (derived from
      ``base_block_size`` or horizon), with pilot escalation at ``stage0_block_size // 2``;
    - observed-length-adaptive geometric checkpoints for intermediate stages;
    - automatic multi-particle Block-SMC (``beam_particles = 2`` when ``S >= 3`` and
      ``M >= 4``, otherwise ``1``) with systematic resampling and exact parent-mass
      conservation;
    - selective K-lookahead, stagnation-only straggler pruning, ESS-bounded reward
      temperature calibration, and ``block_metropolized`` selection.
    """

    candidate_count: int
    rollout_count: int
    stages: int
    total_length: int
    reward_temperature: float
    # Internal hardware KV-page base quantum; None derives from total_length and stages.
    base_block_size: int | None = None
    # Internal admission cap from the streaming scheduler; None leaves fanout uncapped.
    final_pool_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "stages", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.base_block_size is not None:
            require_positive("base_block_size", self.base_block_size)
        if self.final_pool_limit is not None:
            require_positive("final_pool_limit", self.final_pool_limit)

    @property
    def stage0_block_size(self) -> int:
        """Initial Stage-0 prefix checkpoint width in tokens."""
        if self.stages == 1:
            return self.total_length
        if self.base_block_size is not None:
            return min(self.base_block_size, self.total_length)
        divisor = 2 ** min(self.stages - 1, 10)
        return min(1024, max(1, self.total_length // divisor))

    @property
    def beam_particles(self) -> int:
        """Active particle count: 2-particle Block-SMC when S >= 3 and M >= 4, else 1."""
        return 2 if self.stages >= 3 and self.candidate_count >= 4 else 1


__all__ = ["ConditionalISConfig", "MHConfig"]

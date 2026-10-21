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
    require_probability,
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
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
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

    Completions are both weight estimates and candidate answers, so one or two
    per candidate usually suffice; the budget is better spent on candidates.
    """

    candidate_count: int
    rollout_count: int
    block_size: int
    total_length: int
    reward_temperature: float
    # Draw fresh candidates as blocks and their first completions with the other completions.
    block_first: bool
    # Reuse completed siblings of the selected block; finite-budget sampling changes.
    sibling_replay: bool = False
    # Below one: stop at this mean completed fraction of candidate groups; length-biased.
    completion_fraction: float = 1.0
    # Race at most twice the missing completions per block, retaining the first K; length-biased.
    rollout_race: bool = False
    # After the first complete tree, optionally spend predicted work in one final suffix pool.
    exploration_schedule: str = "fixed"

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        if type(self.sibling_replay) is not bool:
            raise ValueError("sibling_replay must be a boolean")
        if type(self.rollout_race) is not bool:
            raise ValueError("rollout_race must be a boolean")
        if self.exploration_schedule not in {"fixed", "two_stage"}:
            raise ValueError("unknown IS exploration_schedule")
        if self.exploration_schedule == "two_stage" and (
                self.sibling_replay or self.completion_fraction != 1.0 or self.rollout_race):
            raise ValueError("two_stage exploration requires complete trees without sibling replay or rollout races")
        require_probability("completion_fraction", self.completion_fraction, include_zero=False)
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


__all__ = ["ConditionalISConfig", "MHConfig"]

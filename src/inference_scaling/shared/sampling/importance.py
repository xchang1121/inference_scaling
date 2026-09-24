"""Importance weights of terminal completions, shared by AR and diffusion IS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Generic, Literal, TypeVar



PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class RolloutObservation(Generic[PayloadT]):
    """One terminal completion before its IS or replay weight is applied."""

    reward: float
    target_logprob: float | None = None
    proposal_logprob: float | None = None
    payload: PayloadT | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.reward):
            raise ValueError("rollout reward must be finite")
        for name, value in (
            ("target_logprob", self.target_logprob),
            ("proposal_logprob", self.proposal_logprob),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when provided")


@dataclass(frozen=True, slots=True)
class WeightedRollout(Generic[PayloadT]):
    observation: RolloutObservation[PayloadT]
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float


@dataclass(frozen=True, slots=True)
class MonteCarloWeightEstimate(Generic[PayloadT]):
    log_weight: float
    rollouts: tuple[WeightedRollout[PayloadT], ...]


class MonteCarloRolloutWeightProvider(Generic[PayloadT]):
    """Weights on-policy, off-policy, or deliberately uncorrected rollouts."""

    def __init__(
        self,
        *,
        reward_temperature: float,
        correction: Literal["importance", "identity", "none"] = "importance",
        log_ratio_clip: float | None = None,
    ) -> None:
        if reward_temperature <= 0:
            raise ValueError("reward_temperature must be positive")
        if correction not in {"importance", "identity", "none"}:
            raise ValueError(f"unknown rollout correction mode {correction!r}")
        if log_ratio_clip is not None and log_ratio_clip <= 0:
            raise ValueError("log_ratio_clip must be positive when provided")
        if correction != "importance" and log_ratio_clip is not None:
            raise ValueError("log-ratio clipping requires importance correction")
        self.reward_temperature = float(reward_temperature)
        self.correction = correction
        self.log_ratio_clip = log_ratio_clip

    def weight(self, observation: RolloutObservation[PayloadT]) -> WeightedRollout[PayloadT]:
        if self.correction == "importance":
            if observation.target_logprob is None or observation.proposal_logprob is None:
                raise ValueError(
                    "importance correction requires target and proposal log-probabilities"
                )
            ratio = (
                observation.target_logprob - observation.proposal_logprob
            )
            raw_ratio: float | None = ratio
            applied_ratio: float | None = ratio
            if self.log_ratio_clip is not None:
                applied_ratio = min(
                    max(ratio, -self.log_ratio_clip), self.log_ratio_clip
                )
        elif self.correction == "identity":
            raw_ratio = 0.0
            applied_ratio = 0.0
        else:
            raw_ratio = None
            applied_ratio = None
        log_weight = observation.reward / self.reward_temperature
        if applied_ratio is not None:
            log_weight += applied_ratio
        return WeightedRollout(
            observation=observation,
            raw_log_importance_ratio=raw_ratio,
            applied_log_importance_ratio=applied_ratio,
            log_weight=log_weight,
        )

    def estimate(
        self, observations: Sequence[RolloutObservation[PayloadT]]
    ) -> MonteCarloWeightEstimate[PayloadT]:
        weighted = tuple(self.weight(observation) for observation in observations)
        if not weighted:
            raise ValueError("at least one rollout is required to estimate a conditional weight")
        return MonteCarloWeightEstimate(
            log_weight=logmeanexp([rollout.log_weight for rollout in weighted]),
            rollouts=weighted,
        )


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


__all__ = [
    "MonteCarloRolloutWeightProvider",
    "MonteCarloWeightEstimate",
    "RolloutObservation",
    "WeightedRollout",
    "logmeanexp",
]

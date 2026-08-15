"""Importance-weight identities shared by ARLLM and dLLM replay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log1p

import numpy as np


@dataclass(frozen=True, slots=True)
class ProbabilityObservation:
    target_logprob: float
    behavior_logprob: float
    reward: float

    @property
    def base_logprob(self) -> float:
        """Compatibility name retained for the ARLLM replay API."""

        return self.target_logprob

    @property
    def mixture_logprob(self) -> float:
        """Compatibility name retained for the ARLLM replay API."""

        return self.behavior_logprob


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


def corrected_replay_log_energy(
    history: Sequence[ProbabilityObservation],
    fresh: Sequence[ProbabilityObservation],
    *,
    truncation: float,
    reward_temperature: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Truncated history estimator plus its exact fresh-sample tail correction."""

    if not history:
        raise ValueError("corrected replay requires at least one history observation")
    if not fresh:
        raise ValueError("corrected replay requires at least one fresh observation")
    if truncation <= 0 or reward_temperature <= 0:
        raise ValueError("truncation and reward_temperature must be positive")
    log_truncation = log(truncation)
    history_terms: list[float] = []
    for observation in history:
        log_ratio = observation.target_logprob - observation.behavior_logprob
        if not isfinite(log_ratio):
            raise ValueError("history behavior support must lie inside target support")
        history_terms.append(
            min(log_truncation, log_ratio)
            + observation.reward / reward_temperature
        )

    fresh_terms: list[float] = []
    for observation in fresh:
        if observation.behavior_logprob == float("-inf"):
            log_tail = 0.0
        else:
            log_ratio = observation.target_logprob - observation.behavior_logprob
            if log_ratio <= log_truncation:
                log_tail = float("-inf")
            else:
                log_tail = log1p(-exp(log_truncation - log_ratio))
        fresh_terms.append(log_tail + observation.reward / reward_temperature)

    history_mean = logmeanexp(history_terms)
    finite_fresh = [value for value in fresh_terms if value != float("-inf")]
    fresh_mean = (
        logmeanexp(finite_fresh) + log(len(finite_fresh) / len(fresh_terms))
        if finite_fresh
        else float("-inf")
    )
    return (
        float(np.logaddexp(history_mean, fresh_mean)),
        tuple(history_terms),
        tuple(fresh_terms),
    )


__all__ = [
    "ProbabilityObservation",
    "corrected_replay_log_energy",
    "logmeanexp",
]

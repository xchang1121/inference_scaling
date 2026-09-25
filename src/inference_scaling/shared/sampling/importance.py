"""Importance-weight arithmetic shared by AR and diffusion IS."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, isfinite, log

import numpy as np


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


def normalize_log_weights(log_weights: Sequence[float]) -> tuple[float, ...]:
    """Selection probabilities proportional to ``exp(log_weight)``."""

    if not log_weights:
        raise ValueError("at least one candidate log-weight is required")
    values = np.asarray(log_weights, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate log-weights must be finite")
    shifted = np.exp(values - float(np.max(values)))
    return tuple(float(value) for value in shifted / float(shifted.sum()))


def categorical_index_from_uniform(probabilities: Sequence[float], uniform: float) -> int:
    """Select from categorical probabilities using one explicit uniform draw."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("categorical probabilities must be a non-empty finite non-negative vector")
    total = float(values.sum())
    if not isfinite(total) or total <= 0.0:
        raise ValueError("categorical probabilities must have positive mass")
    if not isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be finite and lie in [0, 1)")
    cumulative = np.cumsum(values / total, dtype=np.float64)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, uniform, side="right"))


__all__ = ["categorical_index_from_uniform", "logmeanexp", "normalize_log_weights"]

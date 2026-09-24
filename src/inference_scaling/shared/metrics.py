"""Distributional and sampling diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import fsum
from typing import Hashable, TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)


def empirical_distribution(samples: Iterable[T]) -> dict[T, float]:
    counts = Counter(samples)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("at least one sample is required")
    return {value: count / total for value, count in counts.items()}


def total_variation(left: Mapping[T, float], right: Mapping[T, float]) -> float:
    support = set(left) | set(right)
    return 0.5 * fsum(abs(left.get(value, 0.0) - right.get(value, 0.0)) for value in support)


def importance_effective_sample_size(log_weights: Sequence[float]) -> float:
    if not log_weights:
        return 0.0
    values = np.asarray(log_weights, dtype=np.float64)
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return 0.0
    weights = np.exp(values - maximum)
    denominator = float(np.dot(weights, weights))
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


__all__ = ["empirical_distribution", "importance_effective_sample_size", "total_variation"]

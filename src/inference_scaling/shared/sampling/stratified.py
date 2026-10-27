"""Independent arithmetic strata for complete trajectories.

One randomly assigned stratum per trajectory preserves each base marginal.
Conditional intervals are updated after each token and released as soon as
the whole remaining code interval is available. No prefix history is stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class _Interval:
    lower: float
    upper: float
    generator: np.random.Generator
    tokens: int = 0
    maximum_cdf_error: float = 0.


class InitialStrata:
    """Request-seed registry; resumed requests may alias their original seed.

    Callers serialize registry operations with their sampling worker. Separate
    pools can coexist. ``remove`` is idempotent, including failed registration.
    Each original seed must identify exactly one sequential trajectory.
    """

    def __init__(self) -> None:
        self.intervals: dict[int, _Interval] = {}
        self.aliases: dict[int, int] = {}

    def add(self, entries: Iterable[tuple[int, int, int]]) -> None:
        entries = tuple(entries)
        seeds = [seed for seed, _, _ in entries]
        if (not entries or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0
                or type(index) is not int or type(size) is not int or not 0 <= index < size
                for seed, index, size in entries)):
            raise ValueError("initial strata require distinct nonnegative seeds and valid strata")
        if len({size for _, _, size in entries}) != 1 or len({index for _, index, _ in entries}) != len(entries):
            raise ValueError("register exactly one trajectory per selected stratum of the same partition")
        if any(seed in self.aliases for seed in seeds):
            raise ValueError("an initial-stratum seed is already registered")
        for seed, index, size in entries:
            self.intervals[seed] = _Interval(index / size, (index + 1) / size,
                np.random.default_rng(np.random.SeedSequence([seed, 20261028])))
            self.aliases[seed] = seed

    def bind(self, entries: Iterable[tuple[int, int]]) -> None:
        entries = tuple(entries)
        if any(original not in self.intervals or type(alias) is not int or alias < 0
               or (alias in self.aliases and self.aliases[alias] != original) for original, alias in entries):
            raise ValueError("unknown original seed or colliding initial-stratum alias")
        if len({alias for _, alias in entries}) != len(entries):
            raise ValueError("initial-stratum aliases must be distinct")
        self.aliases.update((alias, original) for original, alias in entries)

    def active(self, seed: int) -> bool:
        original = self.aliases.get(seed)
        return original is not None and not self._resolved(self.intervals[original])

    @staticmethod
    def _resolved(interval: _Interval) -> bool:
        return interval.lower == 0. and interval.upper == 1.

    @property
    def has_active(self) -> bool:
        return any(not self._resolved(interval) for interval in self.intervals.values())

    def sample(self, seed: int, probabilities: np.ndarray) -> int:
        interval = self.intervals[self.aliases[seed]]
        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any() or not values.sum() > 0:
            raise ValueError("initial strata require a finite categorical probability vector")
        cumulative = np.cumsum(values)
        cumulative /= cumulative[-1]
        error = float(np.abs(np.diff(cumulative, prepend=0.) - values / values.sum()).sum() / 2)
        if error > 1e-9:
            raise ArithmeticError("initial-stratum CDF rounding error exceeds its tolerance")
        lower, upper = interval.lower, interval.upper
        draw = min(lower + (upper - lower) * interval.generator.random(), np.nextafter(upper, lower))
        token = int(np.searchsorted(cumulative, draw, side="right"))
        left, right = (cumulative[token - 1] if token else 0.), cumulative[token]
        bottom, top = max(lower, left), min(upper, right)
        if not left <= draw < right or not bottom < top:
            raise ArithmeticError("empty initial-stratum conditional interval")
        interval.lower, interval.upper = float((bottom - left) / (right - left)), float((top - left) / (right - left))
        interval.tokens += 1
        interval.maximum_cdf_error = max(interval.maximum_cdf_error, error)
        return token

    def remove(self, seeds: Iterable[int]) -> dict[str, Any]:
        seeds = set(seeds)
        removed = {seed: self.intervals.pop(seed) for seed in seeds if seed in self.intervals}
        self.aliases = {alias: original for alias, original in self.aliases.items() if original not in seeds}
        return {"registered": len(removed), "touched": sum(interval.tokens > 0 for interval in removed.values()),
                "constrained_tokens": sum(interval.tokens for interval in removed.values()),
                "unresolved": sum(not self._resolved(interval) for interval in removed.values()),
                "maximum_cdf_error": max((interval.maximum_cdf_error for interval in removed.values()), default=0.),
                "trajectories": {str(seed): {"tokens": interval.tokens, "resolved": self._resolved(interval)}
                                 for seed, interval in removed.items()}}

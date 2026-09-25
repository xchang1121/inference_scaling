"""Importance weights of terminal completions, shared by AR and diffusion IS."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, log


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one value is required")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + log(sum(exp(value - maximum) for value in values)) - log(len(values))


__all__ = ["logmeanexp"]

"""Validation helpers and canonical policy identifiers shared by every model family."""

from __future__ import annotations

from math import isfinite


def canonical_float(value: float) -> str:
    """Return a stable, round-trippable identifier for one finite float."""

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"a policy parameter must be finite, got {value!r}")
    return repr(numeric)


def require_finite(name: str, value: int | float) -> None:
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def require_positive(name: str, value: int | float) -> None:
    require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def require_nonnegative(name: str, value: int | float) -> None:
    require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def require_probability(
    name: str,
    value: float,
    *,
    include_zero: bool = True,
) -> None:
    require_finite(name, value)
    lower_valid = value >= 0 if include_zero else value > 0
    if not lower_valid or value > 1:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must lie in {interval}, got {value!r}")


__all__ = [
    "canonical_float",
    "require_finite",
    "require_nonnegative",
    "require_positive",
    "require_probability",
]

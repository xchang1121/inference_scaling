"""Transparent token and dominant-matmul FLOP accounting."""

from __future__ import annotations


def dense_forward_flops(parameter_count: int, forward_token_slots: int) -> int:
    """Conventional ``2 * parameters * tokens`` forward-pass estimate."""

    if parameter_count < 0 or forward_token_slots < 0:
        raise ValueError("parameter_count and forward_token_slots must be non-negative")
    return 2 * parameter_count * forward_token_slots

"""Bounded-memory causal forward passes with the complete context."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_causal_chunks(
    model: Any, input_ids: Any, attention_mask: Any, position_ids: Any, *, chunk_size: int, first_needed: int,
    supports_logits_to_keep: bool, on_logits: Callable[[int, Any], None], cache: Any = None,
) -> Any:
    """Feed ``input_ids`` after ``cache`` in chunks of ``chunk_size`` positions and return the extended cache.

    ``attention_mask`` covers the cached and the new positions. For each chunk
    that reaches position ``first_needed``, ``on_logits(position, logits)``
    receives the logits of the chunk's positions from ``position`` on, so no more
    than one chunk of logits exists at a time. A missing cache is an explicit
    capability error, not a shorter-context approximation. The caller owns the
    model lock and inference-mode context.
    """
    length = input_ids.shape[1]
    past = attention_mask.shape[1] - length
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        needed = end - max(start, first_needed)
        kwargs = {} if cache is None else {"past_key_values": cache}
        if supports_logits_to_keep:
            kwargs["logits_to_keep"] = max(1, needed)
        output = model(input_ids=input_ids[:, start:end], attention_mask=attention_mask[:, : past + end],
                       position_ids=position_ids[:, start:end], use_cache=True, return_dict=True, **kwargs)
        cache = getattr(output, "past_key_values", None)
        if cache is None and end < length:
            raise ValueError("chunked forward passes require a model returning past_key_values")
        if needed > 0:
            if output.logits.shape[1] < needed:
                raise RuntimeError("model omitted required logits")
            on_logits(end - needed, output.logits[:, -needed:, :])
        del output
    return cache

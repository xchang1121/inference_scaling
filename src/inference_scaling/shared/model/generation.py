"""Generation budgets and model context limits, independent of benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

def context_limit(backend: Any) -> int | None:
    """Read advertised limits; ignore tokenizer 'unlimited' sentinel values."""
    limits = []
    visited: set[int] = set()

    def inspect(obj: Any) -> None:
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))
        for name in ("max_position_embeddings", "n_positions", "max_model_len", "model_max_length"):
            value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**9:
                limits.append(value)
        for name in ("backend", "model", "config", "text_config", "tokenizer",
                     "engine", "_engine", "_runtime", "llm_engine", "model_config", "vllm_config"):
            child = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
            if child is not obj:
                inspect(child)

    inspect(backend)
    return min(limits) if limits else None


def generation_budget(
    requested: int | None, prompt_length: int, backends: Sequence[Any], *, context_window: int | None,
    thinking: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The output length a prompt leaves within the models' context limits."""
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError("max_new_tokens must be a positive integer or null")
    if prompt_length < 0:
        raise ValueError("prompt_length must be non-negative")
    limits = [limit for backend in backends if backend is not None and (limit := context_limit(backend)) is not None]
    if context_window is not None:
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        limits.append(context_window)
    if requested is None and not limits:
        raise ValueError("automatic output length requires a model context limit or ar.engine.context_window")
    available = min(limits) - max(1, prompt_length) if limits else requested
    assert available is not None
    if available <= 0:
        raise ValueError("prompt fills the model context window; shorten the prompt or select a longer-context model")
    effective = available if requested is None else min(requested, available)
    result = {
        "requested_max_new_tokens": requested, "effective_max_new_tokens": effective,
        "context_limit": min(limits) if limits else None,
        "prompt_tokens": prompt_length, "length_limited_by_context": requested is None or effective < requested,
    }
    if thinking is not None:
        limit = thinking["requested_thinking_budget"]
        result.update(thinking)
        result.update(effective_thinking_budget=None if limit is None else min(limit, effective),
                      thinking_limited_by_total=limit is not None and limit > effective)
    return result

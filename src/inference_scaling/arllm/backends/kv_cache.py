"""KV caches of the manual decoding loop: in-place growth and a store of reusable prefixes."""

from __future__ import annotations

from typing import Any

import numpy as np

from inference_scaling.shared.types import TokenSequence

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - dependency-free installations
    DynamicCache = object  # type: ignore[assignment,misc]


def cache_layers(cache: Any) -> list[tuple[Any, Any]]:
    """Per-layer ``(key, value)`` tensors of a cache in either Transformers cache layout."""

    return [(layer[0], layer[1]) for layer in cache]


class GrowingCache(DynamicCache):  # type: ignore[misc,valid-type]
    """A ``DynamicCache`` whose layers grow inside over-allocated buffers.

    ``DynamicCache`` concatenates each new position onto a fresh copy of the whole
    layer. Here the new positions are written in place and the layer becomes a view
    of the filled part; a layer that is no longer such a view (after its rows were
    selected) moves to a new buffer at its next update. The values are unchanged.
    """

    def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any) -> Any:
        layers = getattr(self, "layers", None)  # Transformers 5 keeps one object per layer
        if layers is None:
            current = (self.key_cache[layer_idx], self.value_cache[layer_idx]) if layer_idx < len(self.key_cache) else None
        else:
            layer = layers[layer_idx] if layer_idx < len(layers) else None
            plain = layer is not None and type(layer).__name__ == "DynamicLayer" and layer.is_initialized
            current = (layer.keys, layer.values) if plain else None
        if current is None or not current[0].numel():
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        keys, values = current
        length, count = keys.shape[-2], key_states.shape[-2]
        buffers = self.__dict__.setdefault("_growing_buffers", {})
        key_buffer, value_buffer = buffers.get(layer_idx, (None, None))
        if (key_buffer is None or key_buffer.data_ptr() != keys.data_ptr() or key_buffer.shape[:-2] != keys.shape[:-2]
                or key_buffer.shape[-2] < length + count):
            capacity = length + count + max(256, (length + count) // 2)
            key_buffer = keys.new_empty((*keys.shape[:-2], capacity, keys.shape[-1]))
            value_buffer = values.new_empty((*values.shape[:-2], capacity, values.shape[-1]))
            key_buffer[..., :length, :] = keys
            value_buffer[..., :length, :] = values
            buffers[layer_idx] = (key_buffer, value_buffer)
        key_buffer[..., length : length + count, :] = key_states
        value_buffer[..., length : length + count, :] = value_states
        keys, values = key_buffer[..., : length + count, :], value_buffer[..., : length + count, :]
        if layers is None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = keys, values
            if layer_idx == 0:
                self._seen_tokens += count
        else:
            layer.keys, layer.values = keys, values
        return keys, values


class PrefixStore:
    """Per-row KV states of finished generations, reused over a new prefix's longest stored prefix.

    Entries are kept within ``capacity`` bytes and the least recently used leaves
    first; an entry that is a prefix of a newer one is dropped.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._entries: list[tuple[np.ndarray, list[tuple[Any, Any]], int]] = []

    def clear(self) -> None:
        self._entries.clear()

    @staticmethod
    def _common(stored: np.ndarray, query: np.ndarray) -> int:
        limit = min(len(stored), len(query))
        different = np.flatnonzero(stored[:limit] != query[:limit])
        return int(different[0]) if different.size else limit

    def match(self, tokens: TokenSequence) -> tuple[int, list[tuple[Any, Any]]]:
        """The length and layers of the longest stored prefix of ``tokens``; that entry is used last."""

        query = np.asarray(tokens)
        lengths = [self._common(stored, query) for stored, _, _ in self._entries]
        if not lengths or not max(lengths):
            return 0, []
        index = int(np.argmax(lengths))
        entry = self._entries.pop(index)
        self._entries.append(entry)
        return lengths[index], entry[1]

    def add(self, tokens: TokenSequence, layers: list[tuple[Any, Any]]) -> None:
        stored = np.asarray(tokens)
        size = sum(tensor.numel() * tensor.element_size() for layer in layers for tensor in layer)
        if not len(stored) or size > self.capacity or any(
                self._common(entry, stored) == len(stored) for entry, _, _ in self._entries):
            return
        self._entries = [entry for entry in self._entries if self._common(entry[0], stored) < len(entry[0])]
        self._entries.append((stored, layers, size))
        while sum(entry[2] for entry in self._entries) > self.capacity:
            self._entries.pop(0)


__all__ = ["GrowingCache", "PrefixStore", "cache_layers"]

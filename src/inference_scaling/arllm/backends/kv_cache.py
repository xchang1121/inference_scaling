"""KV caches of the manual decoding loop: in-place growth, fixed-shape steps and a store of reusable prefixes."""

from __future__ import annotations

from typing import Any

import numpy as np

from inference_scaling.shared.types import TokenSequence

try:
    import torch
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - dependency-free installations
    torch = None  # type: ignore[assignment]
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


class _ColumnCache(DynamicCache):  # type: ignore[misc,valid-type]
    """Fixed KV buffers of one decode step; each update writes position ``column`` of every row."""

    def __init__(self, buffers: list[tuple[Any, Any]], column: Any) -> None:
        super().__init__()
        self.buffers, self.column = buffers, column

    def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any) -> Any:
        keys, values = self.buffers[layer_idx]
        keys.index_copy_(2, self.column, key_states)
        values.index_copy_(2, self.column, value_states)
        return keys, values

    def get_seq_length(self, *args: Any, **kwargs: Any) -> int:
        # The caller passes positions and the mask, so the model derives nothing from the cache.
        return 0


class StaticDecoder:
    """Decode steps at fixed shapes over kept KV buffers, replayed as CUDA graphs with ``capture``.

    A step writes the KV state of every live row to the same position. Rows and attended
    positions are rounded up to powers of two and the rest is masked, so a batch runs at
    a few shapes and each step equals dynamic decoding up to floating-point rounding.
    """

    def __init__(self, model: Any, logits_to_keep: bool, capture: bool) -> None:
        self.model, self._keep, self._capture = model, logits_to_keep, capture
        self._buffers: list[tuple[Any, ...]] = []
        self._graphs: dict[tuple[int, int], tuple[Any, Any]] = {}

    @staticmethod
    def _round(count: int) -> int:
        return 1 << max(count - 1, 0).bit_length()

    def load(self, layers: list[tuple[Any, Any]], rows: Any, attention_mask: Any, length: int) -> None:
        """Start a batch from the prefill KV states of ``rows`` of ``layers``; it will span ``length`` positions."""

        count, width, key = len(rows), attention_mask.shape[1], layers[0][0]
        shape = (self._round(count), self._round(length))
        if self._buffers:
            shape = (max(shape[0], self._tokens.shape[0]), max(shape[1], self._mask.shape[-1]))
        if not self._buffers or shape != (self._tokens.shape[0], self._mask.shape[-1]):
            # Graphs read fixed addresses, so new buffers need new graphs.
            self._graphs.clear()
            self._buffers = [tuple(state.new_zeros((shape[0], state.shape[1], shape[1], state.shape[-1]))
                                   for state in layer) for layer in layers]
            self._tokens = torch.zeros((shape[0], 1), dtype=torch.long, device=key.device)
            self._positions, self._column = torch.zeros_like(self._tokens), self._tokens.new_zeros(1)
            self._mask = key.new_empty((shape[0], 1, 1, shape[1]))
            self._pool = torch.cuda.graph_pool_handle() if self._capture else None
        for (key_state, value_state), (key_buffer, value_buffer) in zip(layers, self._buffers):
            key_buffer[:count, :, :width] = key_state.index_select(0, rows)
            value_buffer[:count, :, :width] = value_state.index_select(0, rows)
        # An additive mask: 0 where a row attends, the dtype's minimum elsewhere.
        self._mask.fill_(torch.finfo(self._mask.dtype).min)
        self._mask[:count, 0, 0, :width].masked_fill_(attention_mask.bool(), 0.0)

    def layers(self, rows: int, width: int) -> list[tuple[Any, Any]]:
        """The KV states of the first ``rows`` rows over their first ``width`` positions."""

        return [(key[:rows, :, :width], value[:rows, :, :width]) for key, value in self._buffers]

    def select(self, rows: Any, width: int) -> None:
        """Keep ``rows`` of the batch, in order, as its first rows."""

        for key, value in self._buffers:
            key[: len(rows), :, :width] = key[rows, :, :width]
            value[: len(rows), :, :width] = value[rows, :, :width]
        self._mask[: len(rows), :, :, :width] = self._mask[rows, :, :, :width]

    def step(self, tokens: Any, positions: Any, column: int) -> Any:
        """Next-token logits of the live rows after feeding ``tokens`` at ``positions``, stored at ``column``."""

        count = len(tokens)
        self._tokens[:count, 0], self._positions[:count, 0] = tokens, positions
        self._mask[:count, 0, 0, column] = 0.0
        self._column.fill_(column)
        shape = (self._round(count), self._round(column + 1))
        if not self._capture:
            return self._forward(*shape)[:count, -1]
        if shape not in self._graphs:
            # An ordinary call on a side stream initializes lazily built state; it writes the same values.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._forward(*shape)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                self._graphs[shape] = (graph, self._forward(*shape))
        graph, logits = self._graphs[shape]
        graph.replay()
        return logits[:count, -1]

    def _forward(self, rows: int, length: int) -> Any:
        cache = _ColumnCache([(key[:rows, :, :length], value[:rows, :, :length]) for key, value in self._buffers],
                             self._column)
        return self.model(input_ids=self._tokens[:rows], attention_mask=self._mask[:rows, :, :, :length],
                          position_ids=self._positions[:rows], past_key_values=cache, use_cache=True, return_dict=True,
                          **({"logits_to_keep": 1} if self._keep else {})).logits


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


__all__ = ["GrowingCache", "PrefixStore", "StaticDecoder", "cache_layers"]

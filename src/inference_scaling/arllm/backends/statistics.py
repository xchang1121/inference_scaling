"""A task's backend that keeps a reward's token statistic from generation.

A token's statistic depends only on the tokens up to it, so a continuation whose
tokens were all generated after the same context reads its statistics from the
recorded outputs, and only a continuation with other tokens is scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample, TokenStatistic
from inference_scaling.shared.types import TokenSequence


class StatisticRecorder:
    """Ask generation for one statistic, keep it and serve ``token_statistics`` from it; other calls pass through."""

    def __init__(self, backend, statistic: TokenStatistic) -> None:
        self.backend, self.statistic = backend, statistic
        # Generated tokens and their statistics, by the length of their prefix and then the prefix.
        self._outputs: dict[int, dict[TokenSequence, list[tuple[np.ndarray, np.ndarray]]]] = {}
        self._streams: dict[object, dict[str, tuple[TokenSequence, list[int], list[float]]]] = {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.backend, name)

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def _remember(self, prefix, tokens, values) -> None:
        if values is not None and not np.isnan(values).all():
            self._outputs.setdefault(len(prefix), {}).setdefault(prefix, []).append(
                (np.asarray(tokens), np.asarray(values, dtype=np.float64)))

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        samples = self.backend.sample_batch([replace(request, statistic=self.statistic) for request in requests])
        for sample in samples:
            self._remember(sample.prefix, sample.token_ids, sample.token_statistics)
        return samples

    def open_generation_session(self):
        return StatisticSession(self)

    def retain_statistics(self, prompt: TokenSequence, completion: TokenSequence, *siblings: TokenSequence) -> None:
        """Keep completed sequences needed by the next step, without scoring missing positions."""
        if self._streams:
            raise RuntimeError("cannot prune statistics while a generation session is open")
        records = [(np.asarray(tokens), self._recorded(prompt + tokens, len(prompt)))
                   for tokens in dict.fromkeys((completion, *siblings))]
        self._outputs.clear()
        records = [(tokens, values) for tokens, values in records if len(values) and not np.isnan(values).all()]
        if records:
            self._outputs[len(prompt)] = {prompt: records}

    def compact_generation_statistics(self, prompt: TokenSequence, *completed: TokenSequence) -> None:
        """Prune completed history while keeping all live request prefixes.

        A submitted request needs its prefix even before its first event. Active
        deltas remain in the session; this only replaces completed output records.
        Rebuilding is useful only when fewer statistic positions would be stored.
        """
        paths = dict.fromkeys(completed)
        for session in self._streams.values():
            for prefix, _, _ in session.values():
                if prefix[:len(prompt)] != prompt:
                    raise ValueError("retained requests must share the declared prompt")
                paths.setdefault(prefix[len(prompt):], None)
        retained: list[TokenSequence] = []
        for tokens in sorted(paths, key=len, reverse=True):
            if not any(existing[:len(tokens)] == tokens for existing in retained):
                retained.append(tokens)
        stored = sum(len(tokens) for outputs in self._outputs.values()
                     for records in outputs.values() for tokens, _ in records)
        if sum(map(len, retained)) >= stored:
            return
        records = [(np.asarray(tokens), self._recorded(prompt + tokens, len(prompt))) for tokens in retained]
        records = [(tokens, values) for tokens, values in records if len(values) and not np.isnan(values).all()]
        self._outputs = {len(prompt): {prompt: records}} if records else {}

    def _recorded(self, sequence: TokenSequence, start: int) -> np.ndarray:
        """Statistics of ``sequence[start:]`` from outputs generated inside it; NaN where there is none."""

        tokens, values = np.asarray(sequence), np.full(len(sequence) - start, np.nan)
        records = [(length, generated, statistics) for length, outputs in self._outputs.items()
                   for generated, statistics in outputs.get(sequence[:length], ()) if length < len(sequence)]
        records.extend((len(prefix), np.asarray(generated), np.asarray(statistics))
                       for session in self._streams.values() for prefix, generated, statistics in session.values()
                       if sequence[:len(prefix)] == prefix and len(prefix) < len(sequence))
        for length, generated, statistics in records:
            same = generated[: len(sequence) - length] == tokens[length : length + len(generated)]
            # The output follows the sequence over positions [length, end).
            end = length + (len(same) if same.all() else int(same.argmin()))
            low = max(length, start)
            if low < end:
                found = statistics[low - length : end - length]
                np.copyto(values[low - start : end - start], found, where=~np.isnan(found))
        return values

    def token_statistics(self, requests: Sequence[ScoreRequest], statistic: TokenStatistic) -> list[tuple[float, ...]]:
        if statistic != self.statistic:
            return self.backend.token_statistics(requests, statistic)
        found = [(request.prefix, continuation, self._recorded(request.prefix + continuation, len(request.prefix)))
                 for request in requests for continuation in request.continuations]
        missing = [ScoreRequest(prefix, (continuation,)) for prefix, continuation, values in found
                   if np.isnan(values).any()]
        scored = iter(self.backend.token_statistics(missing, statistic) if missing else ())
        return [tuple(next(scored)) if np.isnan(values).any() else tuple(values.tolist()) for _, _, values in found]

    def recorded_statistics(self, prompt: TokenSequence, continuation: TokenSequence,
                            statistic: TokenStatistic) -> tuple[float, ...] | None:
        """Read a generated prefix without a forward pass; None if any position is unavailable."""
        if statistic != self.statistic:
            return None
        values = self._recorded(prompt + continuation, len(prompt))
        return tuple(values.tolist()) if np.isfinite(values).all() else None


class StatisticSession:
    def __init__(self, recorder: StatisticRecorder) -> None:
        self.recorder = recorder
        self.inner = recorder.backend.open_generation_session()
        recorder._streams[self] = {}

    def submit(self, requests, *, phase="rollout", checkpoint=None) -> None:
        records = self.recorder._streams[self]
        for request in requests:
            if request.request_id in records:
                raise ValueError("duplicate pending request")
            records[request.request_id] = (request.prefix, [], [])
        self.inner.submit([replace(request, statistic=self.recorder.statistic) for request in requests],
                          phase=phase, checkpoint=checkpoint)

    def next_event(self):
        event = self.inner.next_event()
        sample = event.delta
        records = self.recorder._streams[self]
        prefix, tokens, values = records.setdefault(sample.request_id, (sample.prefix, [], []))
        tokens.extend(sample.token_ids)
        values.extend(sample.token_statistics if sample.token_statistics is not None else [float("nan")] * len(sample.token_ids))
        if event.finished:
            self.recorder._remember(prefix, tokens, values)
            del records[sample.request_id]
        return event

    def seal(self) -> None:
        self.inner.seal()

    def stop(self, request_id: str) -> None:
        self.inner.stop(request_id)

    def close(self) -> None:
        try:
            self.inner.close()
        finally:
            # A finished child may reuse a cancelled parent's generated block.
            # Keep its observed statistics until retain_statistics prunes the pool.
            for prefix, tokens, values in self.recorder._streams.pop(self, {}).values():
                self.recorder._remember(prefix, tokens, values)


__all__ = ["StatisticRecorder", "StatisticSession"]

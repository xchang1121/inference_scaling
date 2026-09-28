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

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.backend, name)

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        samples = self.backend.sample_batch([replace(request, statistic=self.statistic) for request in requests])
        for sample in samples:
            if sample.token_statistics is not None:
                self._outputs.setdefault(len(sample.prefix), {}).setdefault(sample.prefix, []).append(
                    (np.asarray(sample.token_ids), np.asarray(sample.token_statistics, dtype=np.float64)))
        return samples

    def _recorded(self, sequence: TokenSequence, start: int) -> np.ndarray:
        """Statistics of ``sequence[start:]`` from outputs generated inside it; NaN where there is none."""

        tokens, values = np.asarray(sequence), np.full(len(sequence) - start, np.nan)
        for length, outputs in self._outputs.items():
            for generated, statistics in outputs.get(sequence[:length], ()) if length < len(sequence) else ():
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


__all__ = ["StatisticRecorder"]

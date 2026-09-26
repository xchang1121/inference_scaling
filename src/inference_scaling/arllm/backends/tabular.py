"""An exact finite backend used for distributional tests.

This backend is intentionally small enough that target distributions can be
enumerated.  It also implements temperature, top-k, and nucleus truncation so
tests exercise actual behavior-policy probabilities, and it samples like the
Transformers backend: by inverse CDF from the request's uniform stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from inference_scaling.arllm.backends.replay import sample_with_drafts
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample, TokenSequence
from inference_scaling.shared.rng import uniform_stream


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have a positive finite sum")
    return values / total


class TabularAutoregressiveBackend:
    def __init__(
        self,
        probabilities: Mapping[TokenSequence, Sequence[float]],
        *,
        fallback: Sequence[float] | None = None,
        model_id: str = "tabular",
    ) -> None:
        if not probabilities and fallback is None:
            raise ValueError("at least one transition or a fallback distribution is required")
        raw = {tuple(prefix): np.asarray(row, dtype=np.float64) for prefix, row in probabilities.items()}
        first = next(iter(raw.values()), np.asarray(fallback, dtype=np.float64))
        self._vocab_size = int(first.shape[0])
        self._probabilities = {prefix: self._validate_row(row) for prefix, row in raw.items()}
        self._fallback = None if fallback is None else self._validate_row(np.asarray(fallback, dtype=np.float64))
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _validate_row(self, row: np.ndarray) -> np.ndarray:
        if row.ndim != 1 or row.shape[0] != self._vocab_size:
            raise ValueError("all transition rows must have the same one-dimensional vocabulary")
        if np.any(row < 0) or np.any(~np.isfinite(row)):
            raise ValueError("transition probabilities must be finite and non-negative")
        return _normalize(row.copy())

    def _base_probabilities(self, prefix: TokenSequence) -> np.ndarray:
        if prefix in self._probabilities:
            return self._probabilities[prefix]
        if self._fallback is not None:
            return self._fallback
        raise KeyError(f"no transition probabilities for prefix {prefix!r}")

    def probabilities(self, prefix: TokenSequence, sampling: SamplingConfig | None = None) -> np.ndarray:
        base = self._base_probabilities(prefix)
        if sampling is None:
            return base.copy()
        positive = base > 0
        scaled = np.zeros_like(base)
        scaled[positive] = np.exp(np.log(base[positive]) / sampling.temperature)
        if sampling.top_k is not None and sampling.top_k < self._vocab_size:
            keep = np.argpartition(scaled, -sampling.top_k)[-sampling.top_k :]
            scaled[np.setdiff1d(np.arange(self._vocab_size), keep)] = 0
        scaled = _normalize(scaled)
        if sampling.top_p < 1:
            order = np.argsort(-scaled, kind="stable")
            count = int(np.searchsorted(np.cumsum(scaled[order]), sampling.top_p, side="left")) + 1
            scaled[order[count:]] = 0
            scaled = _normalize(scaled)
        return scaled

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        return sample_with_drafts(requests, self._generate, model_id=self.model_id, honors_stops=False)[0]

    def _generate(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        outputs: list[SequenceSample] = []
        for request in requests:
            context = list(request.prefix)
            tokens: list[int] = []
            logprobs: list[float] = []
            references: list[float] = []
            bounds: list[tuple[float, float]] = []
            finish_reason = "length"
            stop, running = request.log_weight_stop, 0.0 if request.log_weight_stop is None else request.log_weight_stop.start
            for uniform in uniform_stream(request.seed, request.uniform_offset, request.max_new_tokens):
                probs = self.probabilities(tuple(context), request.sampling)
                cdf = np.cumsum(probs)
                cdf /= cdf[-1]
                token = min(int(np.searchsorted(cdf, uniform, side="left")), self._vocab_size - 1)
                tokens.append(token)
                logprobs.append(float(np.log(probs[token])))
                references.append(float(np.log(self.probabilities(tuple(context), request.reference_policy)[token])))
                bounds.append((float(cdf[token - 1]) if token else -1.0, float(cdf[token])))
                context.append(token)
                if request.sampling.eos_token_id == token:
                    finish_reason = "eos"
                    break
                if stop is not None and stop.rejects(running := stop.advance(running, references[-1], logprobs[-1])):
                    finish_reason = "rejected"
                    break
            outputs.append(SequenceSample(
                prefix=request.prefix, token_ids=tuple(tokens), token_logprobs=tuple(logprobs),
                policy_id=request.sampling.policy_id, model_id=self.model_id, request_id=request.request_id,
                finish_reason=finish_reason, reference_token_logprobs=tuple(references),
                reference_policy_id=request.reference_policy.policy_id, token_cdf_bounds=tuple(bounds),
            ))
        return outputs

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        outputs: list[tuple[float, ...]] = []
        for request in requests:
            for continuation in request.continuations:
                context = list(request.prefix)
                token_logprobs: list[float] = []
                for token in continuation:
                    probability = float(self.probabilities(tuple(context), request.sampling)[token])
                    token_logprobs.append(float("-inf") if probability == 0 else float(np.log(probability)))
                    context.append(token)
                outputs.append(tuple(token_logprobs))
        return outputs

"""Reuse deterministic experiment requests while charging their cold-run cost.

This is an experiment-only view: snapshot() includes the forward work a method
would perform independently. The underlying backend retains physical counters.
It is not a deployment speedup measurement or a zero-cost rollout proposal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace


class ColdCostRequestReplay:
    """Memoize batch-one, request-seeded ordinary decoding on one fixed model."""

    def __init__(self, backend):
        if getattr(backend, "_speculation", None) is not None:
            raise ValueError("request replay requires ordinary, non-speculative decoding")
        self.backend = backend
        self.model_id = backend.model_id
        self.tokenizer = backend.tokenizer
        self.parameter_count = backend.parameter_count
        self._cache = {}
        self._offset: Counter[str] = Counter()
        self.cache_hits = 0

    @staticmethod
    def _key(request):
        return (request.prefix, request.sampling, request.seed, request.uniforms, request.arithmetic_uniform)

    def sample_batch(self, requests):
        if len(requests) != 1:
            raise ValueError("cold-cost replay is defined for batch-one experiment requests")
        request = requests[0]
        key = self._key(request)
        for maximum, sample, cost in self._cache.get(key, ()):
            compatible = maximum == request.max_new_tokens or (
                sample.finish_reason == "eos" and len(sample.token_ids) <= request.max_new_tokens
            )
            if compatible:
                self.cache_hits += 1
                self._offset.update(cost)
                return [replace(sample, request_id=request.request_id)]
        before = asdict(self.backend.snapshot())
        samples = self.backend.sample_batch(requests)
        after = asdict(self.backend.snapshot())
        cost = {key: after[key] - before[key] for key in before}
        self._cache.setdefault(key, []).append((request.max_new_tokens, samples[0], cost))
        return samples

    def score_batch(self, requests):
        return self.backend.score_batch(requests)

    def score_statistics_batch(self, requests, **kwargs):
        return self.backend.score_statistics_batch(requests, **kwargs)

    def snapshot(self):
        current = self.backend.snapshot()
        return replace(current, **{key: getattr(current, key) + value for key, value in self._offset.items()})

    def encode(self, text, **kwargs):
        return self.backend.encode(text, **kwargs)

    def decode(self, tokens, **kwargs):
        return self.backend.decode(tokens, **kwargs)

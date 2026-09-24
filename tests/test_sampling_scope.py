from math import exp
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.backends.reference import ReferencePolicyBackend
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest
from inference_scaling.shared.model.output import ThinkingFormat


class _Tokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"<think>": 3, "</think>": 1, "7": 0, "<eos>": 2}


class _Backend(TabularAutoregressiveBackend):
    tokenizer = _Tokenizer()
    parameter_count = 1

    def __init__(self):
        super().__init__({
            (3,): (1, 0, 0, 0), (3, 0): (0, 1, 0, 0),
            (3, 0, 1): (1, 0, 0, 0), (3, 0, 1, 0): (0, 0, 1, 0),
        }, fallback=(0, 0, 1, 0))

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join("7" for token in tokens if token == 0)

    def encode(self, text, *, add_special_tokens=False):
        return (self.tokenizer.get_vocab()[text],)

    def score_statistics_batch(self, requests, *, confidence_top_k=None):
        # In this fixture token 0 is the entire thinking span. The end marker
        # and the identically worded final content must stay outside scoring.
        assert all(request.prefix == (3,) for request in requests)
        assert all(tokens == (0,) for request in requests for tokens in request.continuations)
        return [
            SimpleNamespace(token_topk_confidences=(1.0,))
            for request in requests for _ in request.continuations
        ]


def test_reference_policy_matches_direct_temperature_scoring():
    raw = TabularAutoregressiveBackend({}, fallback=(0.75, 0.25))
    backend = ReferencePolicyBackend(raw, temperature=0.6)
    score = backend.score_batch([ScoreRequest((), ((0,), (1,)))])
    expected = raw.score_batch([ScoreRequest((), ((0,), (1,)), SamplingConfig(temperature=0.6))])
    assert score == expected
    assert sum(exp(sum(item)) for item in score) == pytest.approx(1)
    request = GenerationRequest((), 3, SamplingConfig(), 1, "sample")
    sample = backend.sample_batch([request])[0]
    assert sample.reference_token_logprobs == sample.token_logprobs
    assert sample.policy_id == SamplingConfig().policy_id


def test_scope_finishes_content_from_original_backend_with_remaining_budget():
    backend = _Backend()
    scope = SamplingScope("thinking", ThinkingFormat((1,), (3,)), generation_chunk_size=1)
    stopped = scope.wrap(backend, (3,))
    thinking = stopped.sample_batch([GenerationRequest((3,), 6, SamplingConfig(), 0, "thinking")])[0]
    assert thinking.token_ids == (0, 1, 2, 2, 2, 2)
    tokens, info = scope.finish(
        backend, (3,), thinking.token_ids, max_new_tokens=6,
        sampling=SamplingConfig(eos_token_id=2), seed=1,
    )
    assert tokens == (0, 1, 0, 2)
    assert info["thinking_token_ids"] == (0,)
    assert info["content_token_ids"] == (0,)
    assert info["final_content_generated_tokens"] == 2
    assert info["thinking_status"] == "complete"

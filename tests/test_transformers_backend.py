from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_scaling.backends import TransformersBackend
from inference_scaling.config import SamplingConfig
from inference_scaling.types import GenerationRequest, ScoreRequest


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 0
    eos_token_id = 2


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("constant_logits", torch.log(torch.tensor(probabilities)))
        self.config = SimpleNamespace(_name_or_path="constant-logit-model")
        self.forward_calls = 0

    @property
    def device(self):
        return self.constant_logits.device

    def forward(self, input_ids, **_kwargs):
        self.forward_calls += 1
        batch, length = input_ids.shape
        logits = self.constant_logits.expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits, past_key_values=RepeatableCache())


class RepeatableCache:
    def batch_repeat_interleave(self, _repeats):
        return None


def test_request_local_randomness_is_independent_of_batch_order() -> None:
    model = ConstantLogitModel([0.55, 0.3, 0.15])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    requests = [
        GenerationRequest((0,), 5, SamplingConfig(), seed, f"request-{seed}")
        for seed in (3, 9, 27)
    ]
    together = backend.sample_batch(requests)
    reversed_outputs = backend.sample_batch(list(reversed(requests)))
    by_id = {sample.request_id: sample for sample in reversed_outputs}

    for sample in together:
        assert sample == by_id[sample.request_id]


def test_sampled_logprobabilities_match_exact_rescoring() -> None:
    model = ConstantLogitModel([0.5, 0.35, 0.15])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    sampling = SamplingConfig(temperature=0.7, top_k=2)
    samples = backend.sample_batch(
        [
            GenerationRequest((0,), 4, sampling, 100 + index, f"sample-{index}")
            for index in range(6)
        ]
    )
    scores = backend.score_batch(
        [ScoreRequest(sample.prefix, (sample.token_ids,), sampling) for sample in samples]
    )
    for sample, token_scores in zip(samples, scores, strict=True):
        assert sample.token_logprobs == pytest.approx(token_scores)
    assert model.forward_calls == 5


def test_top_p_scoring_uses_the_actual_truncated_policy() -> None:
    model = ConstantLogitModel([0.6, 0.3, 0.1])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    scores = backend.score_batch(
        [ScoreRequest((0,), ((0,), (1,), (2,)), SamplingConfig(top_p=0.7))]
    )
    assert scores[0][0] == pytest.approx(np.log(2 / 3))
    assert scores[1][0] == pytest.approx(np.log(1 / 3))
    assert scores[2][0] == float("-inf")


def test_eos_stops_generation_and_statistics_count_real_tokens() -> None:
    model = ConstantLogitModel([0.0, 0.0, 1.0])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    samples = backend.sample_batch(
        [
            GenerationRequest(
                (0,),
                8,
                SamplingConfig(eos_token_id=2),
                4,
                "eos",
            )
        ]
    )
    snapshot = backend.snapshot()
    assert samples[0].token_ids == (2,)
    assert samples[0].finish_reason == "eos"
    assert snapshot.generated_tokens == 1
    assert snapshot.prefill_tokens == 1


def test_identical_prefix_prefill_is_computed_once_then_forked() -> None:
    model = ConstantLogitModel([0.5, 0.3, 0.2])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    backend.sample_batch(
        [
            GenerationRequest((0, 1, 0), 1, SamplingConfig(), index, str(index))
            for index in range(5)
        ]
    )
    snapshot = backend.snapshot()
    assert model.forward_calls == 1
    assert snapshot.prefill_tokens == 3
    assert snapshot.shared_prefill_tokens_saved == 12

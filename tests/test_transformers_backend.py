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
        self.constant_logits = torch.nn.Parameter(
            torch.log(torch.tensor(probabilities)), requires_grad=False
        )
        self.config = SimpleNamespace(
            _name_or_path="constant-logit-model", model_type="qwen2"
        )
        self.forward_calls = 0
        self.logits_to_keep_calls = []

    @property
    def device(self):
        return self.constant_logits.device

    def forward(self, input_ids, **_kwargs):
        self.forward_calls += 1
        batch, length = input_ids.shape
        logits = self.constant_logits.expand(batch, length, -1).clone()
        logits_to_keep = int(_kwargs.get("logits_to_keep", 0))
        self.logits_to_keep_calls.append(logits_to_keep)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
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


def test_inverse_cdf_accumulates_large_vocabulary_in_float64() -> None:
    vocabulary_size = 1000
    seed = 12434
    model = ConstantLogitModel([1 / vocabulary_size] * vocabulary_size)
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")

    sample = backend.sample_batch(
        [GenerationRequest((0,), 1, SamplingConfig(), seed, "large-vocabulary")]
    )[0]
    probabilities = torch.log_softmax(model.constant_logits, dim=-1).exp().double().numpy()
    uniform = np.random.default_rng(seed).random()
    expected = int((np.cumsum(probabilities, dtype=np.float64) < uniform).sum())

    # This seed lies on a boundary where a float32 CDF returns token 668.
    assert expected == 669
    assert sample.token_ids == (expected,)


def test_sampled_logprobabilities_match_exact_rescoring() -> None:
    model = ConstantLogitModel([0.5, 0.35, 0.15])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    sampling = SamplingConfig(temperature=0.7, top_k=2)
    samples = backend.sample_batch(
        [
            GenerationRequest((0,), 4, sampling, 100 + index, f"sample-{index}")
            for index in range(4)
        ]
    )
    scores = backend.score_batch(
        [
            *(
                ScoreRequest(sample.prefix, (sample.token_ids,), sampling)
                for sample in samples
            ),
            *(
                ScoreRequest(sample.prefix, (sample.token_ids,), SamplingConfig())
                for sample in samples
            ),
        ]
    )
    for sample, token_scores, reference_scores in zip(
        samples, scores[: len(samples)], scores[len(samples) :], strict=True
    ):
        assert sample.token_logprobs == pytest.approx(token_scores)
        assert sample.reference_token_logprobs == pytest.approx(reference_scores)
        assert sample.reference_policy_id == SamplingConfig().policy_id
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
    assert snapshot.generation_forward_token_slots == 1
    assert snapshot.estimated_dense_forward_flops == 6


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
    assert snapshot.generation_forward_token_slots == 3
    assert snapshot.estimated_dense_forward_flops == 18


def test_each_repeated_prefix_group_is_prefilled_once_then_forked() -> None:
    model = ConstantLogitModel([0.5, 0.3, 0.2])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    prefixes = ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0))
    outputs = backend.sample_batch(
        [
            GenerationRequest(prefix, 1, SamplingConfig(), index, str(index))
            for index, prefix in enumerate(prefixes)
        ]
    )

    assert [sample.request_id for sample in outputs] == [str(i) for i in range(6)]
    snapshot = backend.snapshot()
    assert model.forward_calls == 1
    assert snapshot.prefill_tokens == 4
    assert snapshot.shared_prefill_tokens_saved == 8
    assert snapshot.generation_forward_token_slots == 4
    assert snapshot.estimated_dense_forward_flops == 24


def test_scoring_counts_padded_forward_slots_and_dense_flops() -> None:
    model = ConstantLogitModel([0.6, 0.3, 0.1])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    backend.score_batch(
        [ScoreRequest((0,), ((0,), (1,), (0, 1)), SamplingConfig())]
    )

    snapshot = backend.snapshot()
    assert snapshot.scored_tokens == 4
    assert snapshot.score_forward_token_slots == 9
    assert snapshot.estimated_dense_forward_flops == 54
    assert model.logits_to_keep_calls == [3]


def test_scoring_keeps_only_required_tail_logits() -> None:
    model = ConstantLogitModel([0.6, 0.3, 0.1])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")
    scores = backend.score_batch(
        [ScoreRequest((0, 1, 0, 1), ((0,), (1,), (0, 1)), SamplingConfig())]
    )

    assert [len(score) for score in scores] == [1, 1, 2]
    assert model.logits_to_keep_calls == [3]


def test_confidence_statistics_match_reference_policy_definitions() -> None:
    probabilities = np.asarray([0.5, 0.3, 0.2])
    model = ConstantLogitModel(probabilities)
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")

    result = backend.score_statistics_batch(
        [ScoreRequest((0,), ((0, 1),), SamplingConfig())]
    )[0]

    assert result.token_logprobs == pytest.approx(np.log([0.5, 0.3]))
    assert result.mean_logprob == pytest.approx(np.log([0.5, 0.3]).mean())
    assert result.mean_negative_entropy == pytest.approx(
        np.sum(probabilities * np.log(probabilities))
    )
    assert result.mean_self_certainty == pytest.approx(
        -np.mean(np.log(len(probabilities)) + np.log(probabilities))
    )
    snapshot = backend.snapshot()
    assert snapshot.scored_tokens == 2
    assert snapshot.score_forward_token_slots == 3


def test_confidence_statistics_reject_truncated_support() -> None:
    model = ConstantLogitModel([0.5, 0.3, 0.2])
    backend = TransformersBackend(model, TinyTokenizer(), device="cpu")

    with pytest.raises(ValueError, match="full-support"):
        backend.score_statistics_batch(
            [ScoreRequest((0,), ((1,),), SamplingConfig(top_k=2))]
        )

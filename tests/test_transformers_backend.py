from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest


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
        self.batch_sizes = []

    @property
    def device(self):
        return self.constant_logits.device

    def forward(self, input_ids, logits_to_keep=0, **_kwargs):
        self.forward_calls += 1
        batch, length = input_ids.shape
        self.batch_sizes.append(batch)
        logits = self.constant_logits.expand(batch, length, -1).clone()
        logits_to_keep = int(logits_to_keep)
        self.logits_to_keep_calls.append(logits_to_keep)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits, past_key_values=RepeatableCache())


class RepeatableCache:
    def batch_select_indices(self, _indices):
        return None

    def crop(self, _length):
        return None


def _backend(model):
    return TransformersBackend(model, TinyTokenizer(), device="cpu", max_score_batch_size=8, score_chunk_size=256)


def test_request_local_randomness_is_independent_of_batch_order() -> None:
    model = ConstantLogitModel([0.55, 0.3, 0.15])
    backend = _backend(model)
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
    backend = _backend(model)

    sample = backend.sample_batch(
        [GenerationRequest((0,), 1, SamplingConfig(), seed, "large-vocabulary")]
    )[0]
    probabilities = (
        torch.log_softmax(model.constant_logits, dim=-1).exp().double().numpy()
    )
    uniform = np.random.default_rng(seed).random()
    expected = int((np.cumsum(probabilities, dtype=np.float64) < uniform).sum())

    # This seed lies on a boundary where a float32 CDF returns token 668.
    assert expected == 669
    assert sample.token_ids == (expected,)


def test_sampled_logprobabilities_match_exact_rescoring() -> None:
    model = ConstantLogitModel([0.5, 0.35, 0.15])
    backend = _backend(model)
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
    backend = _backend(model)
    scores = backend.score_batch(
        [ScoreRequest((0,), ((0,), (1,), (2,)), SamplingConfig(top_p=0.7))]
    )
    assert scores[0][0] == pytest.approx(np.log(2 / 3))
    assert scores[1][0] == pytest.approx(np.log(1 / 3))
    assert scores[2][0] == float("-inf")


def test_eos_stops_generation_and_statistics_count_real_tokens() -> None:
    model = ConstantLogitModel([0.0, 0.0, 1.0])
    backend = _backend(model)
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
    backend = _backend(model)
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


def test_each_repeated_prefix_is_prefilled_once_then_forked() -> None:
    model = ConstantLogitModel([0.5, 0.3, 0.2])
    backend = _backend(model)
    # Prefixes may repeat unequally often.
    prefixes = ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1))
    outputs = backend.sample_batch(
        [
            GenerationRequest(prefix, 1, SamplingConfig(), index, str(index))
            for index, prefix in enumerate(prefixes)
        ]
    )

    assert [sample.request_id for sample in outputs] == [str(i) for i in range(5)]
    snapshot = backend.snapshot()
    assert model.batch_sizes == [2]
    assert snapshot.prefill_tokens == 4
    assert snapshot.shared_prefill_tokens_saved == 6
    assert snapshot.generation_forward_token_slots == 4
    assert snapshot.estimated_dense_forward_flops == 24


def test_finished_rows_leave_the_batch() -> None:
    model = ConstantLogitModel([0.3, 0.3, 0.4])
    backend = _backend(model)
    samples = backend.sample_batch(
        [GenerationRequest((0,), 8, SamplingConfig(eos_token_id=2), seed, str(seed)) for seed in range(6)]
    )
    lengths = [len(sample.token_ids) for sample in samples]
    # After the shared prefill, each decode step runs only the rows still generating.
    assert model.batch_sizes == [1] + [sum(length > step for length in lengths) for step in range(1, max(lengths))]
    assert model.batch_sizes[-1] < len(samples)
    assert backend.snapshot().generation_forward_token_slots == sum(model.batch_sizes)


def test_stop_sequences_end_generation_and_reference_uses_its_temperature() -> None:
    model = ConstantLogitModel([0.2, 0.7, 0.1])
    backend = _backend(model)
    request = GenerationRequest((0,), 12, SamplingConfig(temperature=0.5), 3, "stop", stop_sequences=((1, 1),),
                                reference_temperature=2.0)
    sample = backend.sample_batch([request])[0]
    pairs = list(zip(sample.token_ids, sample.token_ids[1:]))
    assert sample.finish_reason == "stop" and pairs.index((1, 1)) == len(pairs) - 1
    reference = backend.score_batch([ScoreRequest((0,), (sample.token_ids,), request.reference_policy)])[0]
    assert sample.reference_token_logprobs == pytest.approx(reference)
    assert sample.reference_policy_id == request.reference_policy.policy_id


def test_scoring_counts_padded_forward_slots_and_dense_flops() -> None:
    model = ConstantLogitModel([0.6, 0.3, 0.1])
    backend = _backend(model)
    backend.score_batch([ScoreRequest((0,), ((0,), (1,), (0, 1)), SamplingConfig())])

    snapshot = backend.snapshot()
    # The inputs are the prefix and every target but the last: (0,), (0,) and (0, 0).
    assert snapshot.scored_tokens == 4
    assert snapshot.score_forward_token_slots == 6
    assert snapshot.estimated_dense_forward_flops == 36
    assert model.logits_to_keep_calls == [2]


def test_scoring_keeps_only_required_tail_logits() -> None:
    model = ConstantLogitModel([0.6, 0.3, 0.1])
    backend = _backend(model)
    scores = backend.score_batch(
        [ScoreRequest((0, 1, 0, 1), ((0,), (1,), (0, 1)), SamplingConfig())]
    )

    assert [len(score) for score in scores] == [1, 1, 2]
    assert model.logits_to_keep_calls == [2]


def test_confidence_statistics_match_reference_policy_definitions() -> None:
    probabilities = np.asarray([0.5, 0.3, 0.2])
    model = ConstantLogitModel(probabilities)
    backend = _backend(model)

    result = backend.score_statistics_batch(
        [ScoreRequest((0,), ((0, 1),), SamplingConfig())],
        confidence_top_k=2,
    )[0]

    assert result.token_topk_confidences == pytest.approx(
        [-np.mean(np.log([0.5, 0.3]))] * 2
    )
    snapshot = backend.snapshot()
    assert snapshot.scored_tokens == 2
    assert snapshot.score_forward_token_slots == 2


def test_confidence_statistics_reject_truncated_support() -> None:
    model = ConstantLogitModel([0.5, 0.3, 0.2])
    backend = _backend(model)

    with pytest.raises(ValueError, match="full-support"):
        backend.score_statistics_batch(
            [ScoreRequest((0,), ((1,),), SamplingConfig(top_k=2))], confidence_top_k=2
        )


def test_confidence_statistics_reject_nonpositive_top_k() -> None:
    model = ConstantLogitModel([0.5, 0.3, 0.2])
    backend = _backend(model)

    with pytest.raises(ValueError, match="confidence_top_k must be positive"):
        backend.score_statistics_batch(
            [ScoreRequest((0,), ((1,),), SamplingConfig())],
            confidence_top_k=0,
        )

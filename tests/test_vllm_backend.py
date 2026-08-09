from __future__ import annotations

from dataclasses import dataclass

import pytest

from inference_scaling.backends import VLLMBackend
from inference_scaling.config import SamplingConfig
from inference_scaling.types import GenerationRequest, ScoreRequest


@dataclass
class _Logprob:
    logprob: float


@dataclass
class _Completion:
    token_ids: list[int]
    logprobs: list[dict[int, _Logprob]]
    finish_reason: str = "length"


@dataclass
class _Output:
    outputs: list[_Completion]
    prompt_logprobs: list[dict[int, _Logprob] | None] | None = None
    num_cached_tokens: int = 0


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Tokenizer:
    bos_token_id = 9
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=True):
        values = [ord(value) % 10 for value in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def decode(self, tokens, skip_special_tokens=True):
        return ",".join(str(token) for token in tokens)


class _Engine:
    def __init__(self):
        self.calls = []
        self.closed = False

    @staticmethod
    def _ids(prompt):
        return list(prompt["prompt_token_ids"])

    def generate(self, prompts, *, sampling_params, use_tqdm, **kwargs):
        self.calls.append((prompts, sampling_params, use_tqdm, kwargs))
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompts)
        )
        outputs = []
        for prompt, policy in zip(prompts, params, strict=True):
            prompt_ids = self._ids(prompt)
            if hasattr(policy, "prompt_logprobs"):
                prompt_scores = [None] + [
                    {token: _Logprob(-float(index) / 10)}
                    for index, token in enumerate(prompt_ids[1:], 1)
                ]
                outputs.append(
                    _Output(
                        [_Completion([7], [{7: _Logprob(-0.7)}])],
                        prompt_logprobs=prompt_scores,
                    )
                )
                continue
            count = min(2, policy.max_tokens)
            tokens = [int(policy.seed % 5) + 3] * count
            if policy.stop_token_ids and policy.seed == 12:
                tokens[-1] = policy.stop_token_ids[0]
            outputs.append(
                _Output(
                    [_Completion(tokens, [{token: _Logprob(-0.25)} for token in tokens])],
                    num_cached_tokens=min(2, len(prompt_ids)),
                )
            )
        return outputs

    def shutdown(self):
        self.closed = True


class _Fallback:
    model_id = "fake"

    def sample_batch(self, requests):
        raise AssertionError("fallback generation must not be used")

    def score_batch(self, requests):
        return [tuple(-0.5 for _ in continuation) for request in requests for continuation in request.continuations]


def _backend(*, fallback=None):
    engine = _Engine()
    backend = VLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        scoring_backend=fallback,
    )
    return backend, engine


def test_vllm_sampling_preserves_per_request_seed_policy_and_order() -> None:
    backend, engine = _backend()
    sampling = SamplingConfig(temperature=0.7, top_p=0.9, top_k=4, eos_token_id=2)
    requests = [
        GenerationRequest((1, 2), 2, sampling, seed, f"r{seed}")
        for seed in (11, 12)
    ]

    samples = backend.sample_batch(requests)

    assert [sample.request_id for sample in samples] == ["r11", "r12"]
    assert samples[0].token_logprobs == (-0.25, -0.25)
    assert samples[1].token_ids[-1] == 2
    assert samples[1].finish_reason == "eos"
    params = engine.calls[0][1]
    assert [item.seed for item in params] == [11, 12]
    assert all(item.temperature == 0.7 for item in params)
    assert all(item.top_p == 0.9 and item.top_k == 4 for item in params)
    assert all(item.stop_token_ids == [2] and item.ignore_eos for item in params)
    snapshot = backend.snapshot()
    assert snapshot.sampled_sequences == 2
    assert snapshot.generated_tokens == 4
    assert snapshot.shared_prefill_tokens_saved == 4
    assert snapshot.prefill_tokens == 0


def test_vllm_native_score_extracts_continuation_prompt_logprobs() -> None:
    backend, _ = _backend()

    scores = backend.score_batch(
        [ScoreRequest((8, 6), ((4, 5), (), (3,)), SamplingConfig())]
    )

    assert scores == [(-0.2, -0.3), (), (-0.2,)]
    snapshot = backend.snapshot()
    assert snapshot.native_score_sequences == 2
    assert snapshot.scored_tokens == 3
    assert snapshot.score_forward_token_slots == 7


def test_vllm_nonunit_score_requires_or_uses_exact_fallback() -> None:
    backend, _ = _backend()
    request = ScoreRequest((1,), ((2, 3),), SamplingConfig(temperature=0.7))
    with pytest.raises(ValueError, match="exact scoring_backend"):
        backend.score_batch([request])

    backend, _ = _backend(fallback=_Fallback())
    assert backend.score_batch([request]) == [(-0.5, -0.5)]
    assert backend.snapshot().delegated_score_sequences == 1


def test_vllm_encode_decode_and_close() -> None:
    backend, engine = _backend()
    assert backend.encode("ab", add_special_tokens=False) == (7, 8)
    assert backend.decode((1, 2, 3)) == "1,2,3"
    backend.close()
    backend.close()
    assert engine.closed

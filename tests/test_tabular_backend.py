import numpy as np

from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import Draft, GenerationRequest, LogWeightStop, ScoreRequest
from inference_scaling.shared.rng import uniform_stream


def test_sample_logprob_matches_actual_truncated_policy() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.3, 0.1])
    sampling = SamplingConfig(temperature=1.0, top_k=2)
    request = GenerationRequest((), 4, sampling, 5, "sample")
    sample = backend.sample_batch([request])[0]

    assert 2 not in sample.token_ids
    rescored = backend.score_batch([ScoreRequest((), (sample.token_ids,), sampling)])[0]
    np.testing.assert_allclose(rescored, sample.token_logprobs)


def test_base_and_behavior_scores_are_distinct() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    continuation = (1, 0)
    base = backend.score_batch([ScoreRequest((), (continuation,), None)])[0]
    behavior = backend.score_batch(
        [ScoreRequest((), (continuation,), SamplingConfig(temperature=0.5))]
    )[0]
    assert not np.allclose(base, behavior)


def test_a_split_request_continues_the_stream_and_its_bounds_hold_the_uniforms() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.3, 0.2])
    whole = backend.sample_batch([GenerationRequest((), 6, SamplingConfig(), 7, "whole")])[0]
    head = backend.sample_batch([GenerationRequest((), 2, SamplingConfig(), 7, "head")])[0]
    tail = backend.sample_batch([GenerationRequest(head.token_ids, 4, SamplingConfig(), 7, "tail", uniform_offset=2)])[0]
    assert head.token_ids + tail.token_ids == whole.token_ids
    assert all(below < uniform <= through
               for (below, through), uniform in zip(whole.token_cdf_bounds, uniform_stream(7, 0, 6), strict=True))


def test_a_draft_replays_exactly_the_tokens_plain_generation_would_draw() -> None:
    backend = TabularAutoregressiveBackend({(1,): (0.2, 0.7, 0.1)}, fallback=[0.5, 0.3, 0.2])
    old = backend.sample_batch([GenerationRequest((), 6, SamplingConfig(eos_token_id=2), 3, "old")])[0]
    draft = Draft(old.token_ids, old.token_logprobs, old.reference_token_logprobs, old.token_cdf_bounds)
    for seed in range(8):
        plain = backend.sample_batch([GenerationRequest((), 6, SamplingConfig(eos_token_id=2), seed, "new")])[0]
        assert backend.sample_batch([GenerationRequest((), 6, SamplingConfig(eos_token_id=2), seed, "new",
                                                       draft=draft)])[0] == plain


def test_a_log_weight_stop_ends_generation_at_the_same_token_with_or_without_a_draft() -> None:
    backend = TabularAutoregressiveBackend({(1,): (0.2, 0.7, 0.1)}, fallback=[0.5, 0.3, 0.2])
    sampling = SamplingConfig(temperature=0.5)
    old = backend.sample_batch([GenerationRequest((), 8, sampling, 3, "old")])[0]
    draft = Draft(old.token_ids, old.token_logprobs, old.reference_token_logprobs, old.token_cdf_bounds)
    stop = LogWeightStop(2.0, 0.0, -1.5)
    for seed in range(8):
        plain = backend.sample_batch([GenerationRequest((), 8, sampling, seed, "new", log_weight_stop=stop)])[0]
        weight = stop.weight(plain.reference_token_logprobs, plain.token_logprobs)
        assert (plain.finish_reason == "rejected") == stop.rejects(weight)
        assert plain.finish_reason != "rejected" or not stop.rejects(stop.weight(
            plain.reference_token_logprobs[:-1], plain.token_logprobs[:-1]))
        assert backend.sample_batch([GenerationRequest((), 8, sampling, seed, "new", draft=draft,
                                                       log_weight_stop=stop)])[0] == plain

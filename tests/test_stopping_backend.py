from dataclasses import replace
from itertools import product
from math import exp, log

import pytest

from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig, PowerMHConfig
from inference_scaling.arllm.algorithms.mh import run_power_mh_chain
from inference_scaling.arllm.backends.stopping import StoppedSequenceBackend
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import pointwise

# The prompt (3,) opens thinking, token 1 closes it and token 2 is EOS.
PARSER = ThinkingParser((ThinkingFormat((1,), (3,)),))
RAW = TabularAutoregressiveBackend({}, fallback=(0.5, 0.3, 0.2))


def _backend(*, parser=PARSER, raw=RAW):
    return StoppedSequenceBackend(raw, thinking_parser=parser, thinking_prompt=(3,), eos_token_id=2)


class StoppingTabular(TabularAutoregressiveBackend):
    """A tabular model whose generation ends right after a stop sequence, as the real backends do."""

    def sample_batch(self, requests):
        samples = []
        for request, sample in zip(requests, super().sample_batch(requests), strict=True):
            ends = [end for end in range(1, len(sample.token_ids) + 1) for stop in request.stop_sequences
                    if sample.token_ids[:end][-len(stop):] == stop]
            end = min(ends, default=None)
            samples.append(sample if end is None else replace(
                sample, token_ids=sample.token_ids[:end], token_logprobs=sample.token_logprobs[:end],
                reference_token_logprobs=sample.reference_token_logprobs[:end],
                token_cdf_bounds=sample.token_cdf_bounds[:end], finish_reason="stop"))
        return samples


def _path(*steps):
    """A model that follows ``steps`` deterministically: each (context, token) forces one transition."""
    return StoppingTabular(
        {context: tuple(float(index == token) for index in range(3)) for context, token in steps},
        fallback=(0.5, 0.3, 0.2),
    )


def _outputs(backend, length):
    """Complete outputs: each ends at its first stop or at the length limit."""
    return [
        tokens for size in range(1, length + 1) for tokens in product(range(3), repeat=size)
        if backend._stop_end(tokens) == size or (backend._stop_end(tokens) is None and size == length)
    ]


def test_stopped_outputs_form_a_normalized_measure():
    backend = _backend()
    scores = backend.score_batch([ScoreRequest((3,), tuple(_outputs(backend, 3)))])
    assert sum(exp(sum(score)) for score in scores) == pytest.approx(1)
    # The boundary keeps its model probability; nothing follows it.
    stopped, past = backend.score_batch([ScoreRequest((3,), ((0, 1), (0, 1, 0)))])
    assert stopped == pytest.approx((log(0.5), log(0.3)))
    assert past == pytest.approx((log(0.5), log(0.3), float("-inf")))


def test_generation_stops_at_a_marker_and_scores_identically():
    raw = _path(((3,), 1), ((3, 1), 0), ((3, 1, 0), 1))
    backend = _backend(parser=ThinkingParser((ThinkingFormat((0, 1), (3,)),)), raw=raw)
    policy = SamplingConfig()
    # The two-token end marker ends the generation.
    sample = backend.sample_batch([GenerationRequest((3,), 5, policy, 4, "test")])[0]
    assert (sample.token_ids, sample.finish_reason) == ((1, 0, 1), "stop")
    assert backend.score_batch([ScoreRequest((3,), (sample.token_ids,), policy)])[0] == sample.token_logprobs
    # ... and across the request prefix and the generation.
    sample = backend.sample_batch([GenerationRequest((3, 1, 0), 3, policy, 4, "cross")])[0]
    assert sample.token_ids == (1,)
    assert backend.score_batch([ScoreRequest((3, 1, 0), ((1, 0),))])[0] == (0.0, float("-inf"))


def test_a_stopped_prefix_has_no_continuation():
    backend = _backend()
    with pytest.raises(ValueError, match="already ended"):
        backend.sample_batch([GenerationRequest((3, 0, 1), 2, SamplingConfig(), 0, "terminal")])
    assert backend.score_batch([ScoreRequest((3, 0, 2), ((0, 0),))]) == [(float("-inf"),) * 2]
    with pytest.raises(ValueError, match="scoped prompt"):
        backend.sample_batch([GenerationRequest((0,), 2, SamplingConfig(), 0, "unscoped")])


def test_is_and_mh_return_complete_outputs_of_the_stopped_backend():
    backend = _backend()
    result = run_conditional_is(
        backend, (3,), ConditionalISConfig(total_length=4, block_size=2, candidate_count=2, rollout_count=2, reward_temperature=1.0),
        pointwise(lambda prompt, sequence: float(sequence[0] == 0)), SeedStream(7),
        sampling=SamplingConfig(eos_token_id=2),
    )
    mh = run_power_mh_chain(
        backend, (3,), PowerMHConfig(total_length=4, block_size=2, steps_per_block=2, alpha=4.0, suffix_schedule="uniform", iterations=None),
        SamplingConfig(temperature=0.5, eos_token_id=2), SeedStream(8),
    )
    for tokens in (result.token_ids, mh.token_ids):
        assert tokens in _outputs(backend, 4)
        assert exp(sum(backend.score_batch([ScoreRequest((3,), (tokens,))])[0])) > 0


def test_thinking_scope_projects_the_full_reward_target():
    backend = _backend()
    full = [
        tokens for size in range(1, 5) for tokens in product(range(3), repeat=size)
        if (2 in tokens and tokens.index(2) == size - 1) or (2 not in tokens and size == 4)
    ]
    stopped = _outputs(backend, 4)
    full_logs = RAW.score_batch([ScoreRequest((3,), tuple(full))])
    stopped_logs = backend.score_batch([ScoreRequest((3,), tuple(stopped))])

    def score(tokens):
        segments = PARSER.split((3,), tokens, eos_token_id=2)
        if segments.has_complete_thinking:
            return 0.25 * len(segments.thinking_token_ids)
        return -0.1 * sum(tokens)  # one fixed full-sequence fallback reward

    projected = dict.fromkeys(stopped, 0.0)
    for tokens, logs in zip(full, full_logs, strict=True):
        projected[tokens[: backend._stop_end(tokens)]] += exp(sum(logs) + score(tokens))
    weighted = {tokens: exp(sum(logs) + score(tokens)) for tokens, logs in zip(stopped, stopped_logs, strict=True)}
    assert projected == pytest.approx(weighted, abs=1e-12)


def test_empty_thinking_block_continues_to_full_sequence():
    raw = _path(((3,), 1), ((3, 1), 0), ((3, 1, 0), 2))
    # The wrapped backend stops at the empty block's marker; the scope resumes to EOS.
    result = _backend(raw=raw).sample_batch([GenerationRequest((3,), 3, SamplingConfig(), 0, "empty")])[0]
    assert (result.token_ids, result.finish_reason) == ((1, 0, 2), "stop")

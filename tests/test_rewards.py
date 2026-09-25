from math import exp, log

import pytest

from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.backends.transformers_backend import SequenceScoreStatistics
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.rewards.intrinsic import (
    ConsilienceReward,
    SequenceLogProbabilityReward,
)
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.output import thinking_format_from_backend
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser


def test_sequence_log_probability_reward_averages_all_token_scores() -> None:
    backend = TabularAutoregressiveBackend(
        {(): (0.75, 0.25), (1,): (0.4, 0.6)},
        fallback=(0.5, 0.5),
    )
    reward = SequenceLogProbabilityReward(backend, SamplingConfig())

    assert reward((), (1, 0)) == pytest.approx((log(0.25) + log(0.4)) / 2)
    assert reward.batch((), ((0,), (1, 1))) == pytest.approx(
        (log(0.75), (log(0.25) + log(0.6)) / 2)
    )


def test_log_probability_reward_exposes_normalization_parameters() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    sampling = SamplingConfig(temperature=0.8)
    reward = SequenceLogProbabilityReward(backend, sampling)

    assert reward.describe() == {
        "source": "model_sequence_log_probability",
        "model_id": "tabular",
        "policy_id": sampling.policy_id,
        "normalization": "mean_per_token",
    }


def test_log_probability_reward_reweighting_uses_length_dependent_exponent() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.75, 0.25))
    completions = ((0,), (1, 0))
    probabilities = (0.75, 0.25 * 0.75)
    temperature = 0.3
    reward = SequenceLogProbabilityReward(backend)
    unnormalized_reward_target = tuple(
        probability * exp(reward((), completion) / temperature)
        for probability, completion in zip(probabilities, completions, strict=True)
    )
    normalizer = sum(unnormalized_reward_target)
    reward_target = tuple(value / normalizer for value in unnormalized_reward_target)
    power_masses = tuple(
        probability ** (1.0 + 1.0 / (temperature * len(completion)))
        for probability, completion in zip(probabilities, completions, strict=True)
    )
    power_target = tuple(value / sum(power_masses) for value in power_masses)

    assert reward_target == pytest.approx(power_target)


def test_log_probability_reward_is_length_and_batch_order_invariant() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    reward = SequenceLogProbabilityReward(backend)
    completions = ((0,), (0,) * 100, ())
    expected = (log(0.5), log(0.5), 0.0)
    assert reward.batch((), completions) == pytest.approx(expected)
    assert reward.batch((), completions[::-1]) == pytest.approx(expected[::-1])
    assert reward.batch((), ()) == ()


def test_log_probability_reward_counts_genuine_zero_logprobs() -> None:
    assert SequenceLogProbabilityReward.from_token_logprobs((-2.0, 0.0)) == -1.0
    assert SequenceLogProbabilityReward.from_token_logprobs(()) == 0.0


class _ConsilienceBackend:
    model_id = "trajectory-model"

    def __init__(self, trajectories: dict[tuple[int, ...], tuple[float, ...]]) -> None:
        self.trajectories = trajectories
        self.requests = []
        self.confidence_top_k = None

    def score_statistics_batch(self, requests, *, confidence_top_k=None):
        self.requests.extend(requests)
        self.confidence_top_k = confidence_top_k
        return [
            SequenceScoreStatistics(
                token_logprobs=tuple(-1.0 for _ in continuation),
                mean_logprob=-1.0,
                mean_negative_entropy=-1.0,
                mean_self_certainty=1.0,
                token_topk_confidences=self.trajectories[tuple(continuation)],
                confidence_top_k=confidence_top_k,
            )
            for request in requests
            for continuation in request.continuations
        ]


def test_consilience_reward_uses_initial_and_final_confidence_windows() -> None:
    completion = (1, 2, 3, 4, 5)
    backend = _ConsilienceBackend({completion: (10.0, 8.0, 4.0, 2.0, 1.0)})
    reward = ConsilienceReward(
        backend,
        SamplingConfig(),
        top_k=3,
        window_fraction=0.4,
        skip_fraction=0.2,
        initial_penalty=1.0,
        scope="full",
    )

    # Skip the first token, average (8, 4), and compare with final (2, 1).
    assert reward((9,), completion) == pytest.approx(1.5 - 6.0)
    assert backend.confidence_top_k == 3
    assert backend.requests[0].prefix == (9,)


def test_consilience_reward_is_batch_order_invariant_and_pointwise() -> None:
    first = (1, 2, 3, 4)
    second = (5, 6, 7, 8)
    backend = _ConsilienceBackend(
        {
            first: (6.0, 5.0, 2.0, 1.0),
            second: (2.0, 2.0, 3.0, 4.0),
        }
    )
    reward = ConsilienceReward(
        backend,
        window_fraction=0.5,
        skip_fraction=0.0,
        initial_penalty=1.0,
        scope="full",
    )

    forward = reward.batch((), (first, second))
    reverse = reward.batch((), (second, first))

    assert forward == pytest.approx((-4.0, 1.5))
    assert reverse == pytest.approx((forward[1], forward[0]))
    assert len(backend.requests) == 2
    assert all(len(request.continuations) == 2 for request in backend.requests)


def test_consilience_reward_can_isolate_reasoning_before_a_token_marker() -> None:
    full = (1, 2, 90, 91, 3, 4)
    reasoning = (1, 2)
    backend = _ConsilienceBackend({reasoning: (4.0, 1.0)})
    reward = ConsilienceReward(
        backend,
        window_tokens=1,
        skip_fraction=0.0,
        initial_penalty=1.0,
        thinking_format=ThinkingParser((ThinkingFormat((90, 91)),)),
    )

    assert reward((), full) == pytest.approx(-3.0)
    assert backend.requests[0].continuations == (reasoning,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"window_fraction": 0.0}, "window_fraction"),
        ({"window_tokens": 0}, "window_tokens"),
        ({"skip_fraction": 1.0}, "skip_fraction"),
        ({"initial_penalty": -1.0}, "initial_penalty"),
    ],
)
def test_consilience_reward_validates_parameters(kwargs, message) -> None:
    backend = _ConsilienceBackend({})

    with pytest.raises(ValueError, match=message):
        ConsilienceReward(backend, **kwargs)


def test_consilience_thinking_scope_reports_full_fallback_without_a_recognized_format() -> None:
    backend = _ConsilienceBackend({(1, 2): (1.0, 3.0)})
    with pytest.raises(ValueError, match="needs a thinking format"):
        ConsilienceReward(backend)
    reward = ConsilienceReward(backend, thinking_format=thinking_format_from_backend(
        backend, load_settings()["ar"]["output"]))
    assert reward((), (1, 2)) == 0.0
    assert reward.describe_completion((), (1, 2)) == {
        "requested_reward_scope": "thinking", "reward_scope": "full",
        "reward_mode": "consilience_full",
        "reward_fallback_reason": "unrecognized_format",
    }


def test_consilience_missing_empty_and_truncated_thinking_use_pointwise_full_scores() -> None:
    backend = _ConsilienceBackend({
        (1, 2): (1.0, 3.0), (5, 6): (2.0, 5.0),
        (90, 91, 9): (1.0, 2.0, 5.0), (90, 1, 2): (2.0, 3.0, 7.0),
    })
    reward = ConsilienceReward(
        backend, thinking_format=ThinkingParser((ThinkingFormat((91,), (90,)),)),
        window_tokens=1, skip_fraction=0, initial_penalty=1,
    )
    sequences = ((5, 6), (90, 91, 9), (90, 1, 2), (90, 1, 2, 91, 9))
    assert reward.batch((), sequences) == pytest.approx((3, 4, 5, 2))
    assert len(backend.requests) == 2
    assert backend.requests[0].prefix == ()
    assert backend.requests[0].continuations == sequences[:3]
    assert backend.requests[1].prefix == (90,)
    assert backend.requests[1].continuations == ((1, 2),)
    assert [reward.describe_completion((), tokens)["reward_fallback_reason"] for tokens in sequences] == [
        "absent", "empty", "incomplete", None,
    ]
    assert reward.scope_statistics() == {
        "evaluated_sequences": 4, "thinking_sequences": 1, "full_sequences": 3,
        "fallback_reasons": {"absent": 1, "empty": 1, "incomplete": 1},
    }


def test_consilience_ignores_content_but_preserves_opening_token_context() -> None:
    backend = _ConsilienceBackend({(1, 2): (1.0, 3.0)})
    reward = ConsilienceReward(
        backend, thinking_format=ThinkingParser((ThinkingFormat((91, 92), (90,)),)),
        window_tokens=1, skip_fraction=0, initial_penalty=1,
    )
    assert reward((8,), (90, 1, 2, 91, 92, 5)) == 2.0
    assert reward((8,), (90, 1, 2, 91, 92, 6, 7, 8)) == 2.0
    assert reward((8, 90), (1, 2, 91, 92, 6)) == 2.0
    assert all(request.prefix == (8, 90) for request in backend.requests)


def test_consilience_rejects_nonfinite_backend_statistics() -> None:
    backend = _ConsilienceBackend({(1,): (float("nan"),)})
    reward = ConsilienceReward(backend, thinking_format=ThinkingParser((ThinkingFormat((91,)),)))
    with pytest.raises(ValueError, match="finite confidence"):
        reward((), (1, 91))

from math import exp, log

import pytest

from inference_scaling.app.rewards import Reward, memoized
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig, RewardMHConfig
from inference_scaling.arllm.algorithms.mh import run_reward_mh_chain
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import thinking_format_from_backend
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, SequenceLogProbabilityReward
from inference_scaling.arllm.types import TokenStatistic
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream

LOGPROB, TOP = TokenStatistic(SamplingConfig()), TokenStatistic(SamplingConfig(), 3)


def test_sequence_log_probability_reward_averages_all_token_scores() -> None:
    backend = TabularAutoregressiveBackend({(): (0.75, 0.25), (1,): (0.4, 0.6)}, fallback=(0.5, 0.5))
    reward = SequenceLogProbabilityReward(backend, LOGPROB)
    assert reward((), (1, 0)) == pytest.approx((log(0.25) + log(0.4)) / 2)
    assert reward.batch((), ((0,), (1, 1))) == pytest.approx((log(0.75), (log(0.25) + log(0.6)) / 2))
    # Length and batch order leave a reward unchanged; an empty completion has reward zero.
    completions = ((1, 0), (1,) + (0,) * 100, ())
    expected = ((log(0.25) + log(0.4)) / 2, (log(0.25) + log(0.4) + 99 * log(0.5)) / 101, 0.0)
    assert reward.batch((), completions) == pytest.approx(expected)
    assert reward.batch((), completions[::-1]) == pytest.approx(expected[::-1])
    assert reward.batch((), ()) == () and reward.reduce((-2.0, 0.0)) == -1.0


def test_log_probability_reward_describes_its_policy_and_reads_log_probabilities() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    sampling = SamplingConfig(temperature=0.8)
    assert SequenceLogProbabilityReward(backend, TokenStatistic(sampling)).describe() == {
        "source": "model_sequence_log_probability", "model_id": "tabular", "policy_id": sampling.policy_id,
        "normalization": "mean_per_token",
    }
    with pytest.raises(ValueError, match="log-probabilities"):
        SequenceLogProbabilityReward(backend, TOP)


def test_log_probability_reward_reweighting_uses_length_dependent_exponent() -> None:
    reward = SequenceLogProbabilityReward(TabularAutoregressiveBackend({}, fallback=(0.75, 0.25)), LOGPROB)
    completions, probabilities, temperature = ((0,), (1, 0)), (0.75, 0.25 * 0.75), 0.3
    reweighted = [probability * exp(reward((), completion) / temperature)
                  for probability, completion in zip(probabilities, completions, strict=True)]
    powers = [probability ** (1.0 + 1.0 / (temperature * len(completion)))
              for probability, completion in zip(probabilities, completions, strict=True)]
    assert [value / sum(reweighted) for value in reweighted] == pytest.approx([value / sum(powers) for value in powers])


def test_problem_reward_scores_each_sequence_once() -> None:
    batches = []

    def batch(_prompt, sequences):
        batches.append(list(sequences))
        return [float(sum(tokens)) for tokens in sequences]

    reward = Reward(1.0, memoized(batch), 1, {})
    assert reward.batch((), [(1, 2), (3,), (1, 2)]) == [3.0, 3.0, 3.0]
    assert reward.batch((), [(3,), (4,)]) == [3.0, 4.0]
    assert batches == [[(1, 2), (3,)], [(4,)]]


class _Confidences:
    """A fixed confidence for each token, whatever its context."""

    model_id = "trajectory-model"

    def __init__(self, values: dict[int, float]) -> None:
        self.values, self.requests = values, []

    def token_statistics(self, requests, statistic):
        assert statistic == TOP
        self.requests.extend(requests)
        return [tuple(self.values[token] for token in continuation)
                for request in requests for continuation in request.continuations]


def test_consilience_reward_uses_initial_and_final_confidence_windows() -> None:
    backend = _Confidences({1: 10.0, 2: 8.0, 3: 4.0, 4: 2.0, 5: 1.0})
    reward = ConsilienceReward(backend, TOP, window_fraction=0.4, skip_fraction=0.2, initial_penalty=1.0, scope="full")
    # Skip the first token, average (8, 4), and compare with final (2, 1).
    assert reward((9,), (1, 2, 3, 4, 5)) == pytest.approx(1.5 - 6.0)
    assert backend.requests[0].prefix == (9,)


def test_consilience_reward_is_batch_order_invariant_and_pointwise() -> None:
    backend = _Confidences({1: 6.0, 2: 5.0, 3: 2.0, 4: 1.0, 5: 2.0, 6: 2.0, 7: 3.0, 8: 4.0})
    reward = ConsilienceReward(backend, TOP, window_fraction=0.5, skip_fraction=0.0, initial_penalty=1.0, scope="full")
    forward, reverse = reward.batch((), ((1, 2, 3, 4), (5, 6, 7, 8))), reward.batch((), ((5, 6, 7, 8), (1, 2, 3, 4)))
    assert forward == pytest.approx((-4.0, 1.5)) and reverse == pytest.approx(forward[::-1])
    assert [len(request.continuations) for request in backend.requests] == [2, 2]


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"window_fraction": 0.0}, "window_fraction"), ({"window_tokens": 0}, "window_tokens"),
    ({"skip_fraction": 1.0}, "skip_fraction"), ({"initial_penalty": -1.0}, "initial_penalty"),
    ({"statistic": LOGPROB}, "top-K confidence"),
])
def test_consilience_reward_validates_parameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ConsilienceReward(**{"backend": _Confidences({}), "statistic": TOP, "scope": "full", **kwargs})


def test_consilience_thinking_scope_falls_back_to_the_full_sequence_without_a_recognized_format() -> None:
    backend = _Confidences({1: 1.0, 2: 3.0})
    with pytest.raises(ValueError, match="needs a thinking format"):
        ConsilienceReward(backend, TOP)
    reward = ConsilienceReward(backend, TOP, thinking_format=thinking_format_from_backend(
        backend, load_settings()["ar"]["output"]))
    assert reward((), (1, 2)) == 0.0 and reward.fallback((), (1, 2)) == "unrecognized_format"


def test_consilience_scores_a_complete_thinking_segment_or_the_whole_completion() -> None:
    backend = _Confidences({5: 2.0, 6: 5.0, 90: 1.0, 91: 2.0, 9: 5.0, 1: 3.0, 2: 7.0})
    reward = ConsilienceReward(backend, TOP, thinking_format=ThinkingParser((ThinkingFormat((91,), (90,)),)),
                               window_tokens=1, skip_fraction=0, initial_penalty=1)
    sequences = ((5, 6), (90, 91, 9), (90, 1, 2), (90, 1, 2, 91, 9))
    assert reward.batch((), sequences) == pytest.approx((3, 4, 6, 4))
    # Each span is read with the context before it, and nothing after it.
    assert backend.requests[0].continuations == sequences[:3] + ((90, 1, 2),)
    assert [reward.fallback((), tokens) for tokens in sequences] == ["absent", "empty", "incomplete", None]


def test_consilience_ignores_content_but_keeps_the_opening_token_context() -> None:
    backend = _Confidences({90: 9.0, 1: 1.0, 2: 3.0})
    reward = ConsilienceReward(backend, TOP, thinking_format=ThinkingParser((ThinkingFormat((91, 92), (90,)),)),
                               window_tokens=1, skip_fraction=0, initial_penalty=1)
    assert reward((8,), (90, 1, 2, 91, 92, 5)) == reward((8,), (90, 1, 2, 91, 92, 6, 7)) == 2.0
    assert reward((8, 90), (1, 2, 91, 92, 6)) == 2.0
    assert [(request.prefix, request.continuations) for request in backend.requests] == [
        ((8,), ((90, 1, 2),)), ((8,), ((90, 1, 2),)), ((8, 90), ((1, 2),))]


def test_consilience_rejects_nonfinite_backend_statistics() -> None:
    reward = ConsilienceReward(_Confidences({1: float("nan")}), TOP, thinking_format=ThinkingParser((ThinkingFormat((91,)),)))
    with pytest.raises(ValueError, match="finite confidence"):
        reward((), (1, 91))


class _Counting(TabularAutoregressiveBackend):
    scored = 0

    def token_statistics(self, requests, statistic):
        self.scored += sum(len(continuation) for request in requests for continuation in request.continuations)
        return super().token_statistics(requests, statistic)


@pytest.mark.parametrize("statistic", [TokenStatistic(SamplingConfig(temperature=0.7)), TokenStatistic(SamplingConfig(), 2)])
def test_model_rewards_read_the_statistics_that_generation_recorded(statistic) -> None:
    table = {(): (0.5, 0.3, 0.2), (0,): (0.2, 0.5, 0.3), (1,): (0.6, 0.2, 0.2)}

    def run(backend):
        reward = (SequenceLogProbabilityReward(backend, statistic) if statistic.top_k is None else
                  ConsilienceReward(backend, statistic, window_tokens=1, scope="full")).batch
        conditional = run_conditional_is(backend, (), ConditionalISConfig(
            candidate_count=3, rollout_count=2, block_size=1, total_length=4, reward_temperature=0.5, block_first=False),
            reward, SeedStream(3))
        chain = run_reward_mh_chain(backend, (), RewardMHConfig(
            total_length=4, block_size=2, steps_per_block=4, reward_temperature=0.5, suffix_schedule="uniform",
            iterations=None, suffix_replay=True), SamplingConfig(), reward, SeedStream(3))
        return conditional.steps, chain.trace, chain.token_ids

    plain, recorded = _Counting(table, fallback=(0.4, 0.3, 0.3)), _Counting(table, fallback=(0.4, 0.3, 0.3))
    runs = run(plain)
    assert run(StatisticRecorder(recorded, statistic)) == runs and any(step.replayed_tokens for step in runs[1])
    # Every scored token was generated after the same context, a replayed MH suffix too, so nothing is scored.
    assert plain.scored > 0 and recorded.scored == 0

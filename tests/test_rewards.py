from math import log

import pytest

from inference_scaling.app.rewards import Reward, memoized
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig, MHConfig
from inference_scaling.arllm.algorithms.mh import run_mh_chain
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import thinking_format_from_backend
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, SelfCertaintyReward
from inference_scaling.arllm.types import ScoreRequest, TokenStatistic
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream

TOP, FULL = TokenStatistic(SamplingConfig(), 3), TokenStatistic(SamplingConfig())


def test_token_statistics_read_the_distribution_not_the_sampled_token() -> None:
    backend, request = TabularAutoregressiveBackend({(): (0.7, 0.2, 0.1)}, fallback=(0.4, 0.3, 0.3)), ScoreRequest((), ((0,), (2,)))
    # The top-2 confidence and the divergence from uniform are the same whichever token was sampled.
    for statistic, value in ((TokenStatistic(SamplingConfig(), 2), -(log(0.7) + log(0.2)) / 2),
                             (FULL, -(log(0.7) + log(0.2) + log(0.1)) / 3 - log(3))):
        assert backend.token_statistics([request], statistic) == [pytest.approx((value,))] * 2


def test_self_certainty_is_the_mean_divergence_whatever_the_length() -> None:
    reward = SelfCertaintyReward(_Confidences({1: 9.0, 2: 3.0, 3: 5.0}), FULL, scope="full")
    # Repeating a trajectory keeps its mean, so a longer thought earns nothing for its length; an empty one scores zero.
    assert reward.batch((), ((1, 2), (1, 2, 1, 2), (3,), ())) == pytest.approx((6.0, 6.0, 5.0, 0.0))
    assert reward.describe() == {"source": "model_self_certainty", "model_id": "trajectory-model",
                                 "policy_id": FULL.policy.policy_id, "top_k": None, "scope": "full",
                                 "fallback": "full_sequence", "thinking_format": None}
    with pytest.raises(ValueError, match="finite"):
        SelfCertaintyReward(_Confidences({1: float("nan")}), FULL, scope="full")((), (1,))
    with pytest.raises(ValueError, match="whole vocabulary"):
        SelfCertaintyReward(_Confidences({}), TOP, scope="full")


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
    """A fixed statistic for each token, whatever its context."""

    model_id = "trajectory-model"

    def __init__(self, values: dict[int, float]) -> None:
        self.values, self.requests = values, []

    def token_statistics(self, requests, statistic):
        assert statistic in (TOP, FULL)
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
    ({"scope": "answer"}, "thinking or full"), ({"statistic": FULL}, "top-K confidence"),
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


@pytest.mark.parametrize(("reward_class", "statistic"), [(SelfCertaintyReward, TokenStatistic(SamplingConfig(temperature=0.7))),
                                                         (ConsilienceReward, TokenStatistic(SamplingConfig(), 2))])
def test_model_rewards_read_the_statistics_that_generation_recorded(reward_class, statistic) -> None:
    table = {(): (0.5, 0.3, 0.2), (0,): (0.2, 0.5, 0.3), (1,): (0.6, 0.2, 0.2)}

    def run(backend):
        reward = reward_class(backend, statistic, scope="full").batch
        conditional = run_conditional_is(backend, (), ConditionalISConfig(
            candidate_count=3, rollout_count=2, block_size=1, total_length=4, reward_temperature=0.5, block_first=False),
            reward, SeedStream(3))
        chain = run_mh_chain(backend, (), MHConfig(
            alpha=1.0, reward_temperature=0.5, total_length=4, block_size=2, steps_per_block=4, suffix_schedule="uniform",
            iterations=None, suffix_replay=True, early_rejection=False), SeedStream(3), base=SamplingConfig(),
            proposal=SamplingConfig(), reward=reward)
        return conditional.steps, chain.trace, chain.token_ids

    plain, recorded = _Counting(table, fallback=(0.4, 0.3, 0.3)), _Counting(table, fallback=(0.4, 0.3, 0.3))
    runs = run(plain)
    assert run(StatisticRecorder(recorded, statistic)) == runs and any(step.replayed_tokens for step in runs[1])
    # Every scored token was generated after the same context, a replayed MH suffix too, so nothing is scored.
    assert plain.scored > 0 and recorded.scored == 0

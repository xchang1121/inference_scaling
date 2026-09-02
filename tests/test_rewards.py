from math import exp, log

import pytest

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.rewards import SequenceLogProbabilityReward


def test_sequence_log_probability_reward_sums_all_token_scores() -> None:
    backend = TabularAutoregressiveBackend(
        {(): (0.75, 0.25), (1,): (0.4, 0.6)},
        fallback=(0.5, 0.5),
    )
    reward = SequenceLogProbabilityReward(backend, SamplingConfig(), scale=2.0)

    assert reward((), (1, 0)) == pytest.approx(2.0 * (log(0.25) + log(0.4)))
    assert reward.batch((), ((0,), (1, 1))) == pytest.approx(
        (2.0 * log(0.75), 2.0 * (log(0.25) + log(0.6)))
    )


def test_log_probability_reward_exposes_power_target_parameters() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    sampling = SamplingConfig(temperature=0.8)
    reward = SequenceLogProbabilityReward(backend, sampling, scale=0.6)

    assert reward.describe() == {
        "source": "model_sequence_log_probability",
        "model_id": "tabular",
        "policy_id": sampling.policy_id,
        "scale": 0.6,
    }


def test_log_probability_reward_reweighting_equals_power_distribution() -> None:
    probabilities = (0.75, 0.25)
    scale = 0.6
    temperature = 0.3
    unnormalized_reward_target = tuple(
        probability * exp(scale * log(probability) / temperature)
        for probability in probabilities
    )
    normalizer = sum(unnormalized_reward_target)
    reward_target = tuple(value / normalizer for value in unnormalized_reward_target)
    exponent = 1.0 + scale / temperature
    power_normalizer = sum(value**exponent for value in probabilities)
    power_target = tuple(
        value**exponent / power_normalizer for value in probabilities
    )

    assert reward_target == pytest.approx(power_target)

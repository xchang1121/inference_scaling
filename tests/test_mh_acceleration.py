from itertools import product
from math import exp, log, prod

from inference_scaling.arllm.algorithms.mh_acceleration import (
    FrozenReplaySuffixProposal,
    run_reward_mh_chain_replay_proposal,
)
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.algorithms.config import RewardMHConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.metrics import empirical_distribution, total_variation
from inference_scaling.shared.rng import SeedStream


class GenerationOnlyBackend(TabularAutoregressiveBackend):
    def score_batch(self, requests):
        raise AssertionError("replay reuses the history's generation log-probabilities")


def _reward_target(probabilities, *, length, temperature, reward):
    weights = {
        sequence: prod(probabilities[token] for token in sequence)
        * exp(reward((), sequence) / temperature)
        for sequence in product(range(len(probabilities)), repeat=length)
    }
    normalizer = sum(weights.values())
    return {sequence: weight / normalizer for sequence, weight in weights.items()}


def test_replay_mixture_at_zero_reward_and_zero_history_weight_accepts_all() -> None:
    proposal = FrozenReplaySuffixProposal(
        TabularAutoregressiveBackend({}, fallback=(0.8, 0.2)), (), [], history_mixture=0.0, sampling=SamplingConfig(),
    )
    config = RewardMHConfig(
        total_length=4,
        block_size=2,
        steps_per_block=8,
        reward_temperature=1.0, suffix_schedule="uniform", iterations=None,
    )
    result = run_reward_mh_chain_replay_proposal(proposal, config, lambda _prompt, _sequence: 0.0, SeedStream(4))
    assert result.accepted == result.attempts


def test_frozen_replay_proposal_approaches_the_exact_reward_target() -> None:
    probabilities = (0.65, 0.35)
    temperature = 0.7
    history = [(sequence, tuple(log(probabilities[token]) for token in sequence))
               for sequence in ((1, 1),) * 40 + ((1, 0),) * 10]
    proposal = FrozenReplaySuffixProposal(GenerationOnlyBackend({}, fallback=probabilities), (), history,
                                          history_mixture=0.65, sampling=SamplingConfig())

    def reward(_, sequence):
        return float(sequence == (1, 1))

    config = RewardMHConfig(total_length=2, block_size=1, steps_per_block=20, reward_temperature=temperature, suffix_schedule="uniform", iterations=None)
    outputs = [
        run_reward_mh_chain_replay_proposal(proposal, config, reward, SeedStream(117), chain_id=chain)
        for chain in range(2500)
    ]
    empirical = empirical_distribution(result.token_ids for result in outputs)
    target = _reward_target(
        probabilities,
        length=2,
        temperature=temperature,
        reward=reward,
    )
    assert total_variation(empirical, target) < 0.04
    assert {step.proposal_source for result in outputs for step in result.trace} == {"base", "history"}

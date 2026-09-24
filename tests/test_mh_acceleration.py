from itertools import product
from math import exp, prod

import pytest

from inference_scaling.arllm.algorithms.mh_acceleration import (
    FrozenReplaySuffixProposal,
    run_reward_mh_chain_replay_proposal,
)
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.algorithms.config import RewardMHConfig
from inference_scaling.shared.metrics import empirical_distribution, total_variation
from inference_scaling.shared.rng import SeedStream


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
        TabularAutoregressiveBackend({}, fallback=(0.8, 0.2)),
        history_mixture=0.0,
    )
    config = RewardMHConfig(
        total_length=4,
        block_size=2,
        steps_per_block=8,
        reward_temperature=1.0,
    )
    result = run_reward_mh_chain_replay_proposal(
        proposal,
        (),
        config,
        lambda _prompt, _sequence: 0.0,
        SeedStream(4),
    )
    assert result.accepted == result.attempts
    assert all(step.log_acceptance == pytest.approx(0.0) for step in result.trace)


def test_frozen_replay_proposal_approaches_the_exact_reward_target() -> None:
    probabilities = (0.65, 0.35)
    temperature = 0.7
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    proposal = FrozenReplaySuffixProposal(backend, history_mixture=0.65)
    proposal.observe_sequences((), ((1, 1),) * 40 + ((1, 0),) * 10)

    def reward(_, sequence):
        return float(sequence == (1, 1))

    config = RewardMHConfig(total_length=2, block_size=1, steps_per_block=20, reward_temperature=temperature)
    outputs = [
        run_reward_mh_chain_replay_proposal(proposal, (), config, reward, SeedStream(117), chain_id=chain)
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
    snapshot = proposal.snapshot()
    assert snapshot.base_draws > 0
    assert snapshot.history_draws > 0
    assert snapshot.logprob_queries > 0


def test_replay_history_is_frozen_before_the_chain_starts() -> None:
    proposal = FrozenReplaySuffixProposal(
        TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)),
        history_mixture=0.5,
    )
    proposal.observe_sequence((), (1, 1))
    proposal.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        proposal.observe_sequence((), (0, 0))

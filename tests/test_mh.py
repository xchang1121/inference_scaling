from dataclasses import replace
from itertools import product
from math import exp, prod

import pytest

from inference_scaling.arllm.algorithms.mh import (
    run_power_mh_chain,
    run_reward_mh_chain,
    suffix_length_probabilities,
)
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.algorithms.config import PowerMHConfig, RewardMHConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.metrics import empirical_distribution, total_variation
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import pointwise
from inference_scaling.arllm.types import SequenceSample


def _power_target(probabilities: tuple[float, ...], length: int, alpha: float):
    weights = {
        sequence: float(prod(probabilities[token] for token in sequence) ** alpha)
        for sequence in product(range(len(probabilities)), repeat=length)
    }
    normalizer = sum(weights.values())
    return {sequence: weight / normalizer for sequence, weight in weights.items()}


def _complete_target(probabilities, eos, length, weight):
    """Normalized weights over outputs that end at their first EOS or at the length limit."""
    outputs = [
        tokens for size in range(1, length + 1) for tokens in product(range(len(probabilities)), repeat=size)
        if (eos in tokens and tokens.index(eos) == size - 1) or (eos not in tokens and size == length)
    ]
    weights = {tokens: weight(tokens, prod(probabilities[token] for token in tokens)) for tokens in outputs}
    normalizer = sum(weights.values())
    return {tokens: value / normalizer for tokens, value in weights.items()}


def test_explicit_iterations_preserve_full_length_kernel_and_seed_stream():
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    sampling = SamplingConfig(temperature=0.8)
    explicit = PowerMHConfig(suffix_replay=False, alpha=2, total_length=7, block_size=2, steps_per_block=10, iterations=3, suffix_schedule="uniform")
    full_stage = PowerMHConfig(suffix_replay=False, alpha=2, total_length=7, block_size=7, steps_per_block=3, suffix_schedule="uniform", iterations=None)
    left = run_power_mh_chain(backend, (), explicit, sampling, SeedStream(2))
    right = run_power_mh_chain(backend, (), full_stage, sampling, SeedStream(2))
    assert left == right
    assert left.attempts == 3
    assert {step.stage_length for step in left.trace} == {7}
    reward = pointwise(lambda prompt, completion: float(sum(completion)))
    left = run_reward_mh_chain(backend, (), RewardMHConfig(suffix_replay=False, total_length=7, block_size=2, iterations=3, steps_per_block=10, reward_temperature=0.1, suffix_schedule="uniform"), sampling, reward, SeedStream(2))
    right = run_reward_mh_chain(backend, (), RewardMHConfig(suffix_replay=False, total_length=7, block_size=7, steps_per_block=3, reward_temperature=0.1, suffix_schedule="uniform", iterations=None), sampling, reward, SeedStream(2))
    assert left == right
    assert left.attempts == 3


def test_mh_returns_fixed_length_and_all_suffix_starts_are_reachable() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    result = run_power_mh_chain(
        backend,
        (),
        PowerMHConfig(suffix_replay=False, alpha=2, total_length=5, block_size=2, steps_per_block=40, suffix_schedule="uniform", iterations=None),
        SamplingConfig(temperature=0.8),
        SeedStream(11),
    )
    assert len(result.token_ids) == 5
    assert {step.stage_length for step in result.trace} == {2, 4, 5}
    final_cuts = {step.cut for step in result.trace if step.stage_length == 5}
    assert final_cuts == set(range(5))
    assert 0 <= result.acceptance_rate <= 1


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_suffix_length_schedules_have_normalized_full_support(schedule: str) -> None:
    probabilities = suffix_length_probabilities(16, schedule)
    assert len(probabilities) == 16
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in probabilities)


def test_nonuniform_schedules_reduce_the_expected_proposed_suffix_length() -> None:
    lengths = range(1, 17)

    def expectation(schedule: str) -> float:
        return sum(
            length * probability
            for length, probability in zip(
                lengths, suffix_length_probabilities(16, schedule), strict=True
            )
        )

    uniform = expectation("uniform")
    assert expectation("inverse_length") < uniform
    assert expectation("multiscale") < uniform


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_mh_empirical_output_approaches_enumerated_power_target(
    schedule: str,
) -> None:
    probabilities = (0.65, 0.35)
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    config = PowerMHConfig(
        suffix_replay=False, alpha=2,
        total_length=2,
        block_size=2,
        steps_per_block=20,
        suffix_schedule=schedule, iterations=None,
    )
    outputs = [
        run_power_mh_chain(backend, (), config, SamplingConfig(temperature=0.7), SeedStream(2026), chain_id=chain)
        for chain in range(2500)
    ]
    empirical = empirical_distribution(result.token_ids for result in outputs)
    target = _power_target(probabilities, length=2, alpha=2)
    assert total_variation(empirical, target) < 0.045


def test_base_proposal_at_alpha_one_accepts_every_move() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    result = run_power_mh_chain(
        backend,
        (),
        PowerMHConfig(suffix_replay=False, alpha=1, total_length=4, block_size=4, steps_per_block=20, suffix_schedule="uniform", iterations=None),
        SamplingConfig(),
        SeedStream(3),
    )
    assert result.accepted == result.attempts
    assert all(step.log_acceptance == pytest.approx(0.0) for step in result.trace)


def test_mh_reuses_reference_scores_emitted_during_proposal_generation() -> None:
    class DualScoreBackend:
        model_id = "dual-score"

        def __init__(self) -> None:
            self.score_calls = 0

        def sample_batch(self, requests):
            return [
                SequenceSample(
                    prefix=request.prefix,
                    token_ids=(0,) * request.max_new_tokens,
                    token_logprobs=(-0.2,) * request.max_new_tokens,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    reference_token_logprobs=(-0.4,) * request.max_new_tokens,
                    reference_policy_id=SamplingConfig().policy_id,
                )
                for request in requests
            ]

        def score_batch(self, requests):
            self.score_calls += 1
            raise AssertionError("cached reference scores should avoid rescoring")

    backend = DualScoreBackend()
    result = run_power_mh_chain(
        backend,
        (),
        PowerMHConfig(suffix_replay=False, alpha=2, total_length=4, block_size=2, steps_per_block=3, suffix_schedule="uniform", iterations=None),
        SamplingConfig(temperature=0.5),
        SeedStream(7),
    )
    assert len(result.token_ids) == 4
    assert backend.score_calls == 0


def test_reward_mh_approaches_enumerated_base_times_weight_target() -> None:
    probabilities = (0.7, 0.3)
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    temperature = 0.8

    def reward(_, sequence):
        return float(sequence == (1, 1))

    config = RewardMHConfig(suffix_replay=False, total_length=2, block_size=1, steps_per_block=25, reward_temperature=temperature, suffix_schedule="uniform", iterations=None)
    outputs = [
        run_reward_mh_chain(backend, (), config, SamplingConfig(temperature=0.7), pointwise(reward), SeedStream(91),
                            chain_id=chain)
        for chain in range(3000)
    ]
    weights = {
        sequence: float(prod(probabilities[token] for token in sequence) * exp(reward((), sequence) / temperature))
        for sequence in product(range(2), repeat=2)
    }
    normalizer = sum(weights.values())
    target = {sequence: weight / normalizer for sequence, weight in weights.items()}
    empirical = empirical_distribution(result.token_ids for result in outputs)
    assert total_variation(empirical, target) < 0.04


@pytest.mark.parametrize("chain", ["power", "reward"])
def test_variable_length_chains_target_complete_outputs(chain) -> None:
    probabilities = (0.6, 0.4)  # token 1 is EOS
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    proposal = SamplingConfig(temperature=0.7, eos_token_id=1)
    if chain == "power":
        config = PowerMHConfig(suffix_replay=False, alpha=2, total_length=3, block_size=3, steps_per_block=12, suffix_schedule="uniform", iterations=None)
        results = [run_power_mh_chain(backend, (), config, proposal, SeedStream(5), chain_id=index) for index in range(2000)]
        target = _complete_target(probabilities, 1, 3, lambda tokens, probability: probability**2)
    else:
        settings = RewardMHConfig(suffix_replay=False, total_length=3, block_size=3, steps_per_block=12, reward_temperature=0.8, suffix_schedule="uniform", iterations=None)
        results = [
            run_reward_mh_chain(backend, (), settings, proposal, pointwise(lambda _, tokens: float(len(tokens))), SeedStream(5),
                                chain_id=index)
            for index in range(2000)
        ]
        target = _complete_target(probabilities, 1, 3, lambda tokens, probability: probability * exp(len(tokens) / 0.8))
    assert total_variation(empirical_distribution(result.token_ids for result in results), target) < 0.03
    # A cut past the end of a stopped output is skipped without a proposal.
    assert all(result.attempts + result.skipped == 12 for result in results)
    assert sum(result.skipped for result in results) > 0


@pytest.mark.parametrize("sampling", [SamplingConfig(top_k=1), SamplingConfig(top_p=0.9)])
def test_mh_rejects_truncated_proposals(sampling) -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    with pytest.raises(ValueError):
        run_power_mh_chain(
            backend,
            (),
            PowerMHConfig(suffix_replay=False, total_length=2, block_size=2, steps_per_block=1, alpha=4.0, suffix_schedule="uniform", iterations=None),
            sampling,
            SeedStream(0),
        )


def test_suffix_replay_leaves_both_chains_unchanged_and_replays_repeated_tokens() -> None:
    backend = TabularAutoregressiveBackend({(): (0.5, 0.3, 0.2), (0,): (0.7, 0.2, 0.1)}, fallback=(0.45, 0.45, 0.1))
    power = [run_power_mh_chain(backend, (), PowerMHConfig(
        suffix_replay=replay, alpha=3.0, total_length=5, block_size=2, steps_per_block=4, suffix_schedule="uniform",
        iterations=None), SamplingConfig(temperature=1 / 3, eos_token_id=2), SeedStream(5)) for replay in (False, True)]
    reward = pointwise(lambda _prompt, sequence: float(sum(sequence)))
    rewarded = [run_reward_mh_chain(backend, (), RewardMHConfig(
        suffix_replay=replay, total_length=5, block_size=2, steps_per_block=4, reward_temperature=0.5,
        suffix_schedule="uniform", iterations=None), SamplingConfig(eos_token_id=2), reward, SeedStream(6))
        for replay in (False, True)]
    for plain, replayed in (power, rewarded):
        assert (plain.token_ids, plain.base_token_logprobs) == (replayed.token_ids, replayed.base_token_logprobs)
        assert [replace(step, replayed_tokens=0) for step in replayed.trace] == list(plain.trace)
        assert replayed.replayed_tokens > 0 == plain.replayed_tokens

from dataclasses import replace
from itertools import product
from math import exp, log, prod

import pytest

from inference_scaling.arllm.algorithms.config import MHConfig
from inference_scaling.arllm.algorithms.mh import run_mh_chain, suffix_length_probabilities
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample
from inference_scaling.shared.metrics import empirical_distribution, total_variation
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import pointwise

BASE = SamplingConfig()


def _config(**overrides) -> MHConfig:
    """A power chain of alpha 2 without replay or early rejection, changed by ``overrides``."""

    return MHConfig(**{"alpha": 2.0, "reward_temperature": None, "total_length": 2, "block_size": 2,
                       "steps_per_block": 1, "suffix_schedule": "uniform", "iterations": None, "suffix_replay": False,
                       "early_rejection": False, **overrides})


def _chain(backend, config, proposal, seed, *, base=BASE, prompt=(), **options):
    return run_mh_chain(backend, prompt, config, SeedStream(seed), base=base, proposal=proposal, **options)


def _target(probabilities: tuple[float, ...], length: int, weight):
    weights = {sequence: weight(sequence, prod(probabilities[token] for token in sequence))
               for sequence in product(range(len(probabilities)), repeat=length)}
    return {sequence: value / sum(weights.values()) for sequence, value in weights.items()}


def _complete_target(probabilities, eos, length, weight):
    """Normalized weights over outputs that end at their first EOS or at the length limit."""
    outputs = [
        tokens for size in range(1, length + 1) for tokens in product(range(len(probabilities)), repeat=size)
        if (eos in tokens and tokens.index(eos) == size - 1) or (eos not in tokens and size == length)
    ]
    weights = {tokens: weight(tokens, prod(probabilities[token] for token in tokens)) for tokens in outputs}
    normalizer = sum(weights.values())
    return {tokens: value / normalizer for tokens, value in weights.items()}


def test_explicit_iterations_preserve_the_full_length_kernel_and_seed_stream():
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    sampling = SamplingConfig(temperature=0.8)
    for options in ({}, {"alpha": 1.0, "reward_temperature": 0.1, "reward": pointwise(lambda _, tokens: sum(tokens))}):
        reward = {key: options.pop(key) for key in ("reward",) if key in options}
        left = _chain(backend, _config(total_length=7, block_size=2, steps_per_block=10, iterations=3, **options),
                      sampling, 2, **reward)
        right = _chain(backend, _config(total_length=7, block_size=7, steps_per_block=3, **options), sampling, 2, **reward)
        assert left == right and left.attempts == 3 and {step.stage_length for step in left.trace} == {7}


def test_a_power_chain_grows_in_stages_and_reaches_every_suffix_start() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
    result = _chain(backend, _config(total_length=5, steps_per_block=40), SamplingConfig(temperature=0.8), 11)
    assert len(result.token_ids) == 5
    assert {step.stage_length for step in result.trace} == {2, 4, 5}
    assert {step.cut for step in result.trace if step.stage_length == 5} == set(range(5))
    assert 0 <= result.acceptance_rate <= 1


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_suffix_length_schedules_have_normalized_full_support(schedule: str) -> None:
    probabilities = suffix_length_probabilities(16, schedule)
    assert len(probabilities) == 16
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in probabilities)


def test_nonuniform_schedules_reduce_the_expected_proposed_suffix_length() -> None:
    def expectation(schedule: str) -> float:
        return sum(length * probability for length, probability in zip(
            range(1, 17), suffix_length_probabilities(16, schedule), strict=True))

    assert expectation("inverse_length") < expectation("uniform") and expectation("multiscale") < expectation("uniform")


@pytest.mark.parametrize("schedule", ["uniform", "inverse_length", "multiscale"])
def test_the_power_chain_approaches_the_enumerated_power_target(schedule: str) -> None:
    probabilities = (0.65, 0.35)
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    config = _config(steps_per_block=20, suffix_schedule=schedule)
    outputs = [_chain(backend, config, SamplingConfig(temperature=0.7), 2026, chain_id=chain) for chain in range(2500)]
    target = _target(probabilities, 2, lambda _, probability: probability**2)
    assert total_variation(empirical_distribution(result.token_ids for result in outputs), target) < 0.045


def test_the_base_policy_at_another_temperature_is_the_target_base() -> None:
    probabilities, temperature = (0.65, 0.35), 0.6
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    base = SamplingConfig(temperature=temperature)
    outputs = [_chain(backend, _config(steps_per_block=20), SamplingConfig(temperature=0.7 * temperature), 7, base=base,
                      chain_id=chain) for chain in range(2500)]
    tempered = tuple(value ** (1 / temperature) / sum(other ** (1 / temperature) for other in probabilities)
                     for value in probabilities)
    target = _target(tempered, 2, lambda _, probability: probability**2)
    assert total_variation(empirical_distribution(result.token_ids for result in outputs), target) < 0.045


def test_the_base_proposal_at_alpha_one_accepts_every_move() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2])
    result = _chain(backend, _config(alpha=1, total_length=4, block_size=4, steps_per_block=20), BASE, 3)
    assert result.accepted == result.attempts
    assert all(step.log_acceptance == pytest.approx(0.0) for step in result.trace)


def test_the_chain_reuses_reference_scores_emitted_during_proposal_generation() -> None:
    class DualScoreBackend:
        model_id = "dual-score"

        def sample_batch(self, requests):
            return [SequenceSample(
                prefix=request.prefix, token_ids=(0,) * request.max_new_tokens,
                token_logprobs=(-0.2,) * request.max_new_tokens, policy_id=request.sampling.policy_id,
                model_id=self.model_id, request_id=request.request_id,
                reference_token_logprobs=(-0.4,) * request.max_new_tokens, reference_policy_id=BASE.policy_id,
            ) for request in requests]

        def score_batch(self, requests):
            raise AssertionError("cached reference scores should avoid rescoring")

    result = _chain(DualScoreBackend(), _config(total_length=4, steps_per_block=3), SamplingConfig(temperature=0.5), 7)
    assert len(result.token_ids) == 4


def test_the_reward_chain_approaches_the_enumerated_base_times_weight_target() -> None:
    probabilities, temperature = (0.7, 0.3), 0.8

    def reward(_, sequence):
        return float(sequence == (1, 1))

    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    config = _config(alpha=1, reward_temperature=temperature, block_size=1, steps_per_block=25)
    outputs = [_chain(backend, config, SamplingConfig(temperature=0.7), 91, reward=pointwise(reward), chain_id=chain)
               for chain in range(3000)]
    target = _target(probabilities, 2, lambda sequence, probability: probability * exp(reward((), sequence) / temperature))
    assert total_variation(empirical_distribution(result.token_ids for result in outputs), target) < 0.04


@pytest.mark.parametrize("chain", ["power", "reward"])
def test_variable_length_chains_target_complete_outputs(chain) -> None:
    probabilities = (0.6, 0.4)  # token 1 is EOS
    backend = TabularAutoregressiveBackend({}, fallback=probabilities)
    proposal, base = SamplingConfig(temperature=0.7, eos_token_id=1), SamplingConfig(eos_token_id=1)
    if chain == "power":
        results = [_chain(backend, _config(total_length=3, block_size=3, steps_per_block=12), proposal, 5, base=base,
                          chain_id=index) for index in range(2000)]
        target = _complete_target(probabilities, 1, 3, lambda tokens, probability: probability**2)
    else:
        config = _config(alpha=1, reward_temperature=0.8, total_length=3, block_size=3, steps_per_block=12)
        results = [_chain(backend, config, proposal, 5, base=base, reward=pointwise(lambda _, tokens: float(len(tokens))),
                          chain_id=index) for index in range(2000)]
        target = _complete_target(probabilities, 1, 3, lambda tokens, probability: probability * exp(len(tokens) / 0.8))
    assert total_variation(empirical_distribution(result.token_ids for result in results), target) < 0.03
    # A cut past the end of a stopped output is skipped without a proposal.
    assert all(result.attempts + result.skipped == 12 for result in results)
    assert sum(result.skipped for result in results) > 0


@pytest.mark.parametrize("sampling", [SamplingConfig(top_k=1), SamplingConfig(top_p=0.9)])
def test_the_chain_rejects_truncated_proposals(sampling) -> None:
    with pytest.raises(ValueError):
        _chain(TabularAutoregressiveBackend({}, fallback=[0.8, 0.2]), _config(alpha=4.0), sampling, 0)


def test_suffix_replay_leaves_both_targets_unchanged_and_replays_repeated_tokens() -> None:
    backend = TabularAutoregressiveBackend({(): (0.5, 0.3, 0.2), (0,): (0.7, 0.2, 0.1)}, fallback=(0.45, 0.45, 0.1))
    base = SamplingConfig(eos_token_id=2)
    power = [_chain(backend, _config(alpha=3.0, total_length=5, steps_per_block=4, suffix_replay=replay),
                    SamplingConfig(temperature=1 / 3, eos_token_id=2), 5, base=base) for replay in (False, True)]
    rewarded = [_chain(backend, _config(alpha=1, reward_temperature=0.5, total_length=5, steps_per_block=4,
                                        suffix_replay=replay), base, 6, base=base,
                       reward=pointwise(lambda _prompt, sequence: float(sum(sequence)))) for replay in (False, True)]
    for plain, replayed in (power, rewarded):
        assert (plain.token_ids, plain.base_token_logprobs) == (replayed.token_ids, replayed.base_token_logprobs)
        assert [replace(step, replayed_tokens=0) for step in replayed.trace] == list(plain.trace)
        assert replayed.replayed_tokens > 0 == plain.replayed_tokens


def test_early_rejection_and_replay_leave_the_power_chain_unchanged() -> None:
    backend = TabularAutoregressiveBackend({(): (0.5, 0.3, 0.2), (0,): (0.7, 0.2, 0.1)}, fallback=(0.45, 0.45, 0.1))
    runs = {(rejection, replay): _chain(backend, _config(
        alpha=3.0, total_length=6, block_size=3, steps_per_block=6, early_rejection=rejection, suffix_replay=replay),
        SamplingConfig(temperature=1 / 3, eos_token_id=2), 11, base=SamplingConfig(eos_token_id=2))
        for rejection in (False, True) for replay in (False, True)}
    plain = runs[False, False]
    for run in runs.values():
        assert (run.token_ids, [(step.cut, step.accepted) for step in run.trace]) == (
            plain.token_ids, [(step.cut, step.accepted) for step in plain.trace])
    assert runs[True, True].early_rejected > 0 == plain.early_rejected


def test_early_rejection_needs_a_power_target_and_a_monotone_proposal_temperature() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    with pytest.raises(ValueError, match="between 1/alpha and 1"):
        _chain(backend, _config(early_rejection=True), SamplingConfig(temperature=0.4), 1)
    with pytest.raises(ValueError, match="without a reward"):
        _chain(backend, _config(alpha=1, reward_temperature=1.0, early_rejection=True), BASE, 1,
               reward=pointwise(lambda *_: 0.0))


def _history(backend, count, sampling):
    return [(sample.token_ids, sample.token_logprobs, sample.token_cdf_bounds) for sample in backend.sample_batch([
        GenerationRequest((), 4, sampling, seed, str(seed)) for seed in range(count)])]


def test_a_frozen_history_of_weight_zero_is_the_plain_reward_chain() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.8, 0.2))
    config, reward = _config(alpha=1, reward_temperature=0.5, total_length=4, steps_per_block=8), pointwise(
        lambda _, tokens: float(sum(tokens)))
    plain = _chain(backend, config, BASE, 4, reward=reward)
    assert plain == _chain(backend, config, BASE, 4, reward=reward, history=_history(backend, 3, BASE))
    with pytest.raises(ValueError, match="rewarded, full-length chain"):
        _chain(backend, _config(), BASE, 4, history=_history(backend, 3, BASE), history_mixture=0.5)


class GenerationOnlyBackend(TabularAutoregressiveBackend):
    def score_batch(self, requests):
        raise AssertionError("replay reuses the history's generation log-probabilities")


def test_a_frozen_history_proposal_approaches_the_exact_reward_target() -> None:
    probabilities, temperature = (0.65, 0.35), 0.7
    history = [(sequence, tuple(log(probabilities[token]) for token in sequence), None)
               for sequence in ((1, 1),) * 40 + ((1, 0),) * 10]

    def reward(_, sequence):
        return float(sequence == (1, 1))

    backend = GenerationOnlyBackend({}, fallback=probabilities)
    config = _config(alpha=1, reward_temperature=temperature, block_size=1, steps_per_block=20)
    outputs = [_chain(backend, config, BASE, 117, reward=pointwise(reward), history=history, history_mixture=0.65,
                      chain_id=chain) for chain in range(2500)]
    target = _target(probabilities, 2, lambda sequence, probability: probability * exp(reward((), sequence) / temperature))
    assert total_variation(empirical_distribution(result.token_ids for result in outputs), target) < 0.04
    assert {step.proposal_source for result in outputs for step in result.trace} == {"base", "history"}


def test_suffix_replay_leaves_the_frozen_history_chain_unchanged() -> None:
    backend = TabularAutoregressiveBackend({(): (0.5, 0.3, 0.2), (0,): (0.7, 0.2, 0.1)}, fallback=(0.45, 0.45, 0.1))
    base = SamplingConfig(eos_token_id=2)
    runs = [_chain(backend, _config(alpha=1, reward_temperature=0.5, total_length=4, steps_per_block=6,
                                    suffix_replay=replay), base, 9, base=base,
                   reward=pointwise(lambda _prompt, sequence: float(sequence.count(1))),
                   history=_history(backend, 6, base), history_mixture=0.4) for replay in (False, True)]
    assert runs[0].token_ids == runs[1].token_ids and runs[0].reward == runs[1].reward
    assert [replace(step, replayed_tokens=0) for step in runs[1].trace] == list(runs[0].trace)
    assert runs[1].replayed_tokens > 0 == runs[0].replayed_tokens

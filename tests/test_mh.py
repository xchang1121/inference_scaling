from dataclasses import replace
from itertools import product
from math import exp, log
from types import SimpleNamespace

import numpy as np

import pytest

from inference_scaling.arllm.algorithms.config import MHConfig
from inference_scaling.arllm.algorithms.mh import run_mh_chain, suffix_length_probabilities
from tests.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample
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


@pytest.mark.parametrize("mode,temperature,eos", [("power", 1.0, None), ("power", .6, None),
                                                 ("power", 1.0, 1), ("reward", 1.0, 1), ("history", 1.0, None)])
def test_exact_transition_matrix_preserves_the_target(monkeypatch, mode, temperature, eos):
    # Enumerate every complete state and suffix proposal; integrate acceptance
    # from the real chain's trace instead of running thousands of random chains.
    probabilities, length = (.65, .35), 3
    base = SamplingConfig(temperature=temperature, eos_token_id=eos)
    proposal = base if mode == "history" else replace(base, temperature=.7 * temperature)
    config = _config(total_length=length, block_size=length, alpha=2 if mode == "power" else 1,
                     reward_temperature=None if mode == "power" else .8)
    states = [tokens for size in range(1, length + 1) for tokens in product((0, 1), repeat=size)
              if (eos in tokens and tokens.index(eos) == size - 1) or (eos not in tokens and size == length)]
    history = [(tokens, tuple(log(probabilities[token]) for token in tokens), None)
               for tokens in ((1, 1, 1),) * 4 + ((1, 0, 0),)]
    reward = lambda _, tokens: float(sum(tokens))
    mixture = .65 if mode == "history" else 0

    class ForcedSeeds(SeedStream):
        def generator(self, *key):
            # Force the base branch while checking the exact mixture density.
            return SimpleNamespace(random=lambda: .99) if key[-1] == "source" else super().generator(*key)

    class Draws(TabularAutoregressiveBackend):
        def sample_batch(self, requests):
            return [SequenceSample(
                request.prefix, current if request.request_id.endswith(":extend") else proposed[cut:],
                super(Draws, self).score_batch([ScoreRequest(request.prefix, (
                    current if request.request_id.endswith(":extend") else proposed[cut:],), request.sampling)])[0],
                request.sampling.policy_id, self.model_id, request.request_id,
                finish_reason="eos" if eos in (current if request.request_id.endswith(":extend") else proposed) else "length",
            ) for request in requests]

    backend = Draws({}, fallback=probabilities)
    masses = [exp(config.alpha * sum(backend.score_batch([ScoreRequest((), (state,), base)])[0])
                  + (reward((), state) / .8 if mode != "power" else 0)) for state in states]
    target = np.asarray(masses) / sum(masses)
    kernels = []
    for cut in range(length):
        monkeypatch.setattr("inference_scaling.arllm.algorithms.mh._draw_suffix",
                            lambda cut=cut, **_: (cut, length - cut, 1.0))
        matrix = np.zeros((len(states), len(states)))
        for row, current in enumerate(states):
            if cut >= len(current):
                result = run_mh_chain(backend, (), config, ForcedSeeds(1), base=base, proposal=proposal,
                                      reward=None if mode == "power" else pointwise(reward))
                assert result.skipped == 1 and result.token_ids == current
                matrix[row, row] = 1
                continue
            matches = [tokens for tokens, _, _ in history if tokens[:cut] == current[:cut]]
            for column, proposed in enumerate(states):
                if proposed[:cut] != current[:cut] or len(proposed) <= cut:
                    continue
                result = run_mh_chain(backend, (), config, ForcedSeeds(1), base=base, proposal=proposal,
                                      reward=None if mode == "power" else pointwise(reward),
                                      history=history if mixture else (), history_mixture=mixture)
                density = exp(sum(backend.score_batch([ScoreRequest(current[:cut], (proposed[cut:],), proposal)])[0]))
                if mixture and matches:
                    density = (1 - mixture) * density + mixture * matches.count(proposed) / len(matches)
                accept = exp(result.trace[0].log_acceptance)
                matrix[row, column] += density * accept
                matrix[row, row] += density * (1 - accept)
        kernels.append(matrix)
    for schedule in ("uniform", "inverse_length", "multiscale"):
        matrix = sum(weight * kernel for weight, kernel in zip(
            reversed(suffix_length_probabilities(length, schedule)), kernels, strict=True))
        np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-13, rtol=0)
        np.testing.assert_allclose(target @ matrix, target, atol=1e-13, rtol=0)


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

from collections import Counter
from math import exp

import numpy as np
import pytest

from inference_scaling.arllm.algorithms.conditional_is import (ConditionalISAdapter, RetainedSequence,
                                                               conditional_is_step, run_conditional_is)
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import ScoreRequest
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.importance import normalize_log_weights
from inference_scaling.shared.types import pointwise


def _backend() -> TabularAutoregressiveBackend:
    return TabularAutoregressiveBackend({(): [0.7, 0.3], (0,): [0.9, 0.1], (1,): [0.2, 0.8]}, fallback=[0.5, 0.5])


def _reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def _exact_first_token_target() -> dict[int, float]:
    base_first = (0.7, 0.3)
    completion = ((0.9, 0.1), (0.2, 0.8))
    weights = [base_first[candidate] * sum(completion[candidate][token] * exp(_reward((), (candidate, token)))
                                           for token in (0, 1)) for candidate in (0, 1)]
    total = sum(weights)
    return {index: weights[index] / total for index in (0, 1)}


def _step(backend, config, *, state=RetainedSequence(), reward=_reward, sampling=SamplingConfig(), seed=1):
    return conditional_is_step(backend=backend, prompt=(), state=state, config=config, sampling=sampling,
                               reward=pointwise(reward), seeds=SeedStream(seed), step_index=0)


def test_first_block_approaches_the_exact_conditional_target() -> None:
    config = ConditionalISConfig(candidate_count=12, rollout_count=8, block_size=1, total_length=2, reward_temperature=1.0)
    counts: Counter[int] = Counter()
    trials = 500
    for trial in range(trials):
        step, _ = _step(_backend(), config, seed=10_000 + trial)
        counts[step.selected.token_ids[0]] += 1
    empirical = {token: count / trials for token, count in counts.items()}
    assert total_variation(empirical, _exact_first_token_target()) < 0.08


def test_completions_run_from_the_end_of_the_block() -> None:
    step, kept = _step(
        TabularAutoregressiveBackend({}, fallback=[0.5, 0.5]),
        ConditionalISConfig(candidate_count=3, rollout_count=2, block_size=2, total_length=5, reward_temperature=1.0),
        reward=lambda _prompt, generated: float(sum(generated)),
        seed=4,
    )
    assert all(len(candidate.token_ids) == 2 for candidate in step.candidates)
    assert all(len(rollout.token_ids) == 3 for candidate in step.candidates for rollout in candidate.rollouts)
    assert len(kept.token_ids) == 5 and kept.fixed == 2


@pytest.mark.parametrize("total_length", [6, 3])
def test_early_eos_candidate_does_not_lengthen_other_completions(total_length) -> None:
    # EOS is likely only as the first token, so candidate 0 can stop at length 1
    # while the others fill the whole block. With total_length == block_size the
    # block is terminal: length-capped candidates must be scored without rollouts.
    backend = TabularAutoregressiveBackend({(): [0.25, 0.25, 0.5]}, fallback=[0.49, 0.49, 0.02])
    config = ConditionalISConfig(candidate_count=4, rollout_count=1, block_size=3, total_length=total_length,
                                 reward_temperature=1.0)
    step, _ = _step(backend, config, reward=lambda _prompt, generated: float(len(generated)),
                    sampling=SamplingConfig(eos_token_id=2))
    lengths = [len(candidate.token_ids) for candidate in step.candidates]
    assert lengths[0] == 1 and 3 in lengths
    assert all(len(candidate.token_ids) + len(rollout.token_ids) <= config.total_length
               for candidate in step.candidates for rollout in candidate.rollouts)
    if total_length == config.block_size:
        assert all(rollout.token_ids == () for c in step.candidates for rollout in c.rollouts)


def test_conditional_is_returns_a_complete_sequence_within_total_length() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    result = run_conditional_is(
        backend, (), ConditionalISConfig(candidate_count=2, rollout_count=2, block_size=2, total_length=5,
                                         reward_temperature=1.0),
        pointwise(lambda _prompt, generated: float(sum(generated))), SeedStream(17))
    assert len(result.token_ids) == 5
    assert [step.generated_length_before for step in result.steps] == [0, 2, 4]
    assert [len(step.selected.token_ids) for step in result.steps] == [2, 2, 1]


@pytest.mark.parametrize("sampling", [SamplingConfig(top_p=0.9), SamplingConfig(top_k=1)])
def test_conditional_is_rejects_policies_that_break_the_weight_formula(sampling) -> None:
    with pytest.raises(ValueError):
        run_conditional_is(_backend(), (), ConditionalISConfig(candidate_count=2, rollout_count=2, block_size=1,
                                                               total_length=2, reward_temperature=1.0),
                           pointwise(_reward), SeedStream(1), sampling=sampling)


def test_conditional_is_scores_one_batch_with_generation_logprobs() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    seen: list[tuple[tuple[int, ...], ...]] = []

    def reward_batch(_prompt, generated, logprobs):
        seen.append(tuple(generated))
        # Every generated token, candidate and completion alike, comes with its log-probability.
        assert all(tuple(values) == pytest.approx((np.log(0.5),) * len(tokens))
                   for values, tokens in zip(logprobs, generated, strict=True))
        return tuple(float(tokens[-1] == 1) for tokens in generated)

    result = run_conditional_is(backend, (), ConditionalISConfig(candidate_count=2, rollout_count=2, block_size=1,
                                                                 total_length=2, reward_temperature=1.0),
                                reward_batch, SeedStream(91))
    assert len(result.token_ids) == 2
    assert len(seen[0]) == 4


def test_kept_completion_is_reused_without_rescoring() -> None:
    scored: list[tuple[int, ...]] = []

    def reward(_prompt, generated) -> float:
        scored.append(tuple(generated))
        return float(sum(generated))

    config = ConditionalISConfig(candidate_count=3, rollout_count=2, block_size=1, total_length=3, reward_temperature=1.0)
    result = run_conditional_is(_backend(), (), config, pointwise(reward), SeedStream(7))
    sequence: tuple[int, ...] | None = None
    kept_reward = 0.0
    for step in result.steps:
        fixed = step.generated_length_before
        assert step.retained_candidate == (sequence is not None)
        if sequence is not None:
            carried = step.candidates[0]
            assert carried.token_ids == sequence[fixed : fixed + 1]
            assert carried.rollouts[0].token_ids == sequence[fixed + 1 :]
            assert carried.rollouts[0].reward == kept_reward
        completion = step.selected.rollouts[step.completion_index]
        sequence = (sequence or ())[:fixed] + step.selected.token_ids + completion.token_ids
        kept_reward = completion.reward
    assert result.token_ids == sequence and len(sequence) == 3
    # Only fresh sequences are scored; the kept completion is never scored again.
    assert len(scored) == sum(step.rollout_evaluations_performed for step in result.steps) == 13


def test_steps_started_at_the_target_stay_at_the_target() -> None:
    # Each step is a conditional SIR move, so a pass started from exact target
    # samples must return exact target samples; the same pass started from
    # nothing is only an approximation.
    backend = TabularAutoregressiveBackend({(): [0.6, 0.4], (0,): [0.8, 0.2], (1,): [0.3, 0.7]}, fallback=[0.5, 0.5])
    sampling = SamplingConfig()

    def reward(_prompt, generated) -> float:
        return 2.0 * sum(generated)

    sequences = [tuple(int(bit) for bit in f"{code:03b}") for code in range(8)]
    logprobs = {sequence: backend.score_batch([ScoreRequest((), (sequence,), sampling)])[0] for sequence in sequences}
    weights = [exp(sum(logprobs[sequence]) + reward((), sequence)) for sequence in sequences]
    target = {sequence: weight / sum(weights) for sequence, weight in zip(sequences, weights)}
    adapter = ConditionalISAdapter(
        backend=backend, prompt=(), sampling=sampling, reward=pointwise(reward),
        config=ConditionalISConfig(candidate_count=2, rollout_count=2, block_size=1, total_length=3, reward_temperature=1.0))
    starts = np.random.default_rng(0).choice(8, size=3000, p=list(target.values()))
    counts: dict[str, Counter[tuple[int, ...]]] = {"target": Counter(), "empty": Counter()}
    for trial, index in enumerate(starts):
        start = sequences[index]
        for label, state in (("target", RetainedSequence(start, logprobs[start], reward((), start))),
                             ("empty", adapter.initial_state)):
            seeds = SeedStream(trial)
            step_index = 0
            while not adapter.is_terminal(state):
                _, state = adapter.step(state, step_index, seeds)
                step_index += 1
            counts[label][state.token_ids] += 1
    empirical = {label: {sequence: counts[label][sequence] / len(starts) for sequence in sequences} for label in counts}
    assert total_variation(empirical["target"], target) < 0.03
    assert total_variation(empirical["empty"], target) > 0.2


def test_log_weight_normalization_matches_softmax() -> None:
    assert normalize_log_weights((0.0, float(np.log(3.0)))) == pytest.approx((0.25, 0.75))

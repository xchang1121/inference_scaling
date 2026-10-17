from dataclasses import replace

import pytest

from inference_scaling.arllm.algorithms.conditional_is import (ConditionalISAdapter, RetainedSequence,
                                                               run_conditional_is)
from tests.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import pointwise


def _backend() -> TabularAutoregressiveBackend:
    return TabularAutoregressiveBackend({(): [0.7, 0.3], (0,): [0.9, 0.1], (1,): [0.2, 0.8]}, fallback=[0.5, 0.5])


def _reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def _step(backend, config, *, state=RetainedSequence(), reward=_reward, sampling=SamplingConfig(), seed=1):
    adapter = ConditionalISAdapter(backend=backend, prompt=(), config=config, sampling=sampling, reward=pointwise(reward))
    return adapter.step(state, 0, SeedStream(seed))


def test_completions_run_from_the_end_of_the_block() -> None:
    step, kept = _step(
        TabularAutoregressiveBackend({}, fallback=[0.5, 0.5]),
        ConditionalISConfig(block_first=False, candidate_count=3, rollout_count=2, block_size=2, total_length=5, reward_temperature=1.0),
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
    config = ConditionalISConfig(block_first=False, candidate_count=4, rollout_count=1, block_size=3, total_length=total_length,
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
        backend, (), ConditionalISConfig(block_first=False, candidate_count=2, rollout_count=2, block_size=2, total_length=5,
                                         reward_temperature=1.0),
        pointwise(lambda _prompt, generated: float(sum(generated))), SeedStream(17))
    assert len(result.token_ids) == 5
    assert [step.generated_length_before for step in result.steps] == [0, 2, 4]
    assert [step.selected.token_count for step in result.steps] == [2, 2, 1]
    assert all(not hasattr(candidate, "token_ids") and all(not hasattr(leaf, "token_ids") for leaf in candidate.rollouts)
               for step in result.steps for candidate in step.candidates)


@pytest.mark.parametrize("sampling", [SamplingConfig(top_p=0.9), SamplingConfig(top_k=1)])
def test_conditional_is_rejects_policies_that_break_the_weight_formula(sampling) -> None:
    with pytest.raises(ValueError):
        run_conditional_is(_backend(), (), ConditionalISConfig(block_first=False, candidate_count=2, rollout_count=2, block_size=1,
                                                               total_length=2, reward_temperature=1.0),
                           pointwise(_reward), SeedStream(1), sampling=sampling)


def test_conditional_is_scores_a_step_in_one_batch() -> None:
    seen: list[tuple[tuple[int, ...], ...]] = []

    def reward_batch(_prompt, generated):
        seen.append(tuple(generated))
        return tuple(float(tokens[-1] == 1) for tokens in generated)

    result = run_conditional_is(TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)), (), ConditionalISConfig(
        block_first=False, candidate_count=2, rollout_count=2, block_size=1, total_length=2, reward_temperature=1.0),
        reward_batch, SeedStream(91))
    assert len(result.token_ids) == 2 and len(seen[0]) == 4


class _Measured(TabularAutoregressiveBackend):
    """The tabular backend reporting a per-request cost, as a metered engine does."""

    def sample_batch(self, requests):
        samples = [replace(sample, forward_token_slots=len(sample.prefix) + max(0, len(sample.token_ids) - 1))
                   for sample in super().sample_batch(requests)]
        self.actual_slots = getattr(self, "actual_slots", 0) + sum(sample.forward_token_slots for sample in samples)
        return samples


@pytest.mark.parametrize("block_first,rollouts,width", [(False, 1, 5), (False, 2, 1), (True, 1, 2), (True, 2, 1)])
def test_is_reports_the_forward_slots_its_requests_cost(block_first, rollouts, width) -> None:
    config = ConditionalISConfig(block_first=block_first, candidate_count=2, rollout_count=rollouts, block_size=width,
                                 total_length=5, reward_temperature=1.0)
    backend = _Measured({}, fallback=(0.5, 0.5))
    measured = run_conditional_is(backend, (), config, pointwise(_reward), SeedStream(91))
    assert measured.forward_token_slots == sum(step.forward_token_slots for step in measured.steps)
    assert measured.forward_token_slots == backend.actual_slots > 0
    # A backend that does not measure per request leaves the cost unknown rather than zero.
    silent = run_conditional_is(TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)), (), config, pointwise(_reward),
                                SeedStream(91))
    assert silent.token_ids == measured.token_ids and silent.forward_token_slots is None


def test_kept_completion_is_reused_without_rescoring() -> None:
    scored: list[tuple[int, ...]] = []

    def reward(_prompt, generated) -> float:
        scored.append(tuple(generated))
        return float(sum(generated))

    config = ConditionalISConfig(block_first=False, candidate_count=3, rollout_count=2, block_size=1, total_length=3, reward_temperature=1.0)
    adapter = ConditionalISAdapter(backend=_backend(), prompt=(), config=config, sampling=SamplingConfig(), reward=pointwise(reward))
    state, evaluations, index = adapter.initial_state, 0, 0
    sequence: tuple[int, ...] | None = None
    kept_reward = 0.0
    while not adapter.is_terminal(state):
        step, state = adapter.step(state, index, SeedStream(7))
        index += 1
        evaluations += step.rollout_evaluations_performed
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
    assert state.token_ids == sequence and len(sequence) == 3
    # Only fresh sequences are scored; the kept completion is never scored again.
    assert len(scored) == evaluations == 13


def test_block_first_candidates_leave_every_step_unchanged() -> None:
    backend = TabularAutoregressiveBackend({(): [0.3, 0.3, 0.4], (0,): [0.5, 0.2, 0.3]}, fallback=[0.4, 0.4, 0.2])
    runs = [run_conditional_is(backend, (), ConditionalISConfig(
        block_first=block_first, candidate_count=3, rollout_count=2, block_size=2, total_length=7, reward_temperature=0.5),
        pointwise(lambda _prompt, generated: float(sum(generated))), SeedStream(5), sampling=SamplingConfig(eos_token_id=2))
        for block_first in (False, True)]
    assert runs[0].token_ids == runs[1].token_ids and runs[0].steps == runs[1].steps

from dataclasses import replace

import numpy as np
import pytest

from inference_scaling.arllm.algorithms.conditional_is import (
    ConditionalISAdapter,
    RetainedSequence,
    run_conditional_is,
)
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import (
    ConditionalCandidate,
    RolloutEvaluation,
    _metropolized_probabilities,
    select_metropolized,
)
from inference_scaling.shared.types import pointwise
from tests.tabular import TabularAutoregressiveBackend


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
        ConditionalISConfig(candidate_count=3, rollout_count=2, stages=2, total_length=5, reward_temperature=1.0, base_block_size=2),
        reward=lambda _prompt, generated: float(sum(generated)),
        seed=4,
    )
    assert all(len(candidate.token_ids) == 2 for candidate in step.candidates)
    assert all(len(rollout.token_ids) == 3 for candidate in step.candidates for rollout in candidate.rollouts)
    assert len(kept.token_ids) == 5 and kept.fixed == 2


@pytest.mark.parametrize("total_length,stages", [(6, 2), (3, 1)])
def test_early_eos_candidate_does_not_lengthen_other_completions(total_length, stages) -> None:
    # With stages=1, the block covers total_length and is terminal: candidates are scored without extra rollouts.
    backend = TabularAutoregressiveBackend({(): [0.25, 0.25, 0.5]}, fallback=[0.49, 0.49, 0.02])
    config = ConditionalISConfig(
        candidate_count=4, rollout_count=1, stages=stages, total_length=total_length,
        reward_temperature=1.0, base_block_size=3,
    )
    step, _ = _step(backend, config, reward=lambda _prompt, generated: float(len(generated)),
                    sampling=SamplingConfig(eos_token_id=2))
    assert all(len(candidate.token_ids) + len(rollout.token_ids) <= config.total_length
               for candidate in step.candidates for rollout in candidate.rollouts)
    if stages == 1:
        assert all(rollout.token_ids == () for c in step.candidates for rollout in c.rollouts)


@pytest.mark.parametrize("stages,candidates", [(1, 4), (2, 4), (4, 6)])
def test_conditional_is_returns_a_complete_sequence_within_total_length(stages, candidates) -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    committed: list[tuple[int, ...]] = []
    config = ConditionalISConfig(
        candidate_count=candidates, rollout_count=2, stages=stages, total_length=16,
        reward_temperature=1.0, base_block_size=2,
    )
    reward = pointwise(lambda _prompt, generated: float(sum(tok * (i + 1) for i, tok in enumerate(generated))))
    result = run_conditional_is(backend, (), config, reward, SeedStream(17), on_commit=committed.append)
    assert len(result.token_ids) == 16
    assert 1 <= len(result.steps) <= stages
    assert committed[-1] == result.token_ids
    assert all(result.token_ids[:len(prefix)] == prefix for prefix in committed)
    assert run_conditional_is(backend, (), config, reward, SeedStream(17)) == result
    assert all(not hasattr(candidate, "token_ids") and all(not hasattr(leaf, "token_ids") for leaf in candidate.rollouts)
               for step in result.steps for candidate in step.candidates)
    if stages >= 2:
        committed.clear()
        stopped = run_conditional_is(backend, (), config, reward, SeedStream(17), deadline=0, on_commit=committed.append)
        assert stopped.stopping_reason == "deadline"
        assert len(stopped.steps) == 1
        assert committed[-1] == stopped.token_ids


def test_conditional_is_accepts_truncated_sampling_on_policy() -> None:
    for sampling in (SamplingConfig(top_p=0.9), SamplingConfig(top_k=1)):
        result = run_conditional_is(
            _backend(), (),
            ConditionalISConfig(candidate_count=2, rollout_count=2, stages=2, total_length=2, reward_temperature=1.0, base_block_size=1),
            pointwise(_reward), SeedStream(1), sampling=sampling,
        )
        assert len(result.token_ids) == 2


def test_conditional_is_scores_a_step_in_one_batch() -> None:
    seen: list[tuple[tuple[int, ...], ...]] = []

    def reward_batch(_prompt, generated):
        seen.append(tuple(generated))
        return tuple(float(tokens[-1] == 1) for tokens in generated)

    result = run_conditional_is(
        TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)), (),
        ConditionalISConfig(candidate_count=2, rollout_count=2, stages=2, total_length=2, reward_temperature=1.0, base_block_size=1),
        reward_batch, SeedStream(91),
    )
    assert len(result.token_ids) == 2 and len(seen[0]) == 4


class _Measured(TabularAutoregressiveBackend):
    """The tabular backend reporting a per-request cost, as a metered engine does."""

    def sample_batch(self, requests):
        samples = [replace(sample, forward_token_slots=len(sample.prefix) + max(0, len(sample.token_ids) - 1))
                   for sample in super().sample_batch(requests)]
        self.actual_slots = getattr(self, "actual_slots", 0) + sum(sample.forward_token_slots for sample in samples)
        return samples


@pytest.mark.parametrize("stages,rollouts,width", [(1, 1, 5), (2, 2, 1), (3, 2, 2)])
def test_is_reports_the_forward_slots_its_requests_cost(stages, rollouts, width) -> None:
    config = ConditionalISConfig(
        candidate_count=4, rollout_count=rollouts, stages=stages, total_length=8,
        reward_temperature=1.0, base_block_size=width,
    )
    backend = _Measured({}, fallback=(0.5, 0.5))
    measured = run_conditional_is(backend, (), config, pointwise(_reward), SeedStream(91))
    assert measured.forward_token_slots == sum(step.forward_token_slots for step in measured.steps)
    assert measured.forward_token_slots == backend.actual_slots > 0
    silent = run_conditional_is(TabularAutoregressiveBackend({}, fallback=(0.5, 0.5)), (), config, pointwise(_reward),
                                SeedStream(91))
    assert silent.token_ids == measured.token_ids and silent.forward_token_slots is None


def test_kept_completion_is_reused_without_rescoring() -> None:
    scored: list[tuple[int, ...]] = []

    def reward(_prompt, generated) -> float:
        scored.append(tuple(generated))
        return float(sum(generated))

    config = ConditionalISConfig(
        candidate_count=3, rollout_count=2, stages=3, total_length=3, reward_temperature=1.0, base_block_size=1,
    )
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
    assert len(scored) == evaluations == 13


def test_metropolized_index_balance_and_block_grouping() -> None:
    for mass in ([1.], [1., 2., 4.], [1., 0., 1e-320], [0., 1., 3.]):
        weights = np.array(mass)
        matrix = np.empty((len(mass), len(mass)))
        for i in range(len(mass)):
            order = [i, *(j for j in range(len(mass)) if j != i)]
            matrix[i, order] = _metropolized_probabilities(weights[order])
        assert np.all(matrix >= 0) and np.allclose(matrix.sum(axis=1), 1)
        flow = weights[:, None] * matrix
        assert np.allclose(flow, flow.T, rtol=1e-12, atol=1e-15)
    candidates = tuple(ConditionalCandidate((i,), (RolloutEvaluation((i,), 0., np.log(w)),), np.log(w))
                       for i, w in enumerate((1., 2., 4.)))
    for seed in range(32):
        assert select_metropolized(candidates, SeedStream(seed), group_blocks=False) == select_metropolized(
            candidates, SeedStream(seed), group_blocks=True)
    # Two equal leaves in the reference block, one in a second block: both
    # blocks have unit mass, so a block move must leave the reference block.
    leaf = RolloutEvaluation((), 0., 0.)
    grouped = (ConditionalCandidate((0,), (leaf, leaf), 0.), ConditionalCandidate((1,), (leaf,), 0.))
    assert all(select_metropolized(grouped, SeedStream(seed), group_blocks=True)[0] == 1 for seed in range(32))

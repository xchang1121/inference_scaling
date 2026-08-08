from math import exp, log

import numpy as np
import pytest

from inference_scaling.algorithms.base_replay import (
    ProbabilityObservation,
    base_replay_step,
    corrected_replay_log_energy,
    run_base_replay,
)
from inference_scaling.backends import TabularAutoregressiveBackend
from inference_scaling.config import BaseReplayConfig, SamplingConfig
from inference_scaling.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    sample_replay_record,
)
from inference_scaling.rng import SeedStream


def test_truncated_history_and_fresh_tail_are_exact_in_expectation() -> None:
    p = np.asarray([0.8, 0.2])
    q = np.asarray([0.55, 0.45])
    rewards = np.asarray([0.0, 1.0])
    truncation = 1.25
    expected_estimator = 0.0
    for history_token in (0, 1):
        for fresh_token in (0, 1):
            log_energy, _, _ = corrected_replay_log_energy(
                [
                    ProbabilityObservation(
                        log(p[history_token]), log(q[history_token]), rewards[history_token]
                    )
                ],
                [
                    ProbabilityObservation(
                        log(p[fresh_token]), log(q[fresh_token]), rewards[fresh_token]
                    )
                ],
                truncation=truncation,
                reward_temperature=1.0,
            )
            expected_estimator += q[history_token] * p[fresh_token] * exp(log_energy)
    exact_energy = float(np.dot(p, np.exp(rewards)))
    assert expected_estimator == pytest.approx(exact_energy)


def test_evaluation_claims_are_metadata_only_disjoint_and_single_use() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.6, 0.4])
    behavior = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.8), label="behavior"
    )
    key = ReplayKey((), (), (0,), "reward-v1")
    store = InMemoryReplayStore()
    for index in range(3):
        store.add_evaluation(
            sample_replay_record(
                behavior,
                key,
                1,
                lambda _prompt, generated: float(sum(generated)),
                seed=index,
                record_id=f"record-{index}",
            )
        )

    first, second = store.freeze_claims([key, key], 2)
    assert first.count == 2
    assert second.count == 1
    assert store.inventory(key) == {}
    assert store.reserved_count == 3
    first_records = store.reveal_and_consume(first)
    second_records = store.reveal_and_consume(second)
    assert set(record.record_id for record in first_records).isdisjoint(
        record.record_id for record in second_records
    )
    assert store.design_count == 3
    with pytest.raises(ValueError):
        store.reveal_and_consume(first)


def test_base_replay_uses_matching_history_and_moves_all_used_data_to_design() -> None:
    backend = TabularAutoregressiveBackend(
        {(): [1.0, 0.0], (0,): [0.7, 0.3]}, fallback=[0.5, 0.5]
    )
    behavior = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.6), label="old-policy"
    )
    registry = BehaviorRegistry([behavior])
    store = InMemoryReplayStore()
    key = ReplayKey((), (), (0,), "reward-v1")
    for index in range(2):
        store.add_evaluation(
            sample_replay_record(
                behavior,
                key,
                1,
                lambda _prompt, generated: float(generated[-1]),
                seed=100 + index,
                record_id=f"history-{index}",
            )
        )

    step = base_replay_step(
        base_backend=backend,
        registry=registry,
        store=store,
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=1,
            block_size=1,
            total_length=2,
            max_history_per_candidate=2,
            fresh_rollouts=2,
            truncation=1.5,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(generated[-1]),
        reward_version="reward-v1",
        seeds=SeedStream(7),
        step_index=0,
    )
    estimate = step.selected.estimate
    assert estimate.history_count == 2
    assert estimate.fresh_count == 2
    assert store.evaluation_count == 0
    assert store.design_count == 4
    assert estimate.behavior_counts == (("old-policy", 2),)


def test_no_history_reduces_to_fresh_base_completion_mean() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    store = InMemoryReplayStore()
    step = base_replay_step(
        base_backend=backend,
        registry=BehaviorRegistry(),
        store=store,
        prompt=(),
        generated_prefix=(),
        config=BaseReplayConfig(
            candidate_count=2,
            block_size=1,
            total_length=2,
            max_history_per_candidate=3,
            fresh_rollouts=3,
        ),
        base_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        reward_version="reward-v1",
        seeds=SeedStream(19),
        step_index=0,
    )
    assert all(candidate.estimate.history_count == 0 for candidate in step.candidates)
    assert all(candidate.estimate.fresh_count == 3 for candidate in step.candidates)
    assert store.design_count == 6
    assert store.evaluation_count == 0


def test_post_selection_reserve_can_use_an_off_policy_behavior() -> None:
    backend = TabularAutoregressiveBackend(
        {
            (): [1.0, 0.0],
            (0,): [1.0, 0.0],
            (0, 0): [0.25, 0.75],
        },
        fallback=[0.5, 0.5],
    )
    reserve_policy = BehaviorPolicy.for_backend(
        backend, SamplingConfig(temperature=0.45), label="off-policy-reserve"
    )
    store = InMemoryReplayStore()
    result = run_base_replay(
        backend,
        BehaviorRegistry(),
        store,
        (),
        BaseReplayConfig(
            candidate_count=1,
            block_size=1,
            total_length=3,
            max_history_per_candidate=2,
            fresh_rollouts=1,
            reserve_rollouts=2,
        ),
        lambda _prompt, generated: float(generated[-1]),
        "reward-v1",
        SeedStream(23),
        reserve_policy=reserve_policy,
    )

    replayed = result.steps[1].selected.estimate
    assert result.reserve_records_written == 2
    assert replayed.behavior_counts == (("off-policy-reserve", 2),)
    assert replayed.history_count == 2
    assert store.evaluation_count == 0
    assert store.reserved_count == 0

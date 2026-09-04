from __future__ import annotations

import numpy as np

from online_speculation.online_simulation import (
    AdaptiveStrideController,
    CostModel,
    DriftSchedule,
    FeedbackItem,
    StrategyConfig,
    TabularFastDrafter,
    aggregate_outcomes,
    expected_committed_tokens,
    make_transition_matrix,
    offline_marginal_draft,
    simulate_strategy,
    supervision_weights,
)
from online_speculation.psi_spec import total_variation, uno_linear_step


def test_transition_matrices_and_offline_marginals_are_probabilities() -> None:
    transition = make_transition_matrix(8, shift=0)
    shifted = make_transition_matrix(8, shift=2)
    assert transition.shape == (8, 8)
    assert np.all(transition > 0)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    assert not np.allclose(transition, shifted)

    draft = offline_marginal_draft(transition, speculative_tokens=3, smoothing=0.0)
    np.testing.assert_allclose(draft[:, 0], transition @ transition)
    np.testing.assert_allclose(draft[:, 1], transition @ transition @ transition)
    np.testing.assert_allclose(draft.sum(axis=-1), 1.0)


def test_supervision_masks_rejected_tail_without_losing_verified_rows() -> None:
    np.testing.assert_array_equal(
        supervision_weights(5, 1, "full", 0.25),
        np.ones(5),
    )
    np.testing.assert_array_equal(
        supervision_weights(5, 1, "on_policy", 0.25),
        np.asarray([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        supervision_weights(5, 1, "discounted_tail", 0.25),
        np.asarray([1.0, 1.0, 0.25, 0.0625, 0.015625]),
    )
    np.testing.assert_array_equal(
        supervision_weights(5, None, "discounted_tail", 0.25),
        np.ones(5),
    )


def test_fast_weight_update_moves_draft_toward_verifier() -> None:
    offline = np.asarray([[[0.80, 0.10, 0.10]]], dtype=np.float64)
    target = np.asarray([0.10, 0.80, 0.10], dtype=np.float64)
    drafter = TabularFastDrafter(offline)
    before = drafter.distributions(0)[0]
    report = drafter.update(
        [FeedbackItem(0, 0, target, before.copy(), 1.0)],
        learning_rate=0.35,
        tv_weight=0.5,
        forward_kl_weight=1.0,
        old_q_weight=0.15,
        gradient_clip=1.0,
    )
    after = drafter.distributions(0)[0]
    assert total_variation(target, after) < total_variation(target, before)
    assert report.mean_tv_after < report.mean_tv_before
    assert report.mean_forward_kl_after < report.mean_forward_kl_before


def test_post_verification_update_cannot_mutate_current_round_old_q() -> None:
    offline = np.asarray(
        [[[0.65, 0.25, 0.10], [0.20, 0.30, 0.50]]],
        dtype=np.float64,
    )
    drafter = TabularFastDrafter(offline)
    q_used = drafter.distributions(0)
    target = np.asarray([0.15, 0.70, 0.15], dtype=np.float64)
    step = uno_linear_step(
        (0,),
        lambda _history: target,
        q_used,
        np.random.default_rng(17),
    )
    saved_denominator = step.draft_probabilities.copy()

    drafter.update(
        [FeedbackItem(0, 0, target, q_used[0].copy(), 1.0)],
        learning_rate=0.35,
        tv_weight=0.5,
        forward_kl_weight=1.0,
        old_q_weight=0.15,
        gradient_clip=1.0,
    )

    assert not np.allclose(drafter.distributions(0), q_used)
    np.testing.assert_array_equal(step.draft_probabilities, saved_denominator)
    np.testing.assert_allclose(step.draft_probabilities, q_used, atol=1e-15, rtol=0.0)


def test_expected_committed_tokens_has_correct_extremes() -> None:
    target = np.asarray(
        [[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]],
        dtype=np.float64,
    )
    assert expected_committed_tokens(target, target) == 5.0
    disjoint = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    peaked_target = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    assert expected_committed_tokens(peaked_target, disjoint) == 2.0


def test_adaptive_controller_increases_and_decreases_update_frequency() -> None:
    positive = AdaptiveStrideController(initial_stride=10, window_rounds=10)
    event = None
    for position in range(10):
        event = positive.observe(
            token_position=position,
            current_proxy_efficiency=1.20,
            static_proxy_efficiency=1.00,
            mean_tv=0.20,
            update_cost=0.01,
            total_cost=1.0,
        )
    assert event is not None
    assert event["new_stride"] == 5

    negative = AdaptiveStrideController(initial_stride=10, window_rounds=10)
    event = None
    for position in range(10):
        event = negative.observe(
            token_position=position,
            current_proxy_efficiency=0.90,
            static_proxy_efficiency=1.00,
            mean_tv=0.20,
            update_cost=0.01,
            total_cost=1.0,
        )
    assert event is not None
    assert event["new_stride"] == 20


def test_simulation_runs_static_and_post_verification_online_paths() -> None:
    schedule = DriftSchedule.create(vocabulary_size=8, total_tokens=1_200)
    offline = offline_marginal_draft(schedule.regime_0, speculative_tokens=3)
    cost = CostModel()
    static = simulate_strategy(
        schedule=schedule,
        offline_draft=offline,
        block_size=4,
        strategy=StrategyConfig("static", stride=None),
        cost_model=cost,
        seed=123,
        trace_window_tokens=100,
    )
    online = simulate_strategy(
        schedule=schedule,
        offline_draft=offline,
        block_size=4,
        strategy=StrategyConfig("stride5", stride=5, supervision="discounted_tail"),
        cost_model=cost,
        seed=123,
        trace_window_tokens=100,
    )

    assert static["summary"]["tokens"] == 1_200
    assert static["summary"]["updates"] == 0
    assert np.isclose(
        static["summary"]["dynamic_regret_proxy"],
        static["summary"]["static_regret_proxy"],
    )
    assert online["summary"]["tokens"] == 1_200
    assert online["summary"]["updates"] > 0
    assert online["summary"]["items_updated"] > 0
    assert online["summary"]["mean_update_tv_improvement"] > 0
    assert np.isfinite(online["summary"]["negative_log_likelihood_per_token"])


def test_aggregate_keeps_preregistered_decision_separate_from_selection() -> None:
    outcomes = []
    for seed, online_tpf, online_efficiency in ((1, 1.20, 1.10), (2, 1.22, 1.12)):
        for name, tpf, efficiency, dynamic_regret, update_cost in (
            ("static", 1.0, 1.0, 10.0, 0.0),
            (
                "stride10_discounted",
                online_tpf,
                online_efficiency,
                9.0,
                10.0,
            ),
        ):
            summary = {
                "tpf": tpf,
                "tokens_per_cost": efficiency,
                "dynamic_regret_proxy": dynamic_regret,
                "static_regret_proxy": 10.0,
                "updates": 10 if update_cost else 0,
                "update_cost_fraction": 0.05 if update_cost else 0.0,
                "spec_acceptance_rate": 0.6 if update_cost else 0.5,
                "rounds": 100,
                "update_cost": update_cost,
                "tokens": 240,
            }
            outcomes.append(
                {
                    "seed": seed,
                    "strategy": {"name": name},
                    "summary": summary,
                    "segments": {
                        "all": {"tokens_per_cost": efficiency},
                    },
                }
            )

    aggregate = aggregate_outcomes(
        outcomes,
        cost_model=CostModel(),
        bootstrap_samples=1_000,
        bootstrap_seed=9,
    )
    decision = aggregate["decision"]
    assert decision["preregistered_primary_strategy"] == "stride10_discounted"
    assert decision["primary_learning_success"]
    assert decision["primary_tpf_success"]
    assert decision["primary_proxy_system_success"]
    assert "selection_warning" in decision

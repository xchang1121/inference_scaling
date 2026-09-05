from __future__ import annotations

import numpy as np
import pytest

from online_speculation.psi_spec import one_token_output_distribution
from online_speculation.replay_cache import (
    CostAwareReplayRouter,
    ReplayCacheConfig,
    ReplayCandidate,
    ReplayRouteConfig,
    VerifierReplayCache,
    delta_draft_probabilities,
    independent_match_tpf,
    verify_greedy_replay,
)


def _cache(**overrides: object) -> VerifierReplayCache:
    values: dict[str, object] = {
        "min_suffix_length": 2,
        "max_suffix_length": 6,
        "max_continuation_length": 4,
        "min_confidence": 0.5,
    }
    values.update(overrides)
    return VerifierReplayCache(
        namespace="model@revision|greedy",
        config=ReplayCacheConfig(**values),
    )


def test_replay_cache_exposes_only_closed_verified_requests() -> None:
    cache = _cache()
    prompt = (10, 11, 12, 13)
    completion = (20, 21, 22, 23, 24, 25)
    assert cache.lookup(prompt, max_tokens=4) is None

    records = cache.observe_sequence(
        prompt_tokens=prompt,
        verified_completion_tokens=completion,
    )
    assert records > 0
    first = cache.lookup(prompt, max_tokens=3)
    assert first is not None
    assert first.token_ids == (20, 21, 22)
    assert first.matched_suffix_length == 4
    assert first.confidence == 1.0

    later = cache.lookup(prompt + completion[:3], max_tokens=4)
    assert later is not None
    assert later.token_ids == (23, 24, 25)
    assert later.matched_suffix_length == 6


def test_causal_session_exposes_only_fully_verified_local_horizons() -> None:
    cache = _cache(max_continuation_length=4)
    prompt = (10, 11, 12, 13)
    session = cache.begin_causal_session(prompt_tokens=prompt)

    assert session.append_verified((20, 21, 22)) == 0
    assert session.local_records == 0
    assert cache.lookup(prompt, max_tokens=4) is None
    assert session.lookup(prompt, max_tokens=4) is None

    assert session.append_verified((23,)) > 0
    local = session.lookup(prompt, max_tokens=4)
    assert local is not None
    assert local.token_ids == (20, 21, 22, 23)
    # An in-flight request remains invisible to other requests.
    assert cache.lookup(prompt, max_tokens=4) is None

    published = session.close(publish=True)
    assert published > 0
    assert session.closed
    global_candidate = cache.lookup(prompt, max_tokens=4)
    assert global_candidate is not None
    assert global_candidate.token_ids == (20, 21, 22, 23)
    with pytest.raises(RuntimeError):
        session.append_verified((24,))
    with pytest.raises(RuntimeError):
        session.lookup(prompt, max_tokens=4)


def test_causal_session_chunking_matches_bulk_closed_request_index() -> None:
    prompt = (1, 2, 3, 4)
    completion = (5, 6, 7, 8, 9, 10)
    bulk = _cache(max_entries=1_000, max_continuation_length=4)
    causal = _cache(max_entries=1_000, max_continuation_length=4)
    bulk_records = bulk.observe_sequence(
        prompt_tokens=prompt,
        verified_completion_tokens=completion,
    )
    session = causal.begin_causal_session(prompt_tokens=prompt)
    session.append_verified(completion[:2])
    session.append_verified(completion[2:5])
    session.append_verified(completion[5:])
    causal_records = session.close(publish=True)

    assert causal_records == bulk_records
    assert causal.stats() == bulk.stats()
    for prefix_length in range(len(completion)):
        context = prompt + completion[:prefix_length]
        assert causal.lookup(context, max_tokens=4) == bulk.lookup(
            context,
            max_tokens=4,
        )


def test_discarded_causal_session_never_updates_global_cache() -> None:
    cache = _cache(max_continuation_length=2)
    session = cache.begin_causal_session(prompt_tokens=(1, 2, 3))
    session.append_verified((4, 5, 6))
    assert session.local_records > 0
    assert session.close(publish=False) == 0
    assert cache.stats().entries == 0


def test_cache_uses_frequency_confidence_and_longest_match() -> None:
    cache = _cache(min_confidence=0.6)
    prompt = (1, 2, 3, 4)
    cache.observe_sequence(prompt_tokens=prompt, verified_completion_tokens=(5, 6, 7))
    cache.observe_sequence(prompt_tokens=prompt, verified_completion_tokens=(5, 6, 7))
    cache.observe_sequence(prompt_tokens=prompt, verified_completion_tokens=(8, 9, 10))

    candidate = cache.lookup(prompt, max_tokens=3)
    assert candidate is not None
    assert candidate.token_ids == (5, 6, 7)
    assert candidate.observations == 2
    assert candidate.total_observations == 3
    assert candidate.confidence == pytest.approx(2 / 3)


def test_cache_bounds_entries_and_alternatives_deterministically() -> None:
    cache = _cache(max_entries=3, max_alternatives_per_key=4)
    prompt = (1, 2, 3, 4)
    cache.observe_sequence(prompt_tokens=prompt, verified_completion_tokens=(5, 6))
    cache.observe_sequence(prompt_tokens=prompt, verified_completion_tokens=(7, 8))
    stats = cache.stats()
    assert stats.entries <= 3
    assert stats.evicted_entries > 0

    alternatives = _cache(max_entries=100, max_alternatives_per_key=1)
    alternatives.observe_sequence(
        prompt_tokens=prompt,
        verified_completion_tokens=(5, 6),
    )
    alternatives.observe_sequence(
        prompt_tokens=prompt,
        verified_completion_tokens=(7, 8),
    )
    alternative_stats = alternatives.stats()
    assert alternative_stats.alternatives <= alternative_stats.entries
    assert alternative_stats.evicted_alternatives > 0


def test_delta_cache_proposal_preserves_ar_one_token_distribution() -> None:
    target = np.array([0.05, 0.15, 0.5, 0.3])
    for proposal in range(target.size):
        draft = delta_draft_probabilities(
            [proposal],
            vocabulary_size=target.size,
        )[0]
        actual = one_token_output_distribution(target, draft)
        np.testing.assert_allclose(actual, target, atol=1e-14, rtol=0.0)


def test_greedy_replay_commits_first_target_mismatch_or_lookahead() -> None:
    rejected = verify_greedy_replay(
        (1, 2, 3, 4),
        (1, 2, 9, 8),
        lookahead_token=7,
    )
    assert rejected.accepted_count == 2
    assert rejected.committed_tokens == (1, 2, 9)
    assert rejected.rejection_index == 2
    assert rejected.correction_token == 9
    assert rejected.lookahead_token is None

    accepted = verify_greedy_replay(
        (1, 2, 3),
        (1, 2, 3),
        lookahead_token=4,
    )
    assert accepted.all_accepted
    assert accepted.committed_tokens == (1, 2, 3, 4)


def test_cost_router_explores_then_falls_back_and_periodically_probes() -> None:
    router = CostAwareReplayRouter(
        namespace="model@revision|greedy",
        config=ReplayRouteConfig(
            min_match_length=4,
            min_proposal_tokens=2,
            exploration_trials_per_match_length=1,
            probe_interval=2,
            ema_decay=0.0,
            throughput_margin=0.0,
        )
    )
    candidate = ReplayCandidate(
        token_ids=(1, 2, 3),
        matched_suffix_length=4,
        observations=1,
        total_observations=1,
        confidence=1.0,
        namespace="model@revision|greedy",
    )
    assert not router.decide(candidate).use_replay
    router.observe_static(committed_tokens=3, forwards=2)
    assert router.decide(candidate).reason == "explore"
    router.observe_replay(
        matched_suffix_length=4,
        committed_tokens=1,
        forwards=1,
    )
    assert not router.decide(candidate).use_replay
    probe = router.decide(candidate)
    assert probe.use_replay
    assert probe.reason == "periodic-probe"


def test_cost_router_exploits_replay_above_static_tpf() -> None:
    router = CostAwareReplayRouter(
        namespace="model@revision|greedy",
        config=ReplayRouteConfig(throughput_margin=0.1),
    )
    router.observe_static(committed_tokens=3, forwards=2)
    candidate = ReplayCandidate(
        token_ids=(1, 2, 3),
        matched_suffix_length=8,
        observations=2,
        total_observations=2,
        confidence=1.0,
        namespace="model@revision|greedy",
    )
    assert router.decide(candidate).use_replay
    router.observe_replay(
        matched_suffix_length=8,
        committed_tokens=4,
        forwards=1,
    )
    decision = router.decide(candidate)
    assert decision.use_replay
    assert decision.reason == "exploit"


def test_cost_router_can_share_failure_evidence_across_match_lengths() -> None:
    router = CostAwareReplayRouter(
        namespace="model@revision|greedy",
        config=ReplayRouteConfig(
            min_match_length=8,
            min_proposal_tokens=1,
            exploration_trials_per_match_length=1,
            probe_interval=100,
            ema_decay=0.0,
            throughput_margin=0.0,
            match_length_bucket_width=32,
        ),
    )
    router.observe_static(committed_tokens=4, forwards=2)
    first = ReplayCandidate(
        token_ids=(1, 2, 3),
        matched_suffix_length=8,
        observations=1,
        total_observations=1,
        confidence=1.0,
        namespace="model@revision|greedy",
    )
    second = ReplayCandidate(
        token_ids=(4, 5, 6),
        matched_suffix_length=10,
        observations=1,
        total_observations=1,
        confidence=1.0,
        namespace="model@revision|greedy",
    )
    explored = router.decide(first)
    assert explored.use_replay
    assert explored.match_length_bucket == 8
    router.observe_replay(
        matched_suffix_length=8,
        committed_tokens=1,
        forwards=1,
    )
    shared = router.decide(second)
    assert not shared.use_replay
    assert shared.reason == "below-margin"
    assert shared.match_length_bucket == 8


@pytest.mark.parametrize(
    ("probability", "proposals", "expected"),
    ((0.0, 7, 1.0), (0.5, 1, 1.5), (1.0, 7, 8.0)),
)
def test_independent_match_tpf(
    probability: float,
    proposals: int,
    expected: float,
) -> None:
    assert independent_match_tpf(
        token_match_probability=probability,
        proposals=proposals,
    ) == pytest.approx(expected)


def test_replay_configuration_and_inputs_are_strict() -> None:
    with pytest.raises(ValueError):
        VerifierReplayCache(namespace="", config=ReplayCacheConfig())
    with pytest.raises(ValueError):
        ReplayCacheConfig(min_suffix_length=0).validate()
    with pytest.raises(ValueError):
        delta_draft_probabilities([4], vocabulary_size=4)
    with pytest.raises(ValueError):
        verify_greedy_replay((1, 2), (1,), lookahead_token=3)

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
import pytest
import torch

from online_speculation.recycling import (
    RecyclingConfig, RecyclingController, tail_after_commit, verify_target_draws,
)


@pytest.mark.parametrize("reject", [0, 1, 2, 3])
def test_target_draw_verifier_returns_exact_prefix_and_correction(reject: int) -> None:
    proposals = torch.tensor([4, 5, 6])
    draws = torch.tensor([4, 5, 6, 7])
    if reject < 3:
        draws[reject] = 8
    result = verify_target_draws(proposals, draws)
    assert result.committed == tuple(draws[:reject + 1].tolist())
    assert result.accepted_spec_tokens == reject
    assert result.used_lookahead == (reject == 3)


def test_refill_and_recycle_tail_have_different_alignment() -> None:
    predictions = torch.tensor([10, 11, 12, 13])
    assert tail_after_commit(
        predictions, committed_tokens=2, refill=True, max_candidates=3,
    ).tolist() == [11, 12, 13]
    assert tail_after_commit(
        predictions, committed_tokens=2, refill=False, max_candidates=3,
    ).tolist() == [12, 13]
    assert tail_after_commit(
        predictions, committed_tokens=5, refill=True, max_candidates=3,
    ).numel() == 0
    with pytest.raises(ValueError):
        tail_after_commit(
            predictions, committed_tokens=5, refill=False, max_candidates=3,
        )


def test_history_dependent_recycled_multitoken_law_matches_ar_exactly() -> None:
    """Enumerate fresh verifier draws, stale tails, and changing block lengths."""
    vocabulary = (0, 1, 2)

    def target(history: tuple[int, ...]) -> np.ndarray:
        # Depend on the whole history, so stale conditional rows differ.
        index = (sum(history) + len(history)) % 3
        return np.roll(np.array([0.15, 0.30, 0.55]), index)

    @lru_cache(None)
    def law(history: tuple[int, ...], candidates: tuple[int, ...], left: int):
        if left == 0:
            return {(): 1.0}
        if not candidates:
            candidates = (0, 1)  # Arbitrary refill, not an oracle continuation.
        distributions = [target(history + candidates[:i]) for i in range(len(candidates)+1)]
        predictions = torch.tensor([int(p.argmax()) for p in distributions])
        answer: dict[tuple[int, ...], float] = {}
        for draws in product(vocabulary, repeat=len(distributions)):
            mass = float(np.prod([p[x] for p, x in zip(distributions, draws)]))
            result = verify_target_draws(torch.tensor(candidates), torch.tensor(draws))
            committed = result.committed[:left]
            tail = tuple(tail_after_commit(
                predictions, committed_tokens=len(committed),
                refill=False, max_candidates=2,
            ).tolist())
            for suffix, conditional in law(
                history + committed, tail, left - len(committed),
            ).items():
                key = committed + suffix
                answer[key] = answer.get(key, 0.0) + mass * conditional
        return answer

    actual = law((2,), (1, 0), 3)
    assert sum(actual.values()) == pytest.approx(1.0, abs=1e-12)
    for sequence in product(vocabulary, repeat=3):
        expected = np.prod([
            target((2,) + sequence[:i])[token]
            for i, token in enumerate(sequence)
        ])
        assert actual[sequence] == pytest.approx(expected, abs=1e-12)


def test_controller_uses_tokens_over_time_instead_of_tpf() -> None:
    controller = RecyclingController(RecyclingConfig(
        policy="tps", exploration_trials=1, ema_decay=0, throughput_margin=0.05,
    ))
    controller.observe(recycle=False, candidates=0, tokens=4, seconds=0.02)
    assert controller.decide(candidates=3, depth=0)[1] == "explore"
    # More tokens per forward, but only 150 TPS versus refill 200 TPS.
    controller.observe(recycle=True, candidates=3, tokens=3, seconds=0.02)
    assert controller.decide(candidates=3, depth=0)[1] == "below-tps-margin"
    controller.observe(recycle=True, candidates=3, tokens=3, seconds=0.01)
    assert controller.decide(candidates=3, depth=0)[1] == "tps-exploit"
    assert controller.decide(candidates=3, depth=4)[1] == "depth-refill"


def test_throughput_estimate_is_ratio_of_separate_emas() -> None:
    controller = RecyclingController(RecyclingConfig(ema_decay=0.5))
    controller.observe(recycle=False, candidates=0, tokens=2, seconds=1.0)
    controller.observe(recycle=False, candidates=0, tokens=8, seconds=2.0)
    assert controller.refill.tps == pytest.approx(5 / 1.5)
    assert controller.refill.tps != pytest.approx((2 / 1 + 8 / 2) / 2)
    with pytest.raises(ValueError):
        controller.observe(recycle=True, candidates=2, tokens=2, seconds=float("nan"))

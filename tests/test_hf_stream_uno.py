from __future__ import annotations

import pytest

from online_speculation.hf_stream_uno import break_even_requests, choose_snapshot


def test_snapshot_selection_uses_validation_threshold_and_earliest_tie() -> None:
    selected = choose_snapshot(
        {0: 1.0, 1: 1.001, 2: 1.01, 3: 1.01},
        minimum_gain=0.002,
    )
    assert selected["best_validation_snapshot"] == 2
    assert selected["selected_snapshot"] == 2
    assert selected["nonzero_snapshot_selected"]

    fallback = choose_snapshot(
        {0: 1.0, 1: 1.001},
        minimum_gain=0.002,
    )
    assert fallback["selected_snapshot"] == 0
    assert not fallback["nonzero_snapshot_selected"]


def test_snapshot_selection_rejects_missing_zero_anchor() -> None:
    with pytest.raises(ValueError, match="zero snapshot"):
        choose_snapshot({1: 1.1}, minimum_gain=0.0)


def test_break_even_requires_positive_future_saving() -> None:
    assert break_even_requests(2.1, 0.5) == 5
    assert break_even_requests(-1.0, 0.5) == 0
    assert break_even_requests(1.0, 0.0) is None
    assert break_even_requests(1.0, -0.1) is None

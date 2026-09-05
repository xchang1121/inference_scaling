from __future__ import annotations

import copy
import json
from pathlib import Path

from online_speculation.stage8_analysis import analyze


RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "archive"
    / "2026-09-05-v1"
    / "results"
    / "stage8_greedy_stream_uno1b_rtx3090_hf.json"
)


def _benchmark() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_stage8_analysis_reproduces_preregistered_greedy_result() -> None:
    result = analyze(
        _benchmark(),
        bootstrap_samples=2_000,
        bootstrap_seed=20262405,
    )
    assert result["integrity"]["safety_gate_pass"]
    assert result["integrity"]["paired_output_token_ids_equal_pass"]
    assert result["integrity"]["all_greedy_target_token_ids_identical_pass"]
    assert result["decision"]["nonzero_selection_gate_pass"]
    assert result["decision"]["future_request_learning_statistical_gate_pass"]
    assert result["decision"]["future_request_learning_practical_gate_pass"]
    assert result["decision"]["greedy_online_learning_success"]
    assert not result["decision"]["frozen_serving_system_gate_pass"]
    assert not result["decision"]["all_stage8_gates_pass"]


def test_stage8_exactness_audit_fails_on_changed_online_token() -> None:
    benchmark = copy.deepcopy(_benchmark())
    token_ids = benchmark["test_runs"][0]["persistent_frozen"]["metrics"][
        "output_token_ids"
    ]
    token_ids[0] = (int(token_ids[0]) + 1) % 100
    result = analyze(
        benchmark,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )
    assert not result["integrity"]["paired_output_token_ids_equal_pass"]
    assert not result["integrity"]["all_greedy_target_token_ids_identical_pass"]
    assert not result["integrity"]["safety_gate_pass"]
    assert not result["decision"]["greedy_online_learning_success"]

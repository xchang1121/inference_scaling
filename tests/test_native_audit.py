from __future__ import annotations

from pathlib import Path
import json
import runpy

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "analyze_native_uno.py"))
validate, summarize = MODULE["validate"], MODULE["summarize"]


def fixture():
    payload = {"completed": True, "stage": "complete", "error": None,
               "parameters_frozen": True, "parameters_frozen_after": True,
               "environment": {"tracked_source_clean": True}, "engine_stats": {"preemptions": 0},
               "design": {"methods": [1, 8, "online"], "workloads": [("one", "prompt")],
                          "seed": 100, "repetitions": 1, "max_new_tokens": 8}, "records": []}
    for method in (1, 8, "online"):
        row = {"block_size": method, "workload": "one", "seed": 100, "end_to_end_seconds": 0.1,
               "output_tokens": 8, "e2e_tps": 80.0,
               "output": {"token_ids": list(range(8)), "stats": {"accepts": 7, "forwards": 4}},
               "cuda_graph_hits": 4, "cuda_graph_misses": 0, "gpu_after": "snapshot"}
        if method == "online":
            row["online"] = {"policy": {"widths": [8], "pending": None, "optimizer_steps": 0,
                                        "model_weight_updates": 0, "completed_epochs": 1,
                                        "epoch_cycles": 2, "completed_epochs_by_width": {"8": 1}},
                             "additional_cuda_synchronizations": 0,
                             "instrumented_choice_update_seconds": 0.00001,
                             "cycles": [{"width": 8, "tokens": 3, "reason": "initial_probe"},
                                        {"width": 8, "tokens": 4, "reason": "initial_probe"}]}
        payload["records"].append(row)
    return payload


def test_decode_stats_exclude_prefill_and_online_feedback_reconciles():
    data = fixture()
    assert validate(data) == 3
    result = summarize(data)
    assert result["methods"]["online"]["official_tpf_decode_only"] == 7 / 4
    assert result["methods"]["online"]["ar_exact_matches"] == 1
    assert result["online_over_fixed"]["8"]["aggregate_tps_ratio"] == 1


def test_missing_arm_cannot_be_ignored():
    data = fixture()
    data["records"].pop()
    with pytest.raises(RuntimeError, match="matrix"):
        validate(data)


def test_behavior_divergence_is_reported_not_silently_failed_or_removed():
    data = fixture()
    data["records"][-1]["output"]["token_ids"][3] = 42
    result = summarize(data)
    assert result["methods"]["online"]["ar_exact_matches"] == 0
    assert result["methods"]["online"]["ar_mismatches"][0]["first_difference"] == 3
    assert result["valid_runs"] == 3


def test_fake_prefill_stat_or_online_update_count_is_rejected():
    data = fixture()
    data["records"][-1]["output"]["stats"]["accepts"] = 8
    with pytest.raises(RuntimeError, match="budget"):
        validate(data)
    data = fixture()
    data["records"][-1]["online"]["policy"]["completed_epochs_by_width"]["8"] = 99
    with pytest.raises(RuntimeError, match="epoch"):
        validate(data)


def test_nan_timing_is_rejected():
    data = fixture()
    data["records"][0]["end_to_end_seconds"] = float("nan")
    with pytest.raises(RuntimeError, match="timing"):
        validate(data)


@pytest.mark.parametrize("filename,expected", [
    ("stage12_official_fa2_baseline.json", 32),
    ("stage12_native_online_r7_pilot.json", 60),
    ("stage12_native_shadow8.json", 24),
])
def test_real_completed_native_studies_preserve_the_full_matrix(filename, expected):
    path = Path(__file__).resolve().parents[1] / "results" / filename
    assert validate(json.loads(path.read_text(encoding="utf-8"))) == expected


def test_gpu_shadow_preserves_same_width_outputs_and_official_stats():
    path = Path(__file__).resolve().parents[1] / "results" / "stage12_native_shadow8.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"]
    fixed = {(r["workload"], r["seed"]): r for r in rows if r["block_size"] == 8}
    for row in rows:
        if row["block_size"] == "shadow8":
            reference = fixed[(row["workload"], row["seed"])]
            assert row["output"] == reference["output"]
            assert {c["width"] for c in row["online"]["cycles"]} == {8}

from __future__ import annotations

from pathlib import Path
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


def test_shadow_audit_uses_synthetic_same_width_control():
    data = fixture()
    data["design"]["methods"][-1] = "shadow8"
    data["records"][-1]["block_size"] = "shadow8"
    result = summarize(data)
    assert result["methods"]["shadow8"]["same_width_B8_token_matches"] == 1
    data["records"][-1]["output"]["token_ids"][3] = 42
    assert summarize(data)["methods"]["shadow8"]["same_width_B8_token_matches"] == 0


def test_plain_comparator_requires_real_graph_and_reports_scope():
    data = fixture()
    data["design"]["methods"][-1] = "plain8"
    data["records"][-1]["block_size"] = "plain8"
    data["records"][-1]["online"] = None
    with pytest.raises(RuntimeError, match="separately captured"):
        validate(data)
    data["plain_control_graphs"] = 3
    assert validate(data) == 3


def test_real_learner_audit_distinguishes_plain_and_zero_branch_baselines():
    from copy import deepcopy

    data = fixture()
    data["design"]["methods"] = [1, 8, "plain8", "fast8"]
    data["plain_control_graphs"] = 3
    data["records"][-1]["block_size"] = "plain8"
    data["records"][-1]["online"] = None
    online = deepcopy(data["records"][-1])
    online["block_size"] = "fast8"
    online["online"] = dict(algorithm="last_mlp_online_lora", teacher_weight_updates=0,
                             offline_uno_weight_updates=0, cycles=2, optimizer_steps=0,
                             model_weight_updates=0, events=[], update_seconds=0)
    data["records"].append(online)
    report = summarize(data)
    assert report["online_all_branch_costs_controlled"]
    assert report["methods"]["fast8"]["same_width_plain8_token_matches"] == 1
    assert report["online_over_fixed"]["plain8"]["aggregate_tps_ratio"] == 1
    data["records"][-2]["output"]["token_ids"][3] = 42
    assert summarize(data)["methods"]["fast8"]["same_width_plain8_token_matches"] == 0


def test_order_audit_rejects_an_unbalanced_pair():
    from copy import deepcopy

    data = fixture()
    data["design"]["repetitions"] = 2
    data["design"]["order_pairing"] = "reverse_adjacent_repetitions"
    for row in data["records"]:
        row["order"] = [1, 8, "online"]
    second = deepcopy(data["records"])
    for row in second:
        row["seed"] += 1
        row["order"].reverse()
    data["records"].extend(second)
    assert summarize(data)["order_pairing_complete"]
    for row in data["records"][3:]:
        row["order"].reverse()
    with pytest.raises(RuntimeError, match="counterbalance"):
        validate(data)

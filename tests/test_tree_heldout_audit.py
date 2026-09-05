from __future__ import annotations

import json
import hashlib
from pathlib import Path
import runpy

import pytest


PROJECT = Path(__file__).resolve().parents[1]
validate_study = runpy.run_path(str(PROJECT / "scripts" / "analyze_tree_heldout.py"))["validate_study"]


def _study():
    return json.loads((PROJECT / "results" / "stage11_tree_heldout_fp32.json").read_text(encoding="utf-8"))


def test_frozen_completed_study_has_all_360_runs():
    assert validate_study(_study()) == 360


def test_whole_missing_method_is_not_reinterpreted_as_smaller_design():
    study = _study()
    study["records"] = [r for r in study["records"] if r["method"] != "tree:8:32"]
    with pytest.raises(RuntimeError, match="matrix"):
        validate_study(study)


def test_duplicate_pair_cannot_replace_missing_run():
    study = _study()
    study["records"][-1] = study["records"][0]
    with pytest.raises(RuntimeError, match="matrix"):
        validate_study(study)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_time_cannot_enter_tps_denominator(value):
    study = _study()
    study["records"][0]["metrics"]["end_to_end_seconds"] = value
    with pytest.raises(RuntimeError, match="timing"):
        validate_study(study)


def test_incomplete_output_budget_fails_even_if_ids_and_count_agree():
    study = _study()
    metric = study["records"][0]["metrics"]
    metric["output_token_ids"].pop()
    metric["output_tokens"] -= 1
    with pytest.raises(RuntimeError, match="output budget"):
        validate_study(study)


def test_feedback_pilot_is_complete_and_all_outputs_equal_actual_ar_records():
    study = json.loads((PROJECT / "results" / "stage11_tree_feedback_fp32_pilot.json").read_text(encoding="utf-8"))
    assert study["scope"] == "pilot"
    assert validate_study(study) == 72
    ar = {(r["workload"], r["seed"]): r["metrics"]["output_token_ids"] for r in study["records"] if r["method"] == "ar"}
    for row in study["records"]:
        assert row["metrics"]["output_token_ids"] == ar[(row["workload"], row["seed"])]
        if row["method"] == "treefeedback:8:32":
            diagnostic = row["diagnostics"]
            controller = diagnostic["budget_controller"]
            assert diagnostic["model_parameters_frozen"]
            assert diagnostic["optimizer_steps"] == 0
            assert not controller["pending_feedback"]
            for n, updates in controller["reward_updates"].items():
                assert updates == sum(count for action, count in controller["counts"].items() if int(action) >= int(n))


def test_completed_raw_studies_match_committed_audit_digests():
    for raw_name, audit_name in (
        ("stage11_tree_heldout_fp32.json", "stage11_tree_heldout_audit.json"),
        ("stage11_tree_feedback_fp32_pilot.json", "stage11_tree_feedback_audit.json"),
    ):
        raw = (PROJECT / "results" / raw_name).read_bytes()
        audit = json.loads((PROJECT / "results" / audit_name).read_text(encoding="utf-8"))
        assert hashlib.sha256(raw).hexdigest() == audit["raw_sha256"]

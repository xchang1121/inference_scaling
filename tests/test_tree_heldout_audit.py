from __future__ import annotations

import json
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

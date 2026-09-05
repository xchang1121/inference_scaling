from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("native_numerics", ROOT / "scripts" / "diagnose_native_numerics.py")
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)
sys.path.pop(0)


def test_frozen_contexts_recovered_without_future_token_leak():
    payload = json.loads((ROOT / "results" / "stage12_official_fa2_baseline.json").read_text(encoding="utf-8"))
    contexts = MODULE.select_contexts(payload)
    assert tuple((x["workload"], x["generation_index"]) for x in contexts) == MODULE.EXPECTED_CONTEXTS
    assert all(len(x["ar_prefix"]) == x["generation_index"] for x in contexts)


def test_matrix_cardinality_and_comparison_count():
    assert len(MODULE.expected_keys()) == 96
    probe = {"logits": torch.arange(8.), "hidden": torch.arange(4.),
             "seed_kv": torch.arange(16.), "fp32": torch.arange(8.)}
    comparisons = MODULE.compare_probes(dict.fromkeys(MODULE.expected_keys(), probe))
    assert {name: len(items) for name, items in comparisons.items()} == {
        "width": 18, "future": 24, "repeat": 48, "graph_eager": 12, "mask": 16}
    assert all(pair["native_logits"]["exact"] for rows in comparisons.values() for pair in rows)


def test_argmax_ties_and_softmax_shift_are_distinct():
    logits = torch.tensor([2., 2., 0., -1., -2.])
    summary = MODULE.logit_summary(logits)
    assert summary["argmax"] == 0
    assert summary["margin"] == 0
    assert summary["max_ties"] == 2
    compared = MODULE.logit_distance(logits, logits + 10)
    assert compared["softmax_tv_temperature1"] == 0
    assert compared["perturbation_range"] == 0
    assert not compared["exact"]


def test_argmax_margin_sufficient_not_necessary_and_invalid_tensors():
    reference = torch.tensor([5., 2., 1., 0., -1.])
    assert MODULE.logit_distance(reference, reference + .25)["reference_margin_gt_2_max_abs"]
    tie = torch.tensor([2., 2., 1., 0., -1.])
    result = MODULE.logit_distance(tie, tie + torch.tensor([0., .01, 0., 0., 0.]))
    assert not result["argmax_equal"] and not result["reference_margin_gt_2_max_abs"]
    assert 0 < result["softmax_tv_temperature1"] < .01
    with pytest.raises(ValueError, match="nonfinite"):
        MODULE.distance(reference, reference * float("nan"))
    with pytest.raises(ValueError, match="shapes"):
        MODULE.distance(reference, reference[:1])


def test_incomplete_diagnostics_cannot_be_reported_as_complete():
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.validate({"completed": False})


def test_official_graph_capture_omits_artificial_b1_lora_masks():
    assert sum(MODULE.expected_graph_hit(k) for k in MODULE.expected_keys()) == 40
    assert MODULE.expected_graph_hit("graph/B1/off/f0/r0") == 1
    assert MODULE.expected_graph_hit("graph/B1/zero/f0/r0") == 0
    assert MODULE.expected_graph_hit("graph/B1/noise/f1/r1") == 0
    assert MODULE.expected_graph_hit("graph/B8/zero/f0/r0") == 1
    assert MODULE.expected_graph_hit("eager/B8/off/f0/r0") == 0

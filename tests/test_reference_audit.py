"""Oracle source gates are tested without importing any external model code."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def module():
    spec = importlib.util.spec_from_file_location("base_audit", Path(__file__).resolve().parents[1] /
                                                 "scripts" / "audit_hf_reference.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_error_summary_and_position_weighted_aggregation():
    audit = module()
    zeros = torch.zeros(2, 7)
    a = audit.error_summary(zeros, zeros)
    b = audit.error_summary(torch.ones(1, 7), torch.zeros(1, 7))
    result = audit.summarize([a, b])
    assert result["positions"] == 3 and result["mean_abs"] == pytest.approx(1 / 3)
    assert result["max_abs"] == 1 and result["argmax_mismatches"] == result["mean_tv"] == 0
    with pytest.raises(ValueError, match="invalid"):
        audit.error_summary(zeros, torch.full_like(zeros, float("nan")))


def test_reference_gate_rejects_modified_code_weights_and_index(tmp_path, monkeypatch):
    audit = module()
    source = b"raise RuntimeError('NEVER IMPORT THIS TEST SOURCE')\n"
    (tmp_path / "modeling.py").write_bytes(source.replace(b"\n", b"\r\n"))
    (tmp_path / "weights.safetensors").write_bytes(b"placeholder")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"models": {"k2_1b_base": {
        "reference_lf_sha256": {"modeling.py": hashlib.sha256(source).hexdigest()},
        "weight_filename": "weights.safetensors", "weight_sha256": hashlib.sha256(b"placeholder").hexdigest(),
    }}}))
    monkeypatch.setattr(audit, "LOCK", lock)
    assert audit.checked_reference(tmp_path)["weight_filename"] == "weights.safetensors"
    index.write_text(json.dumps({"weight_map": {"x": "unchecked.safetensors"}}))
    with pytest.raises(ValueError, match="index"):
        audit.checked_reference(tmp_path)
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights differ"):
        audit.checked_reference(tmp_path)
    (tmp_path / "modeling.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source/config differs"):
        audit.checked_reference(tmp_path)

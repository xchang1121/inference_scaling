from __future__ import annotations

from pathlib import Path

import pytest

from inference_scaling.dllm.backends.loader import load_sdar_backend


def _config(tmp_path: Path):
    return {
        "model": {"path": str(tmp_path / "base")},
        "proposal": {"kind": "prefix_layers", "layers": 8},
        "alignment": {"adapter": str(tmp_path / "adapter")},
        "runtime": {"device": "cpu", "dtype": "float32"},
    }


def test_base_and_reduced_layer_roles_pass_distinct_loader_arguments(monkeypatch, tmp_path):
    calls = []

    def fake_loader(cls, path, **kwargs):
        calls.append((path, kwargs))
        return object()

    monkeypatch.setattr(
        "inference_scaling.dllm.backends.loader.SDARTransformersBackend.from_pretrained",
        classmethod(fake_loader),
    )
    config = _config(tmp_path)

    load_sdar_backend(config, "base")
    load_sdar_backend(config, "proposal")

    assert calls[0][1] == {"device": "cpu", "dtype": "float32"}
    assert calls[1][1]["num_hidden_layers"] == 8
    assert calls[1][1]["max_window_layers"] == 8
    assert calls[1][1]["model_id"].endswith("#prefix-layers=8")


def test_aligned_role_requires_a_completed_adapter(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the VRPO stage first"):
        load_sdar_backend(_config(tmp_path), "aligned")

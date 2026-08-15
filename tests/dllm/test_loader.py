from __future__ import annotations

from pathlib import Path

import pytest

from inference_scaling.dllm.backends.loader import load_llada_backend


def _config(tmp_path: Path):
    return {
        "model": {"path": str(tmp_path / "base")},
        "proposal": {"kind": "shared_prefix_layers", "layers": 8},
        "alignment": {"adapter": str(tmp_path / "adapter")},
        "runtime": {"device": "cpu", "dtype": "float32", "max_batch_size": 3},
    }


def test_base_and_proposal_roles_share_the_loaded_model(monkeypatch, tmp_path):
    calls = []

    class FakeBackend:
        def with_prefix_layers(self, layers):
            calls.append(("prefix", layers))
            return "proposal"

    base = FakeBackend()

    def fake_loader(cls, path, **kwargs):
        calls.append((path, kwargs))
        return base

    monkeypatch.setattr(
        "inference_scaling.dllm.backends.loader.LLaDATransformersBackend.from_pretrained",
        classmethod(fake_loader),
    )
    config = _config(tmp_path)

    loaded = load_llada_backend(config, "base")
    proposal = load_llada_backend(config, "proposal", base_backend=loaded)

    assert calls[0][1] == {
        "device": "cpu",
        "dtype": "float32",
        "mask_token_id": 156895,
        "max_batch_size": 3,
    }
    assert calls[1] == ("prefix", 8)
    assert proposal == "proposal"


def test_aligned_role_requires_a_completed_adapter(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the VRPO stage first"):
        load_llada_backend(_config(tmp_path), "aligned")

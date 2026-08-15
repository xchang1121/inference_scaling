from __future__ import annotations

import hashlib

import pytest

from experiments.dllm import download_llada


def test_checkpoint_validation_uses_size_and_official_digest(monkeypatch, tmp_path):
    content = b"pinned-llada-shard"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        download_llada,
        "WEIGHTS",
        (("model-00001-of-00001.safetensors", len(content), digest),),
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(content)

    result = download_llada.validate_checkpoint(tmp_path)

    assert result["weights"]["model-00001-of-00001.safetensors"] == {
        "bytes": len(content),
        "sha256": digest,
    }


def test_checkpoint_validation_rejects_corruption(monkeypatch, tmp_path):
    content = b"corrupted"
    monkeypatch.setattr(
        download_llada,
        "WEIGHTS",
        (("model.safetensors", len(content), "0" * 64),),
    )
    (tmp_path / "model.safetensors").write_bytes(content)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_llada.validate_checkpoint(tmp_path)

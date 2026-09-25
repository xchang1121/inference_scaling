from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_scaling.app.settings import load_settings
from inference_scaling.dllm.backends.loader import load_llada_backend


def _sections(tmp_path: Path):
    dllm = copy.deepcopy(load_settings()["dllm"])
    dllm["model"]["path"] = str(tmp_path / "base")
    dllm["engine"].update(device="cpu", dtype="float32", max_batch_size=3)
    return dllm["model"], dllm["engine"]


def test_base_model_receives_the_engine_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "inference_scaling.dllm.backends.loader.LLaDATransformersBackend.from_pretrained",
        classmethod(lambda cls, path, **kwargs: calls.append((path, kwargs)) or "base"),
    )
    model, engine = _sections(tmp_path)

    assert load_llada_backend(model, engine) == "base"
    assert calls == [(str(tmp_path / "base"), {
        "device": "cpu", "dtype": "float32", "mask_token_id": model["mask_token_id"], "max_batch_size": 3,
        "trust_remote_code": True, "attn_implementation": "sdpa",
    })]


def test_a_configured_adapter_must_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "inference_scaling.dllm.backends.loader.LLaDATransformersBackend.from_pretrained",
        classmethod(lambda cls, path, **kwargs: SimpleNamespace()),
    )
    model, engine = _sections(tmp_path)
    model["adapter"] = {"path": str(tmp_path / "missing")}
    with pytest.raises(FileNotFoundError, match="train it first"):
        load_llada_backend(model, engine)


def test_the_adapter_wraps_the_base_model_and_keeps_the_batch_cap(monkeypatch, tmp_path):
    model, engine = _sections(tmp_path)
    model["adapter"] = {"path": str(tmp_path / "adapter")}
    Path(model["adapter"]["path"]).mkdir()
    constructed = []
    base = SimpleNamespace(model=object(), tokenizer=object(), mask_token_id=17)

    class FakeBackend:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return base

        def __init__(self, model, tokenizer, **kwargs):
            constructed.append((model, tokenizer, kwargs))

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter):
            return SimpleNamespace(merge_and_unload=lambda: SimpleNamespace(eval=lambda: "aligned-model"))

    monkeypatch.setattr("inference_scaling.dllm.backends.loader.LLaDATransformersBackend", FakeBackend)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))

    load_llada_backend(model, engine)
    assert constructed == [("aligned-model", base.tokenizer, {
        "model_id": model["adapter"]["path"], "mask_token_id": 17, "max_batch_size": 3,
    })]

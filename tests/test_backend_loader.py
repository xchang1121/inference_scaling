from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import pytest

from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends import loader


def _sections(backend: str = "transformers"):
    ar = copy.deepcopy(load_settings()["ar"])
    ar["engine"]["backend"] = backend
    ar["model"]["path"] = "local-model"
    return ar["model"], ar["engine"]


def test_transformers_loader_passes_every_configured_option(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained",
                        lambda model, **kwargs: captured.update(model=model, **kwargs) or "transformers")
    model, engine = _sections()
    engine["transformers"]["model_kwargs"] = {"low_cpu_mem_usage": True}

    assert loader.load_backend(model, engine, seed=3, logprobs=0) == "transformers"
    assert captured == {
        "model": "local-model",
        "adapter_name_or_path": None,
        "adapter_revision": None,
        "revision": model["revision"],
        "tokenizer_name_or_path": None,
        "tokenizer_revision": None,
        "tokenizer_kwargs": {},
        "local_files_only": True,
        "trust_remote_code": False,
        "cache_dir": None,
        "device": engine["device"],
        "dtype": engine["dtype"],
        "device_map": None,
        "attn_implementation": "sdpa",
        "model_kwargs": {"low_cpu_mem_usage": True},
        "max_score_batch_size": engine["transformers"]["max_score_batch_size"],
        "score_chunk_size": engine["transformers"]["score_chunk_size"],
    }


def test_adapter_loads_on_top_of_the_base_model(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained",
                        lambda model, **kwargs: captured.update(model=model, **kwargs))
    model, engine = _sections()
    model["adapter"] = {"path": "grpo-adapter", "revision": "adapter-commit"}

    loader.load_backend(model, engine, seed=3, logprobs=0)
    assert captured["model"] == "local-model"
    assert captured["adapter_name_or_path"] == "grpo-adapter"
    assert captured["adapter_revision"] == "adapter-commit"


def test_async_vllm_uses_its_public_signature_and_shares_files_with_the_exact_scorer(monkeypatch, tmp_path) -> None:
    calls = []
    original = loader.AsyncVLLMBackend.from_pretrained
    monkeypatch.setattr(loader.TransformersBackend, "from_pretrained",
                        lambda model, **kwargs: calls.append(("scorer", model, kwargs)) or SimpleNamespace())

    def engine_loader(model, **kwargs):
        assert set(kwargs) <= set(inspect.signature(original).parameters)
        calls.append(("engine", model, kwargs))
        return "engine"

    monkeypatch.setattr(loader.AsyncVLLMBackend, "from_pretrained", engine_loader)
    model, engine = _sections("vllm")
    model["path"] = str(tmp_path)
    engine["vllm"]["exact_scoring"] = "transformers"

    assert loader.load_backend(model, engine, seed=5, logprobs=16) == "engine"
    (_, scorer_model, _), (_, engine_model, kwargs) = calls
    assert scorer_model == engine_model == str(tmp_path)
    assert kwargs["seed"] == 5
    assert kwargs["engine_kwargs"] == {**engine["vllm"]["engine_kwargs"], "max_logprobs": 16}
    assert "enable_mh_fused_logprobs" not in kwargs


def test_fused_mh_logprobs_need_the_synchronous_engine(monkeypatch, tmp_path) -> None:
    model, engine = _sections("vllm")
    model["path"] = str(tmp_path)
    engine["vllm"]["mh_fused_logprobs"] = True
    with pytest.raises(ValueError, match="synchronous"):
        loader.load_backend(model, engine, seed=0, logprobs=0)

    captured = {}
    monkeypatch.setattr(loader.VLLMBackend, "from_pretrained", lambda model, **kwargs: captured.update(kwargs))
    engine["vllm"]["asynchronous"] = False
    loader.load_backend(model, engine, seed=0, logprobs=0)
    assert captured["enable_mh_fused_logprobs"] is True
    assert "max_logprobs" not in captured["engine_kwargs"]

"""Construct the configured autoregressive backend from the ``ar`` settings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.backends.vllm_backend import AsyncVLLMBackend, VLLMBackend
from inference_scaling.shared.model.loading import resolve_checkpoint_path


def _identity(model: Mapping[str, Any]) -> dict[str, Any]:
    """Loading options that every engine shares."""

    adapter = model["adapter"]
    return {
        "adapter_name_or_path": None if adapter is None else str(adapter["path"]),
        "adapter_revision": None if adapter is None else adapter["revision"],
        "revision": model["revision"],
        "tokenizer_name_or_path": model["tokenizer"],
        "tokenizer_revision": model["tokenizer_revision"],
        "tokenizer_kwargs": dict(model["tokenizer_kwargs"]),
        "local_files_only": bool(model["local_files_only"]),
        "trust_remote_code": bool(model["trust_remote_code"]),
        "token_penalty": model["token_penalty"],
    }


def _transformers(path: str, identity: Mapping[str, Any], model: Mapping[str, Any],
                  engine: Mapping[str, Any]) -> TransformersBackend:
    options = engine["transformers"]
    return TransformersBackend.from_pretrained(
        path,
        **identity,
        cache_dir=model["cache_dir"],
        device=str(engine["device"]),
        dtype=str(engine["dtype"]),
        device_map=options["device_map"],
        attn_implementation=options["attn_implementation"],
        model_kwargs=dict(options["model_kwargs"]),
        max_score_batch_size=int(options["max_score_batch_size"]),
        score_chunk_size=int(options["score_chunk_size"]),
        prefix_cache_mib=int(options["prefix_cache_mib"]),
        in_place_kv=bool(options["in_place_kv"]),
    )


def load_backend(model: Mapping[str, Any], engine: Mapping[str, Any], *, seed: int, logprobs: int) -> Any:
    """Load ``ar.model`` on ``ar.engine``.

    ``logprobs`` is the largest number of per-token log-probabilities the
    algorithm requests (beam search); vLLM's limit is raised to cover it. With
    ``vllm.exact_scoring = "transformers"`` a Transformers copy of the model
    scores sequences exactly.
    """

    identity = _identity(model)
    if engine["backend"] == "transformers":
        return _transformers(str(model["path"]), identity, model, engine)
    vllm = engine["vllm"]
    if vllm["mh_fused_logprobs"] and vllm["asynchronous"]:
        raise ValueError("vllm.mh_fused_logprobs needs the synchronous engine (vllm.asynchronous = false)")
    if vllm["mh_fused_logprobs"] and model["token_penalty"] is not None:
        raise ValueError("vllm.mh_fused_logprobs reads the unpenalized logits; disable it or ar.model.token_penalty")
    # Resolve Hub names once so the engine and the exact scorer read the same files.
    path = str(model["path"])
    if not Path(path).is_dir():
        path = str(resolve_checkpoint_path(path, revision=model["revision"], cache_dir=model["cache_dir"],
                                           local_files_only=identity["local_files_only"]))
    adapter = identity["adapter_name_or_path"]
    if adapter is not None and not Path(adapter).is_dir():
        identity["adapter_name_or_path"] = str(resolve_checkpoint_path(
            adapter, revision=identity["adapter_revision"], cache_dir=model["cache_dir"],
            local_files_only=identity["local_files_only"],
        ))
    engine_kwargs = dict(vllm["engine_kwargs"])
    if logprobs > int(engine_kwargs.get("max_logprobs", 0)) and logprobs > 0:
        engine_kwargs["max_logprobs"] = logprobs
    scorer = _transformers(path, identity, model, engine) if vllm["exact_scoring"] == "transformers" else None
    options: dict[str, Any] = {}
    if not vllm["asynchronous"]:
        options["enable_mh_fused_logprobs"] = bool(vllm["mh_fused_logprobs"])
    try:
        return (AsyncVLLMBackend if vllm["asynchronous"] else VLLMBackend).from_pretrained(
            path,
            **identity,
            download_dir=model["cache_dir"],
            dtype=str(engine["dtype"]),
            tensor_parallel_size=int(vllm["tensor_parallel_size"]),
            data_parallel_size=int(vllm["data_parallel_size"]),
            gpu_memory_utilization=float(vllm["gpu_memory_utilization"]),
            max_model_len=vllm["max_model_len"],
            max_num_seqs=vllm["max_num_seqs"],
            max_num_batched_tokens=vllm["max_num_batched_tokens"],
            quantization=vllm["quantization"],
            enforce_eager=bool(vllm["enforce_eager"]),
            enable_prefix_caching=bool(vllm["enable_prefix_caching"]),
            max_lora_rank=int(vllm["max_lora_rank"]),
            parameter_count=vllm["parameter_count"],
            seed=seed,
            scoring_backend=scorer,
            engine_kwargs=engine_kwargs,
            **options,
        )
    except BaseException:
        close_backend(scorer)
        raise


def close_backend(backend: Any | None) -> None:
    """Close vLLM processes/threads when present; safe for other backends."""

    if backend is None:
        return
    try:
        callback = getattr(backend, "close", None)
        if callback is not None:
            callback()
    finally:
        scoring_backend = getattr(backend, "scoring_backend", None)
        if scoring_backend is not None and scoring_backend is not backend:
            close_backend(scoring_backend)


__all__ = ["close_backend", "load_backend"]

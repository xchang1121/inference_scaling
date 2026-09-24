"""Construct the configured LLaDA backend from the ``dllm`` settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from inference_scaling.dllm.backends.llada import LLaDATransformersBackend


def load_llada_backend(model: Mapping[str, Any], engine: Mapping[str, Any]) -> LLaDATransformersBackend:
    """Load ``dllm.model`` on ``dllm.engine``; a configured adapter (e.g. VRPO) is applied on top."""

    options: dict[str, Any] = {
        "device": str(engine["device"]),
        "dtype": str(engine["dtype"]),
        "mask_token_id": int(model["mask_token_id"]),
        "max_batch_size": int(engine["max_batch_size"]),
        "trust_remote_code": bool(model["trust_remote_code"]),
    }
    if engine["attn_implementation"] is not None:
        options["attn_implementation"] = str(engine["attn_implementation"])
    base = LLaDATransformersBackend.from_pretrained(str(model["path"]), **options)
    if model["adapter"] is None:
        return base
    adapter = Path(str(model["adapter"]["path"]))
    if not adapter.is_dir():
        raise FileNotFoundError(f"the LLaDA adapter {adapter} is absent; train it first")
    from peft import PeftModel

    return LLaDATransformersBackend(
        PeftModel.from_pretrained(base.model, adapter).eval(),
        base.tokenizer,
        model_id=str(adapter),
        mask_token_id=base.mask_token_id,
        max_batch_size=int(engine["max_batch_size"]),
    )


__all__ = ["load_llada_backend"]

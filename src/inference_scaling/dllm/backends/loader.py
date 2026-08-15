"""Role-aware SDAR backend construction for paired experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from inference_scaling.dllm.backends.sdar import SDARTransformersBackend

SDARRole = Literal["base", "proposal", "aligned"]


def load_sdar_backend(
    config: Mapping[str, Any],
    role: SDARRole,
) -> SDARTransformersBackend:
    """Load the full, reduced-layer proposal, or aligned SDAR checkpoint."""

    model = config["model"]
    runtime = config["runtime"]
    base_path = str(model["path"])
    common = {
        "device": str(runtime.get("device", "cuda")),
        "dtype": str(runtime.get("dtype", "bfloat16")),
    }
    if role == "base":
        return SDARTransformersBackend.from_pretrained(base_path, **common)

    if role == "proposal":
        proposal = config["proposal"]
        if str(proposal.get("kind")) != "prefix_layers":
            raise ValueError("only the frozen prefix-layer SDAR proposal is supported")
        layers = int(proposal["layers"])
        if layers <= 0:
            raise ValueError("proposal.layers must be positive")
        return SDARTransformersBackend.from_pretrained(
            base_path,
            model_id=f"{base_path}#prefix-layers={layers}",
            num_hidden_layers=layers,
            max_window_layers=layers,
            **common,
        )

    if role != "aligned":
        raise ValueError(f"unknown SDAR role {role!r}")
    adapter_path = Path(str(config["alignment"]["adapter"]))
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"aligned SDAR adapter is absent: {adapter_path}; run the VRPO stage first"
        )
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("loading the aligned SDAR adapter requires PEFT") from exc
    base = SDARTransformersBackend.from_pretrained(base_path, **common)
    aligned_model = PeftModel.from_pretrained(base.model, adapter_path).eval()
    return SDARTransformersBackend(
        aligned_model,
        base.tokenizer,
        model_id=str(adapter_path),
        mask_token_id=base.mask_token_id,
    )


__all__ = ["SDARRole", "load_sdar_backend"]

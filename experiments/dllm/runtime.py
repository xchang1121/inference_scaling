"""Shared validation and device checks for real-model dLLM experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_device(config: dict[str, Any]) -> str:
    device = str(config["runtime"]["device"])
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def validate_llada_weights(config: dict[str, Any]) -> dict[str, str]:
    model_dir = Path(str(config["model"]["path"]))
    names = tuple(str(value) for value in config["model"]["weight_files"])
    hashes = tuple(str(value) for value in config["model"]["weight_sha256"])
    sizes = tuple(int(value) for value in config["model"]["weight_bytes"])
    if not (len(names) == len(hashes) == len(sizes)):
        raise ValueError("LLaDA weight manifest columns have different lengths")
    actual: dict[str, str] = {}
    for name, expected_hash, expected_size in zip(names, hashes, sizes, strict=True):
        weight = model_dir / name
        if not weight.is_file():
            raise FileNotFoundError(
                f"pinned LLaDA weight is absent: {weight}; "
                "run experiments/dllm/download_llada.py"
            )
        if weight.stat().st_size != expected_size:
            raise ValueError(f"LLaDA weight size does not match the manifest: {weight}")
        actual_hash = file_sha256(weight)
        if actual_hash != expected_hash:
            raise ValueError(f"LLaDA weight hash does not match the manifest: {weight}")
        actual[name] = actual_hash
    return actual


__all__ = ["file_sha256", "validate_llada_weights", "validate_runtime_device"]


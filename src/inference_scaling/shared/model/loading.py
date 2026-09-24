"""Checkpoint resolution and model identity, independent of algorithms and datasets."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_checkpoint_path(
    name_or_path: str, *, revision: str | None = None,
    cache_dir: str | None = None, local_files_only: bool = True,
) -> Path:
    """Resolve a local checkpoint or a Hub snapshot; offline by default.

    This function reads/downloads files only when explicitly called by a runtime,
    never during configuration parsing or CLI dry runs.
    """
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or name_or_path.startswith(("./", "../", ".\\", "..\\")):
        raise FileNotFoundError(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        name_or_path, revision=revision, cache_dir=cache_dir,
        local_files_only=local_files_only,
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.gguf", "*.ot", "original/*"],
    ))


def model_identity(model: str, adapter: str | None = None, *, revision: str | None = None,
                   adapter_revision: str | None = None, tokenizer: str | None = None,
                   tokenizer_revision: str | None = None) -> str:
    """Use the same identity for generation, rescoring, and replay caches."""
    value = model if adapter is None else f"{model}+adapter:{adapter}"
    for name, setting in (("revision", revision), ("adapter_revision", adapter_revision),
                          ("tokenizer", tokenizer), ("tokenizer_revision", tokenizer_revision)):
        if setting is not None:
            value += f";{name}={setting}"
    return value


def checkpoint_weight_files(path: Path) -> tuple[Path, ...]:
    """Resolve the standard safetensors/PyTorch weight layout without loading it."""
    directory = path.resolve()
    for single, index in (("model.safetensors", "model.safetensors.index.json"),
                          ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        if (directory / single).is_file():
            return (directory / single,)
        if not (directory / index).is_file():
            continue
        value = json.loads((directory / index).read_text(encoding="utf-8"))
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid checkpoint weight_map: {directory / index}")
        if any(not isinstance(name, str) for name in mapping.values()):
            raise ValueError("checkpoint shard names must be strings")
        files = []
        for name in sorted(set(mapping.values())):
            shard = Path(name)
            # Validate names, not resolved symlinks: Hub snapshots link to blobs.
            if shard.is_absolute() or shard.drive or ".." in shard.parts or "\\" in name:
                raise ValueError(f"checkpoint shard must be relative to its directory: {name}")
            target = directory / name
            if not target.is_file():
                raise FileNotFoundError(target)
            files.append(target)
        return tuple(files)
    raise FileNotFoundError(f"no supported checkpoint weights or shard index in {directory}")

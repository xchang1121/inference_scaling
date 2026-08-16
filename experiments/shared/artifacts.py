"""Stable hashing and dataclass accounting helpers for experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Collection


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataclass_snapshot_delta(
    before: Any,
    after: Any,
    *,
    constant_fields: Collection[str] = (),
) -> dict[str, Any]:
    left = asdict(before)
    right = asdict(after)
    if left.keys() != right.keys():
        raise ValueError("snapshot schemas do not match")
    constants = set(constant_fields)
    unknown = constants - right.keys()
    if unknown:
        raise ValueError(f"constant fields are absent from the snapshot: {sorted(unknown)}")
    return {
        key: right[key] if key in constants else right[key] - left[key]
        for key in right
    }


__all__ = ["dataclass_snapshot_delta", "file_sha256", "json_fingerprint"]

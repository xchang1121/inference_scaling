"""Download the pinned SDAR checkpoint and verify the official weight digest."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "JetLM/SDAR-1.7B-Chat"
REVISION = "14fcc4b796f05b9a11aa565b2353199af757d079"
WEIGHT_NAME = "model.safetensors"
WEIGHT_BYTES = 4_063_515_640
WEIGHT_SHA256 = "1737775176591d7c7f39b884b98d620d87646f8220b9b6b39431b6f6467e3e0f"
DEFAULT_OUTPUT = Path("models/SDAR-1.7B-Chat-14fcc4b7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(directory: Path) -> dict[str, str | int]:
    weight = directory / WEIGHT_NAME
    if not weight.is_file():
        raise FileNotFoundError(f"missing SDAR weight: {weight}")
    size = weight.stat().st_size
    if size != WEIGHT_BYTES:
        raise RuntimeError(f"weight size mismatch: expected {WEIGHT_BYTES}, got {size}")
    digest = sha256(weight)
    if digest != WEIGHT_SHA256:
        raise RuntimeError(
            f"weight SHA-256 mismatch: expected {WEIGHT_SHA256}, got {digest}"
        )
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_bytes": size,
        "weight_sha256": digest,
        "path": str(directory.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--endpoint",
        help="Optional Hugging Face-compatible endpoint; the official SHA is always checked.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download code/tokenizer/config but omit the 4.06 GB weight.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not access the network; validate an existing local checkpoint.",
    )
    args = parser.parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    if not args.validate_only:
        patterns = [
            "*.json",
            "*.jinja",
            "*.py",
            "*.txt",
        ]
        if not args.metadata_only:
            patterns.append(WEIGHT_NAME)
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_dir=args.output,
            allow_patterns=patterns,
        )
    if args.metadata_only:
        print(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "path": str(args.output.resolve()),
                "weights": "not requested",
            }
        )
    else:
        print(validate_checkpoint(args.output))


if __name__ == "__main__":
    main()

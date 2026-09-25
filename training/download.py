"""Fetch the pinned GSM8K splits and model snapshots, then verify their hashes."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from inference_scaling.app.records import cached_file_sha256
from inference_scaling.datasets.gsm8k import GSM8K


def _verified(model: Mapping[str, Any], cache_dir: Path) -> bool:
    """Whether every pinned weight file is present with its hash; unpinned models only need the directory."""

    directory = Path(str(model["path"]))
    if model["weight_sha256"] is None:
        return directory.is_dir()
    try:
        for name, digest in model["weight_sha256"].items():
            cached_file_sha256(directory / name, cache_dir=cache_dir, expected=str(digest))
    except (FileNotFoundError, ValueError):
        return False
    return True


def _fetch(model: Mapping[str, Any], download: Mapping[str, Any]) -> None:
    from huggingface_hub import snapshot_download

    options = download["huggingface"]
    if options["endpoint"] is not None:
        os.environ["HF_ENDPOINT"] = str(options["endpoint"])
    snapshot_download(str(model["repository"]), revision=str(model["revision"]), local_dir=str(model["path"]),
                      allow_patterns=model["allow_patterns"], max_workers=int(options["max_workers"]))


def run(settings: Mapping[str, Any]) -> None:
    for split in ("train", "test"):
        dataset = GSM8K(settings["gsm8k"][split])
        print(f"verified GSM8K {split}: {dataset.settings['path']} ({len(dataset.problems)} problems)", flush=True)
    download, cache_dir = settings["download"], Path(str(settings["hash_cache_dir"]))
    for model in download["models"]:
        attempts = int(download["retries"])
        for attempt in range(1, attempts + 1):
            if _verified(model, cache_dir):
                print(f"verified {model['path']}", flush=True)
                break
            if attempt > 1:
                time.sleep(float(download["retry_wait_seconds"]))
            print(f"downloading {model['repository']}@{model['revision']} (attempt {attempt})", flush=True)
            try:
                _fetch(model, download)
            except Exception as error:  # network failures are retried; the final check reports them
                print(f"download failed: {type(error).__name__}: {error}", flush=True)
        else:
            if not _verified(model, cache_dir):
                raise RuntimeError(f"{model['path']} could not be downloaded with the pinned hashes")


__all__ = ["run"]

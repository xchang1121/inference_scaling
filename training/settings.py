"""The fixed training settings file and its strict schema.

``settings/training.json`` (relative to the working directory) holds every
training parameter; ``stages`` lists what ``python -m training`` runs, in
order. Field meanings are documented in ``docs/SETTINGS.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_scaling.app.settings import SCHEMA as INFERENCE_SCHEMA
from inference_scaling.app.settings import SettingsError, check

SETTINGS_PATH = Path("settings/training.json")
STAGES = ("download", "grpo", "vrpo_preferences", "vrpo")

_NULLABLE_STR = (str, None)
_VERIFIER = {key: value for key, value in INFERENCE_SCHEMA["rewards"]["verifier"].items() if key != "temperature"}
_LORA = {"r": int, "lora_alpha": int, "lora_dropout": float, "bias": str, "target_modules": [str]}

SCHEMA: dict[str, Any] = {
    "stages": [frozenset(STAGES)],
    "hash_cache_dir": str,
    "gsm8k": {"train": INFERENCE_SCHEMA["datasets"]["gsm8k"], "test": INFERENCE_SCHEMA["datasets"]["gsm8k"]},
    "download": {
        "retries": int,
        "retry_wait_seconds": float,
        "huggingface": {"endpoint": _NULLABLE_STR, "max_workers": int},
        "models": [{
            "path": str,
            "repository": str,
            "revision": str,
            "allow_patterns": ([str], None),
            "weight_sha256": (dict, None),
        }],
    },
    "grpo": {
        "model": {
            "path": str,
            "revision": _NULLABLE_STR,
            "weight_sha256": _NULLABLE_STR,
            "tokenizer": _NULLABLE_STR,
            "tokenizer_revision": _NULLABLE_STR,
            "tokenizer_kwargs": dict,
            "cache_dir": _NULLABLE_STR,
            "local_files_only": bool,
            "trust_remote_code": bool,
            "model_kwargs": dict,
        },
        "output": str,
        "resume": bool,
        "lora": _LORA,
        # Passed to trl.GRPOConfig unchanged; TRL validates the names.
        "trainer": dict,
        "verifier": _VERIFIER,
        "power_sample_seconds": float,
    },
    "vrpo": {
        "model": {key: value for key, value in INFERENCE_SCHEMA["dllm"]["model"].items()
                  if key not in {"revision", "adapter"}},
        "engine": INFERENCE_SCHEMA["dllm"]["engine"],
        "prompt": INFERENCE_SCHEMA["dllm"]["prompt"],
        "sampling": INFERENCE_SCHEMA["dllm"]["sampling"],
        "max_new_tokens": int,
        "preferences": {
            "data": str,
            "manifest": str,
            "selection": {"count": int, "seed": int},
            "pairs": int,
            "num_generations": int,
            "include_reference_completion": bool,
            "seed": int,
        },
        "verifier": _VERIFIER,
        "training": {
            "output": str,
            "resume": bool,
            "max_steps": int,
            "gradient_accumulation_steps": int,
            "timestep_samples": int,
            "masks_per_timestep": int,
            "antithetic": bool,
            "learning_rate": float,
            "beta": float,
            "max_grad_norm": float,
            "save_steps": int,
            "seed": int,
            "gradient_checkpointing": bool,
        },
        "lora": _LORA,
    },
}


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise SettingsError(f"{path} does not exist; run from the repository root")
    settings = json.loads(path.read_text(encoding="utf-8"))
    check(settings, SCHEMA, "training")
    return settings


__all__ = ["SCHEMA", "SETTINGS_PATH", "STAGES", "load_settings"]

"""The fixed settings file and its strict schema.

``settings/inference.json`` (relative to the working directory) holds every
behavior parameter; the command line only chooses the algorithm, model family,
reward and dataset. A missing key, an unknown key or a value of the wrong type
is an error. Field meanings are documented in ``docs/SETTINGS.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path("settings/inference.json")

# A schema node is a dict (exact keys), a one-element list (array of that
# node), a tuple (alternatives), a frozenset (allowed strings), ``None``, a
# scalar type, or ``dict`` for a free-form object passed through unchanged.
_Choices = frozenset
_NULLABLE_INT = (int, None)
_NULLABLE_STR = (str, None)
# Weight file name -> SHA-256, enforced when given.
_PINS = (dict, None)

_DLLM_SAMPLING = {
    "block_length": int,
    "steps_per_block": int,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "remasking": _Choices({"low_confidence", "random"}),
}

SCHEMA: dict[str, Any] = {
    "run": {"seed": int, "draws": int, "hash_cache_dir": str},
    "datasets": {
        "gsm8k": {
            "path": str,
            "download": bool,
            "source": {"url": str, "sha256": str, "rows": int},
            "selection": {"count": _NULLABLE_INT, "seed": int},
            "prompt_template": str,
        },
        "math500": {
            "path": str,
            "download": bool,
            "source": {"repository": str, "revision": str, "filename": str},
            "selection": {"seed": int, "minimum_level": int, "excluded_ids": [str], "skip": int, "count": int},
            "prompt_template": str,
            "judge_timeout_seconds": float,
        },
    },
    "rewards": {
        "verifier": {"temperature": float, "source": _Choices({"dataset", "vote"}), "pool_size": int},
        "self_certainty": {"temperature": float, "score_temperature": float, "scope": _Choices({"thinking", "full"})},
        "consilience": {
            "temperature": float,
            "score_temperature": float,
            "scope": _Choices({"thinking", "full"}),
            "top_k": int,
            "window_fraction": float,
            "window_tokens": _NULLABLE_INT,
            "skip_fraction": float,
            "initial_penalty": float,
        },
    },
    "ar": {
        "max_new_tokens": int,
        "model": {
            "path": str,
            "revision": _NULLABLE_STR,
            "weight_sha256": _PINS,
            "adapter": (None, {"path": str, "revision": _NULLABLE_STR}),
            "tokenizer": _NULLABLE_STR,
            "tokenizer_revision": _NULLABLE_STR,
            "tokenizer_kwargs": dict,
            "cache_dir": _NULLABLE_STR,
            "local_files_only": bool,
            "trust_remote_code": bool,
            "token_penalty": (None, {"words": [str], "strength": float}),
        },
        "engine": {
            "backend": _Choices({"transformers", "vllm"}),
            "device": str,
            "dtype": str,
            "context_window": _NULLABLE_INT,
            "transformers": {
                "attn_implementation": _NULLABLE_STR,
                "device_map": (str, dict, None),
                "model_kwargs": dict,
                "max_score_batch_size": int,
                "score_chunk_size": int,
                "prefix_cache_mib": int,
                "in_place_kv": bool,
                "cuda_graphs": bool,
            },
            "vllm": {
                "asynchronous": bool,
                "async_scheduling": (bool, None),
                "stream_interval": int,
                "flat_logprobs": bool,
                "tensor_parallel_size": int,
                "data_parallel_size": int,
                "gpu_memory_utilization": float,
                "max_num_seqs": _NULLABLE_INT,
                "max_num_batched_tokens": _NULLABLE_INT,
                "quantization": _NULLABLE_STR,
                "enforce_eager": bool,
                "enable_prefix_caching": bool,
                "max_lora_rank": int,
                "fused_logprobs": bool,
                "exact_scoring": _Choices({"none", "transformers"}),
                "parameter_count": _NULLABLE_INT,
                "engine_kwargs": dict,
            },
            "continuous_batching": {
                "workers": int,
                "max_batch_size": int,
                "max_batch_tokens": int,
                "batch_wait_seconds": float,
            },
            "rollout_scheduler": {
                "enabled": bool,
                "max_in_flight": int,
                "max_prefill_tokens": int,
                "kv_high_watermark": float,
                "aging_seconds": float,
                "tail_prefill_tokens": int,
                "policy": _Choices({"fair_lpt", "fifo"}),
            },
        },
        "prompt": {"system": _NULLABLE_STR, "format": _Choices({"auto", "chat", "plain"}), "chat_template_kwargs": dict},
        "output": {
            "thinking_mode": _Choices({"auto", "enabled", "disabled"}),
            "thinking_start_text": _NULLABLE_STR,
            "thinking_end_text": _NULLABLE_STR,
            "starts_in_thinking": (bool, None),
            "sampling_scope": _Choices({"full", "thinking"}),
        },
        "sampling": {"temperature": float, "top_p": float, "top_k": _NULLABLE_INT},
        "algorithms": {
            "sample": {},
            "greedy": {},
            "beam": {"num_beams": int},
            "best_of_n": {"samples": int},
            "mh": {
                "ensemble_size": int,
                "execution": _Choices({"batch", "pipeline"}),
                "initialization": _Choices({"iid", "stratified"}),
                "stratified_updates": bool,
                "nonself_updates": bool,
                "block_size": int,
                "steps_per_block": int,
                "iterations": _NULLABLE_INT,
                "suffix_schedule": _Choices({"uniform", "inverse_length", "multiscale", "log_uniform"}),
                "proposal": _Choices({"base", "frozen_history"}),
                "frozen_history": {"samples": int, "mixture": float},
                "suffix_replay": bool,
            },
            "mh_power": {
                "alpha": float,
                "proposal_temperature": float,
                "block_size": int,
                "steps_per_block": int,
                "iterations": _NULLABLE_INT,
                "suffix_schedule": _Choices({"uniform", "inverse_length", "multiscale", "log_uniform"}),
                "suffix_replay": bool,
                "early_rejection": bool,
            },
            "is": {
                "candidate_count": int,
                "rollout_count": int,
                "block_size": int,
                "block_first": bool,
                "sibling_replay": bool,
                "completion_fraction": float,
                "rollout_race": bool,
                "exploration_schedule": _Choices({"fixed", "two_stage", "single_pool"}),
                "last_rollout_filter": bool,
                "weighted_completion_quota": bool,
                "selection_ready": bool,
                "metropolized_blocks": bool,
            },
        },
    },
    "serve": {
        "server": {"host": str, "port": int, "served_model_name": str, "max_concurrent_requests": int,
                   "keepalive_seconds": float},
        "max_seconds": (float, None),
        "effort_aliases": dict,
    },
    "dllm": {
        "model": {
            "path": str,
            "revision": _NULLABLE_STR,
            "weight_sha256": _PINS,
            "mask_token_id": int,
            "trust_remote_code": bool,
            "adapter": (None, {"path": str}),
        },
        "engine": {"device": str, "dtype": str, "attn_implementation": _NULLABLE_STR, "max_batch_size": int,
                   "block_logits_only": bool},
        "prompt": {"system": _NULLABLE_STR},
        "max_new_tokens": int,
        "sampling": _DLLM_SAMPLING,
        "exact_sampling": _DLLM_SAMPLING,
        "algorithms": {
            "sample": {},
            "greedy": {},
            "beam": {"decision_block_size": int, "width": int, "branching_factor": int},
            "best_of_n": {"samples": int},
            "mh": {
                "updates": int,
                "proposal": _Choices({"base", "frozen_history"}),
                "frozen_history": {"samples": int, "mixture": float},
            },
            "mh_power": {"alpha": float, "decision_block_size": int, "updates_per_stage": int},
            "is": {
                "candidate_count": int,
                "rollout_count": int,
                "decision_block_size": int,
                "candidate_canvas": _Choices({"block", "full"}),
                "kept_sequence": bool,
            },
        },
    },
}


class SettingsError(ValueError):
    """The settings file does not match the schema."""


def _describe(spec: Any) -> str:
    if spec is None:
        return "null"
    if isinstance(spec, dict):
        return "an object"
    if isinstance(spec, list):
        return "an array"
    if isinstance(spec, frozenset):
        return "one of " + ", ".join(sorted(spec))
    return {bool: "a boolean", int: "an integer", float: "a number", str: "a string", dict: "an object"}[spec]


def check(value: Any, spec: Any, path: str) -> None:
    """Raise :class:`SettingsError` unless ``value`` matches the schema node ``spec``."""

    if isinstance(spec, tuple):
        if value is None and None in spec:
            return
        options = [option for option in spec if option is not None]
        if len(options) == 1:
            check(value, options[0], path)
            return
        for option in options:
            try:
                check(value, option, path)
                return
            except SettingsError:
                continue
        raise SettingsError(f"{path} must be " + " or ".join(_describe(option) for option in spec))
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            raise SettingsError(f"{path} must be an object")
        missing, unknown = sorted(spec.keys() - value.keys()), sorted(value.keys() - spec.keys())
        if missing or unknown:
            raise SettingsError(f"{path}: missing keys {missing}, unknown keys {unknown}")
        for key, child in spec.items():
            check(value[key], child, f"{path}.{key}")
        return
    if isinstance(spec, list):
        if not isinstance(value, list):
            raise SettingsError(f"{path} must be an array")
        for index, item in enumerate(value):
            check(item, spec[0], f"{path}[{index}]")
        return
    if isinstance(spec, frozenset):
        if value not in spec:
            raise SettingsError(f"{path} must be {_describe(spec)}, not {value!r}")
        return
    valid = (
        value is None if spec is None
        else isinstance(value, dict) if spec is dict
        else isinstance(value, (int, float)) and not isinstance(value, bool) if spec is float
        else isinstance(value, spec) and (spec is bool or not isinstance(value, bool))
    )
    if not valid:
        raise SettingsError(f"{path} must be {_describe(spec)}, not {value!r}")


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise SettingsError(f"{path} does not exist; run from the repository root")
    settings = json.loads(path.read_text(encoding="utf-8"))
    validate_settings(settings)
    return settings


def validate_settings(settings: dict[str, Any]) -> None:
    """Reject conflicting limits before any model is loaded, including benchmark overrides."""

    check(settings, SCHEMA, "settings")
    for family in ("ar", "dllm"):
        if settings[family]["max_new_tokens"] <= 0:
            raise SettingsError(f"{family}.max_new_tokens must be positive")
    ar = settings["ar"]
    context = ar["engine"]["context_window"]
    if context is not None and context <= 0:
        raise SettingsError("ar.engine.context_window must be positive or null")
    if context is not None and ar["max_new_tokens"] >= context:
        raise SettingsError("ar.engine.context_window must exceed ar.max_new_tokens to leave room for the prompt")
    if "max_model_len" in ar["engine"]["vllm"]["engine_kwargs"]:
        raise SettingsError("set ar.engine.context_window, not vllm.engine_kwargs.max_model_len")
    serve = settings["serve"]
    if serve["max_seconds"] is not None and serve["max_seconds"] <= 0:
        raise SettingsError("serve.max_seconds must be positive or null")
    if serve["server"]["max_concurrent_requests"] <= 0 or serve["server"]["keepalive_seconds"] <= 0:
        raise SettingsError("serve.server needs positive concurrency and keepalive")
    if any(not isinstance(name, str) or not name or not isinstance(target, str) or not target
           for name, target in serve["effort_aliases"].items()):
        raise SettingsError("serve.effort_aliases must map nonempty names to template reasoning efforts")


__all__ = ["SCHEMA", "SETTINGS_PATH", "SettingsError", "check", "load_settings", "validate_settings"]

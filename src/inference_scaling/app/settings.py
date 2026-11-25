"""The fixed settings file and its strict schema.

``settings/inference.json`` (relative to the working directory) holds every
behavior parameter; the command line only chooses the algorithm, model family,
reward and dataset. A missing required key, an unknown key or a value of the
wrong type is an error. Optional keys (marked with :class:`OptionalKey`) are
type-checked whenever present while keeping existing settings files compatible.
Field meanings are documented in ``docs/SETTINGS.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path("settings/inference.json")
SETTINGS_ENV_VAR = "INFERENCE_SCALING_SETTINGS"


class OptionalKey(str):
    """Dictionary schema key that may be omitted, but is strictly validated when present."""


# A schema node is a dict (exact required keys + optional OptionalKey entries),
# a one-element list (array of that node), a tuple (alternatives), a frozenset
# (allowed strings), ``None``, a scalar type, or ``dict`` for a free-form object.
_Choices = frozenset
_NULLABLE_INT = (int, None)
_NULLABLE_STR = (str, None)
# Weight file name -> SHA-256, enforced when given.
_PINS = (dict, None)
_SAMPLING_SPEC = {"temperature": float, "top_p": float, "top_k": _NULLABLE_INT}

SCHEMA: dict[str, Any] = {
    "run": {"reward": _Choices({"verifier", "self_certainty", "consilience"}),
            "seed": int, "draws": int, "hash_cache_dir": str},
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
        "self_certainty": {
            "temperature": float,
            "score_temperature": float,
            "scope": _Choices({"thinking", "full"}),
            OptionalKey("aggregation"): _Choices({"mean", "tail_mean", "final_window"}),
            OptionalKey("tail_fraction"): float,
            OptionalKey("window_fraction"): float,
            OptionalKey("final_weight"): float,
            OptionalKey("skip_fraction"): float,
            OptionalKey("incomplete_penalty"): float,
            OptionalKey("temperature_normalized"): bool,
        },
        "consilience": {
            "temperature": float,
            "score_temperature": float,
            "scope": _Choices({"thinking", "full"}),
            "top_k": int,
            "window_fraction": float,
            "window_tokens": _NULLABLE_INT,
            "skip_fraction": float,
            "initial_penalty": float,
            OptionalKey("incomplete_penalty"): float,
        },
    },
    "ar": {
        "max_new_tokens": _NULLABLE_INT,
        "thinking": (None, {
            "reasoning_effort": _NULLABLE_STR, "budget_tokens": _NULLABLE_INT,
            "default_effort": str, "default_budget_tokens": int,
            "effort_budgets": dict, "effort_aliases": dict,
        }),
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
            "backend": _Choices({"transformers", "vllm", "openai"}),
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
                "scalar_self_certainty": bool,
                "exact_scoring": _Choices({"none", "transformers"}),
                "parameter_count": _NULLABLE_INT,
                "engine_kwargs": dict,
            },
            OptionalKey("openai"): (None, {
                "base_url": str,
                OptionalKey("model"): _NULLABLE_STR,
                OptionalKey("api_key"): _NULLABLE_STR,
                OptionalKey("timeout_seconds"): float,
                OptionalKey("max_workers"): int,
                OptionalKey("parameter_count"): _NULLABLE_INT,
                OptionalKey("extra_body"): dict,
            }),
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
            OptionalKey("finish_sampling"): (None, _SAMPLING_SPEC),
        },
        "sampling": _SAMPLING_SPEC,
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
                "exploration_schedule": _Choices({"fixed", "two_stage", "adaptive"}),
                "selection_ready": bool,
                "selection": _Choices({"sir", "metropolized", "block_metropolized"}),
                OptionalKey("pilot_gate"): bool,
                OptionalKey("selective_lookahead"): bool,
                OptionalKey("straggler_pruning"): bool,
                OptionalKey("adaptive_temperature"): bool,
                OptionalKey("pilot_tokens"): _NULLABLE_INT,
            },
        },
    },
    "serve": {
        "server": {"host": str, "port": int, "served_model_name": str, "max_concurrent_requests": int,
                   "keepalive_seconds": float},
        "max_seconds": (float, None),
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
        required = {str(key) for key in spec if not isinstance(key, OptionalKey)}
        allowed = {str(key): child for key, child in spec.items()}
        missing = sorted(required - value.keys())
        unknown = sorted(value.keys() - allowed.keys())
        if missing or unknown:
            raise SettingsError(f"{path}: missing keys {missing}, unknown keys {unknown}")
        for key, child in allowed.items():
            if key in value:
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


def load_settings(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        override = os.environ.get(SETTINGS_ENV_VAR)
        path = Path(override) if override else SETTINGS_PATH
    if not path.is_file():
        raise SettingsError(f"{path} does not exist; run from the repository root or set {SETTINGS_ENV_VAR}")
    settings = json.loads(path.read_text(encoding="utf-8"))
    run = settings.get("run", {})
    if isinstance(run, dict):
        run.setdefault("reward", "self_certainty")
    # Older complete settings files used only SIR; preserve that explicit default.
    config = settings
    for key in ("ar", "algorithms", "is"):
        config = config.get(key, {}) if isinstance(config, dict) else {}
    if isinstance(config, dict):
        config.setdefault("selection", "sir")
    vllm = settings.get("ar", {}).get("engine", {}).get("vllm", {})
    if isinstance(vllm, dict):
        vllm.setdefault("scalar_self_certainty", False)
    validate_settings(settings)
    return settings


def validate_settings(settings: dict[str, Any]) -> None:
    """Reject conflicting limits before any model is loaded, including benchmark overrides."""

    check(settings, SCHEMA, "settings")
    ar = settings["ar"]
    is_cfg = ar["algorithms"]["is"]
    if is_cfg["selection_ready"] and is_cfg["selection"] != "sir":
        raise SettingsError("Metropolized selection requires selection_ready=false")
    if is_cfg["exploration_schedule"] == "adaptive":
        if is_cfg["selection_ready"]:
            raise SettingsError("adaptive exploration_schedule uses stagnation pruning; set selection_ready=false")
        if is_cfg["block_first"]:
            raise SettingsError("adaptive exploration_schedule requires block_first=false")
    if is_cfg.get("pilot_tokens") is not None and is_cfg["pilot_tokens"] <= 0:
        raise SettingsError("ar.algorithms.is.pilot_tokens must be positive or null")
    if ar["max_new_tokens"] is not None and ar["max_new_tokens"] <= 0:
        raise SettingsError("ar.max_new_tokens must be positive")
    context = ar["engine"]["context_window"]
    if context is not None and context <= 0:
        raise SettingsError("ar.engine.context_window must be positive or null")
    if "max_model_len" in ar["engine"]["vllm"]["engine_kwargs"]:
        raise SettingsError("set ar.engine.context_window, not vllm.engine_kwargs.max_model_len")
    serve = settings["serve"]
    if serve["max_seconds"] is not None and serve["max_seconds"] <= 0:
        raise SettingsError("serve.max_seconds must be positive or null")
    if serve["server"]["max_concurrent_requests"] <= 0 or serve["server"]["keepalive_seconds"] <= 0:
        raise SettingsError("serve.server needs positive concurrency and keepalive")
    profile = ar["thinking"]
    if profile is not None:
        limits = profile["effort_budgets"]
        if not limits or any(not isinstance(name, str) or not name or type(value) is not int or value <= 0
                             for name, value in limits.items()) or len(set(limits.values())) != len(limits):
            raise SettingsError("ar.thinking.effort_budgets needs distinct positive integer limits")
        if profile["default_effort"] not in limits or not 0 <= profile["default_budget_tokens"] <= max(limits.values()):
            raise SettingsError("ar.thinking defaults must use a supported effort and budget")
        if any(not isinstance(name, str) or not name or name in limits or
               target is not None and (not isinstance(target, str) or target not in limits)
               for name, target in profile["effort_aliases"].items()):
            raise SettingsError("ar.thinking.effort_aliases must reference supported efforts or null (disabled)")
        if {"reasoning_effort", "thinking_budget", "enable_thinking"} & ar["prompt"]["chat_template_kwargs"].keys():
            raise SettingsError("set effort/budget in ar.thinking and mode in ar.output, not chat_template_kwargs")
        if profile["reasoning_effort"] is not None and profile["budget_tokens"] is not None:
            raise SettingsError("ar.thinking.reasoning_effort and budget_tokens cannot be set together")
        from inference_scaling.shared.model.thinking import resolve_thinking
        try:
            resolve_thinking(ar)
        except ValueError as error:
            raise SettingsError(str(error)) from error


__all__ = ["OptionalKey", "SCHEMA", "SETTINGS_PATH", "SettingsError", "check", "load_settings", "validate_settings"]

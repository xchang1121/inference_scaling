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

_DLLM_SAMPLING = {
    "block_length": int,
    "steps_per_block": int,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "cfg_scale": float,
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
            "max_new_tokens": int,
        },
        "math500": {
            "path": str,
            "download": bool,
            "source": {"repository": str, "revision": str, "filename": str},
            "selection": {"seed": int, "minimum_level": int, "excluded_ids": [str], "skip": int, "count": int},
            "prompt_template": str,
            "judge_timeout_seconds": float,
            "max_new_tokens": int,
        },
    },
    "rewards": {
        "verifier": {
            "temperature": float,
            "source": _Choices({"dataset", "python", "constant"}),
            "dataset": {"correct": float, "incorrect": float, "unparseable": float},
            "python": {"factory": _NULLABLE_STR, "options": dict, "requires_reference": bool},
            "constant": {"value": float},
        },
        "vote": {"temperature": float, "pool_size": int},
        "logprob": {"temperature": float, "score_temperature": float},
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
        "model": {
            "path": str,
            "revision": _NULLABLE_STR,
            "weight_sha256": _NULLABLE_STR,
            "adapter": (None, {"path": str, "revision": _NULLABLE_STR}),
            "tokenizer": _NULLABLE_STR,
            "tokenizer_revision": _NULLABLE_STR,
            "tokenizer_kwargs": dict,
            "cache_dir": _NULLABLE_STR,
            "local_files_only": bool,
            "trust_remote_code": bool,
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
            },
            "vllm": {
                "asynchronous": bool,
                "tensor_parallel_size": int,
                "data_parallel_size": int,
                "gpu_memory_utilization": float,
                "max_model_len": _NULLABLE_INT,
                "max_num_seqs": _NULLABLE_INT,
                "max_num_batched_tokens": _NULLABLE_INT,
                "quantization": _NULLABLE_STR,
                "enforce_eager": bool,
                "enable_prefix_caching": bool,
                "max_lora_rank": int,
                "mh_fused_logprobs": bool,
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
                "block_size": int,
                "steps_per_block": int,
                "iterations": _NULLABLE_INT,
                "suffix_schedule": _Choices({"uniform", "inverse_length", "multiscale"}),
                "proposal": _Choices({"base", "frozen_history"}),
                "frozen_history": {"samples": int, "mixture": float},
            },
            "mh_power": {
                "alpha": float,
                "proposal_temperature": float,
                "block_size": int,
                "steps_per_block": int,
                "iterations": _NULLABLE_INT,
                "suffix_schedule": _Choices({"uniform", "inverse_length", "multiscale"}),
            },
            "is": {
                "planning": _Choices({"fixed", "full_horizon", "chunk_adaptive"}),
                "fixed": {"candidate_count": int, "rollout_count": int, "block_size": int},
                "joint": {
                    "forward_token_budget": int,
                    "block_sizes": [int],
                    "candidate_counts": [int],
                    "rollout_counts": [int],
                    "pilot_candidates": int,
                    "pilot_rollouts": int,
                    "pilot_fraction": float,
                    "relative_variance_floor": float,
                    "expected_output_tokens": _NULLABLE_INT,
                },
                "chunk_adaptive": {
                    "initial_block_size": int,
                    "initial_candidate_count": int,
                    "initial_rollout_count": int,
                    "adjustment_min_improvement": float,
                },
            },
        },
    },
    "dllm": {
        "model": {
            "path": str,
            "revision": _NULLABLE_STR,
            "weight_files": [str],
            "weight_bytes": [int],
            "weight_sha256": [str],
            "mask_token_id": int,
            "trust_remote_code": bool,
            "adapter": (None, {"path": str}),
        },
        "engine": {"device": str, "dtype": str, "attn_implementation": _NULLABLE_STR, "max_batch_size": int},
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
    check(settings, SCHEMA, "settings")
    return settings


__all__ = ["SCHEMA", "SETTINGS_PATH", "SettingsError", "check", "load_settings"]

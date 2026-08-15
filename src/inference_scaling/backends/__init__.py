"""Compatibility aliases for :mod:`inference_scaling.arllm.backends`."""

from __future__ import annotations

import importlib
import sys

_MODULES = (
    "absorbing",
    "batching",
    "cache",
    "candidate_cache",
    "loader",
    "tabular",
    "transformers_backend",
    "vllm_backend",
)

for _name in _MODULES:
    _module = importlib.import_module(f"inference_scaling.arllm.backends.{_name}")
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module

from inference_scaling.arllm.backends import *  # noqa: E402,F401,F403
from inference_scaling.arllm.backends import __all__  # noqa: E402,F401

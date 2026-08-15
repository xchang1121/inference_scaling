"""Compatibility aliases for :mod:`inference_scaling.arllm.algorithms`."""

from __future__ import annotations

import importlib
import sys

_MODULES = (
    "base_replay",
    "conditional_energy",
    "dynamic_is",
    "mh",
    "mh_acceleration",
    "progressive_is",
    "smc_forest",
    "streaming_is",
)

for _name in _MODULES:
    _module = importlib.import_module(f"inference_scaling.arllm.algorithms.{_name}")
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module

from inference_scaling.arllm.algorithms import *  # noqa: E402,F401,F403
from inference_scaling.arllm.algorithms import __all__  # noqa: E402,F401

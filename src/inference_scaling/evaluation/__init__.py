"""Compatibility aliases for :mod:`inference_scaling.shared.evaluation`."""

from __future__ import annotations

import importlib
import sys

_MODULES = ("consensus", "grpo_reward", "gsm8k")

for _name in _MODULES:
    _module = importlib.import_module(f"inference_scaling.shared.evaluation.{_name}")
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module

from inference_scaling.shared.evaluation import *  # noqa: E402,F401,F403
from inference_scaling.shared.evaluation import __all__  # noqa: E402,F401

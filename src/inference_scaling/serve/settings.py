"""``settings/serve.json``: the server and its reasoning-effort profiles, strictly checked.

An effort profile sets the chat template's reasoning effort, the output limit,
and the forward-token budget and wall-clock limit within which budgeted IS
chooses its block size, candidates and completions, as the offline runs do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_scaling.app.settings import SettingsError, check

SERVE_PATH = Path("settings/serve.json")

EFFORT = {
    "reasoning_effort": frozenset({"low", "medium", "xhigh"}),
    "max_new_tokens": int,
    "forward_token_budget": int,
    "max_seconds": float,
}
SCHEMA: dict[str, Any] = {
    "server": {"host": str, "port": int, "served_model_name": str, "max_concurrent_requests": int,
               "keepalive_seconds": float, "seed": int},
    "default_effort": str,
    "efforts": dict,
    "effort_aliases": dict,
}


def load_serve_settings(path: Path = SERVE_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise SettingsError(f"{path} does not exist; run from the repository root")
    settings = json.loads(path.read_text(encoding="utf-8"))
    check(settings, SCHEMA, "serve")
    efforts = settings["efforts"]
    for name, profile in efforts.items():
        check(profile, EFFORT, f"serve.efforts.{name}")
        if min(profile["max_new_tokens"], profile["forward_token_budget"]) <= 0 or profile["max_seconds"] <= 0:
            raise SettingsError(f"serve.efforts.{name} needs positive limits")
    for alias, target in settings["effort_aliases"].items():
        if target not in efforts:
            raise SettingsError(f"serve.effort_aliases.{alias} names no effort")
    if settings["default_effort"] not in efforts:
        raise SettingsError("serve.default_effort names no effort")
    if settings["server"]["max_concurrent_requests"] <= 0 or settings["server"]["keepalive_seconds"] <= 0:
        raise SettingsError("serve.server needs positive concurrency and keepalive")
    return settings


__all__ = ["EFFORT", "SCHEMA", "SERVE_PATH", "load_serve_settings"]

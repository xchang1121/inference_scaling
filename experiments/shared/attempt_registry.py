"""Validation for the machine-readable Qwen 1.5B optimization ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ATTEMPT_STATUSES = frozenset(
    {
        "planned",
        "screening",
        "accepted",
        "accepted_existing",
        "conditional",
        "rejected",
    }
)


def load_attempt_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate the optimization ledger used by reports and tests."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    if document.get("schema_version") != 1:
        raise ValueError("attempt registry requires schema_version=1")
    scope = document.get("scope")
    if not isinstance(scope, dict) or scope.get("model_family") != "arllm":
        raise ValueError("optimization experiments must be scoped to arllm")
    if scope.get("primary_model") != "Qwen2.5-1.5B-Instruct":
        raise ValueError("primary_model must be Qwen2.5-1.5B-Instruct")
    if scope.get("dllm_experiments") is not False:
        raise ValueError("dLLM experiments must be disabled in this study")

    attempts = document.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("attempt registry must contain at least one attempt")
    identifiers: set[str] = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("each attempt must be an object")
        identifier = attempt.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("each attempt requires a non-empty id")
        if identifier in identifiers:
            raise ValueError(f"duplicate attempt id {identifier!r}")
        identifiers.add(identifier)
        status = attempt.get("status")
        if status not in ATTEMPT_STATUSES:
            raise ValueError(f"unknown attempt status {status!r}")
        if not isinstance(attempt.get("active_execution"), bool):
            raise ValueError(f"attempt {identifier!r} requires active_execution")
        if attempt["active_execution"] and status not in {
            "accepted",
            "accepted_existing",
        }:
            raise ValueError(
                f"attempt {identifier!r} is active without an accepted decision"
            )
        for field in ("category", "comparison", "decision_basis"):
            if not isinstance(attempt.get(field), str) or not attempt[field]:
                raise ValueError(f"attempt {identifier!r} requires {field}")
    return document


__all__ = ["ATTEMPT_STATUSES", "load_attempt_registry"]

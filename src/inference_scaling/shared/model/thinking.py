"""Resolve template effort and thinking-token limits from one model profile.

The profile contains provider defaults, not algorithm constants. A missing
request inherits the configured selection; explicit effort and explicit budget
are alternatives. The returned configuration is request-local.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_thinking(
    config: Mapping[str, Any], *, effort: str | None = None,
    budget_tokens: int | None = None, enabled: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = config.get("thinking")
    if effort is not None and budget_tokens is not None:
        raise ValueError("reasoning_effort and thinking_budget cannot be set together")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("enable_thinking must be boolean")
    if profile is None:
        if effort is not None or budget_tokens is not None:
            raise ValueError("this model has no ar.thinking profile")
        output = dict(config["output"])
        if enabled is not None:
            output["thinking_mode"] = "enabled" if enabled else "disabled"
        return {**config, "output": output}, {"reasoning_effort": None, "requested_thinking_budget": None}

    explicit = effort is not None or budget_tokens is not None
    if not explicit:
        effort, budget_tokens = profile["reasoning_effort"], profile["budget_tokens"]
    if budget_tokens is not None and (type(budget_tokens) is not int or budget_tokens < 0):
        raise ValueError("thinking_budget must be a non-negative integer")
    limits = profile["effort_budgets"]
    if effort is not None:
        if not isinstance(effort, str) or not effort:
            raise ValueError("reasoning_effort must be a nonempty string")
        canonical = profile["effort_aliases"].get(effort, effort)
        if canonical is not None and canonical not in limits:
            raise ValueError(f"unknown reasoning effort {effort!r}")
        budget = 0 if canonical is None else limits[canonical]
        source = "effort"
    elif budget_tokens is not None:
        canonical = next((name for name, bound in sorted(limits.items(), key=lambda pair: pair[1])
                          if budget_tokens <= bound), None)
        if canonical is None:
            raise ValueError(f"thinking_budget exceeds the configured maximum {max(limits.values())}")
        budget, source = budget_tokens, "budget"
    else:
        canonical, budget = profile["default_effort"], profile["default_budget_tokens"]
        source = "default"
    if enabled is True and canonical is None:
        raise ValueError("reasoning_effort=none conflicts with enable_thinking=true")
    disabled = enabled is False or canonical is None or (
        enabled is None and not explicit and config["output"]["thinking_mode"] == "disabled")
    if disabled:
        if explicit and (budget_tokens not in (None, 0) or effort is not None and canonical is not None):
            raise ValueError("disabled thinking conflicts with the requested effort or thinking_budget")
        canonical, budget, source = None, 0, "disabled"
    # Zero budget is the low tier with no thinking tokens; render a closed
    # thinking block so Consilience cannot mistake content for an open thought.
    disabled = disabled or budget == 0
    options = dict(config["prompt"]["chat_template_kwargs"])
    if canonical is not None:
        options["reasoning_effort"] = canonical
    options["enable_thinking"] = not disabled
    resolved = {**config, "prompt": {**config["prompt"], "chat_template_kwargs": options},
                "output": {**config["output"], "thinking_mode": "disabled" if disabled else "enabled"}}
    return resolved, {"reasoning_effort": canonical, "requested_thinking_budget": budget,
                      "thinking_budget_source": source, "thinking_enabled": not disabled}


__all__ = ["resolve_thinking"]

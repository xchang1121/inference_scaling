"""Canonical experiment-component names shared by both model families."""

from __future__ import annotations


COMPONENTS = (
    "quality",
    "matched_target",
    "replay",
    "dynamic_is",
    "async",
    "passk",
    "ablations",
    "budget_curve",
    "length_ablation",
    "distribution",
    "infra",
    "vllm",
)

# vLLM currently serves autoregressive models only.  The remaining components
# form the default full comparison shared by the two model families.
FULL_COMPONENTS = tuple(component for component in COMPONENTS if component != "vllm")


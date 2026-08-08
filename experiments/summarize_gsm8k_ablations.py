"""Collect source-aligned GSM8K ablations into one machine-readable report."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


def _groups(tag: str) -> tuple[str, ...]:
    references = {
        "conditional-reference": (
            "candidate_rollout_budget",
            "guidance_steps",
            "sampling_temperature",
            "quality_compute_curve",
            "reward",
        ),
        "best-of-n-reference": (
            "sampling_temperature",
            "quality_compute_curve",
            "reward",
        ),
        "beam-reference": ("quality_compute_curve", "sampling_temperature"),
        "conditional-small-proposal-reference": (
            "quality_compute_curve",
            "importance_ratio_clipping",
            "reward",
            "sampling_temperature",
        ),
        "conditional-small-proposal-unclipped": (
            "importance_ratio_clipping",
        ),
    }
    if tag in references:
        return references[tag]
    if tag.startswith("alpha-") or tag.startswith("steps-"):
        return ("power_sampling",)
    if tag.startswith("candidates-"):
        return ("candidate_rollout_budget",)
    if tag.startswith("guidance-steps-"):
        return ("guidance_steps",)
    if "-reward-" in tag:
        return ("reward",)
    if "-temperature-" in tag:
        return ("sampling_temperature",)
    if tag.startswith("budget-"):
        return ("quality_compute_curve",)
    if tag.startswith("length-"):
        return ("generation_length",)
    return ()


def _per_example(summary: dict[str, Any], field: str) -> float:
    return float(summary[field]) / int(summary["examples"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_standard.toml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k_ablations.json"))
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    profile_root = args.results_root / profile
    grouped: dict[str, list[dict[str, Any]]] = {}
    implementation_variants: dict[str, dict[str, str]] = {}
    for summary_path in sorted(profile_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tag = str(summary["tag"])
        groups = _groups(tag)
        if not groups:
            continue
        manifest_path = summary_path.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        implementation = manifest["effective"]["implementation_sha256"]
        implementation_variants[
            json.dumps(implementation, sort_keys=True)
        ] = implementation
        effective = manifest["effective"]["config"]
        row = {
            "method": summary["method"],
            "tag": tag,
            "examples": int(summary["examples"]),
            "accuracy": float(summary["accuracy"]),
            "accuracy_wilson_95": summary["accuracy_wilson_95"],
            "forward_token_slots_per_example": _per_example(
                summary, "total_forward_token_slots"
            ),
            "shared_prefill_tokens_saved_per_example": _per_example(
                summary, "total_shared_prefill_tokens_saved"
            ),
            "estimated_dense_flops_per_example": _per_example(
                summary, "estimated_dense_forward_flops"
            ),
            "seconds_per_example": float(summary["mean_example_seconds"]),
            "mean_selected_output_tokens": float(
                summary["mean_selected_output_tokens"]
            ),
            "settings": {
                "max_new_tokens": effective["generation"]["max_new_tokens"],
                "sampling_temperature": effective.get("sampling", {}).get(
                    "temperature", 1.0
                ),
                "num_beams": effective["beam"]["num_beams"],
                "best_of_n_samples": effective["best_of_n"]["samples"],
                "mh": effective["mh"],
                "conditional_is": effective["conditional_is"],
            },
        }
        for group in groups:
            grouped.setdefault(group, []).append(row)

    if len(implementation_variants) > 1:
        raise ValueError("ablation summaries were produced by different implementations")
    implementation = next(iter(implementation_variants.values()), {})

    report = {
        "schema_version": 2,
        "profile": profile,
        "implementation_sha256": implementation,
        "groups": grouped,
        "alignment": {
            "main_metric": "single final response accuracy (pass@1)",
            "method_families": [
                "Base",
                "Beam Search",
                "Best-of-N self-consistency",
                "power-distribution suffix MH",
                "conditional energy importance sampling",
                "small-proposal off-policy conditional importance sampling",
                "GRPO",
            ],
            "ablations": [
                "M x K candidate/rollout budget",
                "guidance steps",
                "accuracy-compute trade-off",
                "log-probability, negative-entropy, self-certainty, self-consistency, and exact reward",
                "sampling temperature",
                "small-proposal log-ratio clipping",
                "generation length",
                "power and MCMC steps",
            ],
        },
        "primary_compute": (
            "observed padded forward token slots and 2 * parameter count * token "
            "slots, with base and small proposal models counted separately"
        ),
        "wall_time_role": "hardware-dependent supplemental metric",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

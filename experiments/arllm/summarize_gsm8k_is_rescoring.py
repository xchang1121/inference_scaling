"""Combine the frozen IS pass@k grid with the no-rescoring ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.arllm.gsm8k_is_passk import (
    _cost_ratio,
    _paired_pass_at_k_difference,
)
from inference_scaling.shared.rng import SeedStream

from experiments.shared.artifacts import file_sha256 as _sha256


REFERENCE_METHODS = (
    "conditional_is",
    "conditional_is_small_proposal",
    "conditional_is_small_proposal_unclipped",
)
UNCORRECTED_METHOD = "conditional_is_small_proposal_uncorrected"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(reference: dict[str, Any], ablation: dict[str, Any]) -> None:
    for key in (
        "benchmark",
        "profile",
        "problem_indices",
        "draws_per_problem",
        "workers",
        "input_weight_sha256",
        "compute_definition",
        "independence_definition",
    ):
        if reference.get(key) != ablation.get(key):
            raise ValueError(f"reference and ablation disagree on {key}")
    missing = set(REFERENCE_METHODS) - set(reference.get("methods", {}))
    if missing:
        raise ValueError(f"reference report is missing methods: {sorted(missing)}")
    if set(ablation.get("methods", {})) != {UNCORRECTED_METHOD}:
        raise ValueError("ablation report must contain only the uncorrected method")
    definition = ablation["method_definitions"][UNCORRECTED_METHOD]
    if definition != {
        "candidate_model": "base",
        "rollout_model": "proposal",
        "importance_log_ratio_clip": None,
        "apply_importance_correction": False,
    }:
        raise ValueError("uncorrected method definition does not disable rescoring")
    summary = ablation["methods"][UNCORRECTED_METHOD]
    for model in ("base", "proposal"):
        compute = summary["compute_by_model"][model]
        if int(compute["score_calls"]) != 0 or int(compute["scored_tokens"]) != 0:
            raise ValueError(f"uncorrected {model} backend unexpectedly performed scoring")


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "estimated_pass_at_k",
            "estimated_pass_at_k_problem_bootstrap_95",
            "seconds_excluding_model_load",
            "estimated_dense_forward_flops",
            "estimated_dense_forward_petaflops",
            "compute_by_model",
            "generated_answers",
            "unparseable_fraction",
        )
    }


def _short_wall_time_diagnostic(
    weighted_report: dict[str, Any],
    uncorrected_report: dict[str, Any],
) -> dict[str, Any]:
    for key in (
        "benchmark",
        "profile",
        "problem_indices",
        "draws_per_problem",
        "workers",
        "input_weight_sha256",
    ):
        if weighted_report.get(key) != uncorrected_report.get(key):
            raise ValueError(f"short wall-time reports disagree on {key}")
    if int(weighted_report["draws_per_problem"]) != 1:
        raise ValueError("short wall-time diagnostic must use one draw per problem")
    weighted = weighted_report["methods"]["conditional_is_small_proposal"]
    uncorrected = uncorrected_report["methods"][UNCORRECTED_METHOD]
    weighted_seconds = float(weighted["seconds_excluding_model_load"])
    uncorrected_seconds = float(uncorrected["seconds_excluding_model_load"])
    return {
        "problem_indices": weighted_report["problem_indices"],
        "draws_per_problem": 1,
        "workers": weighted_report["workers"],
        "weighted_seconds_excluding_model_load": weighted_seconds,
        "uncorrected_seconds_excluding_model_load": uncorrected_seconds,
        "uncorrected_over_weighted_wall_time": (
            uncorrected_seconds / weighted_seconds
        ),
        "weighted_compute_by_model": weighted["compute_by_model"],
        "uncorrected_compute_by_model": uncorrected["compute_by_model"],
        "interpretation": (
            "Same-session two-problem diagnostic only. Different candidate selections "
            "change generated lengths, so this is not a fixed-token-trace kernel timing."
        ),
    }


def build_report(
    reference: dict[str, Any],
    ablation: dict[str, Any],
    *,
    reference_path: Path,
    ablation_path: Path,
    weighted_smoke: dict[str, Any] | None = None,
    uncorrected_smoke: dict[str, Any] | None = None,
    weighted_smoke_path: Path | None = None,
    uncorrected_smoke_path: Path | None = None,
) -> dict[str, Any]:
    _validate(reference, ablation)
    uncorrected = ablation["methods"][UNCORRECTED_METHOD]
    methods = {
        name: _compact(reference["methods"][name]) for name in REFERENCE_METHODS
    }
    methods[UNCORRECTED_METHOD] = _compact(uncorrected)
    comparisons: dict[str, Any] = {}
    for reference_method in REFERENCE_METHODS:
        baseline = reference["methods"][reference_method]
        comparisons[f"uncorrected_minus_{reference_method}_pass_at_k"] = (
            _paired_pass_at_k_difference(
                baseline,
                uncorrected,
                draws=int(reference["draws_per_problem"]),
                seed=SeedStream(20260808).derive(
                    "is-rescoring-ablation", reference_method
                ),
            )
        )
        comparisons[f"{reference_method}_over_uncorrected_observed_cost"] = (
            _cost_ratio(baseline, uncorrected)
        )
    source_files = {
        str(reference_path).replace("\\", "/"): _sha256(reference_path),
        str(ablation_path).replace("\\", "/"): _sha256(ablation_path),
    }
    short_diagnostic = None
    if weighted_smoke is not None or uncorrected_smoke is not None:
        if (
            weighted_smoke is None
            or uncorrected_smoke is None
            or weighted_smoke_path is None
            or uncorrected_smoke_path is None
        ):
            raise ValueError("both short wall-time reports and paths are required")
        short_diagnostic = _short_wall_time_diagnostic(
            weighted_smoke, uncorrected_smoke
        )
        source_files[str(weighted_smoke_path).replace("\\", "/")] = _sha256(
            weighted_smoke_path
        )
        source_files[str(uncorrected_smoke_path).replace("\\", "/")] = _sha256(
            uncorrected_smoke_path
        )
    return {
        "schema_version": 1,
        "benchmark": reference["benchmark"],
        "profile": reference["profile"],
        "problem_indices": reference["problem_indices"],
        "draws_per_problem": reference["draws_per_problem"],
        "workers": reference["workers"],
        "methods": methods,
        "comparisons": comparisons,
        "source_files": source_files,
        "short_wall_time_diagnostic": short_diagnostic,
        "quality_interpretation": (
            "Paired intervals resample the 32 problems. The uncorrected method "
            "changes the continuation-weight target from the base model to the "
            "proposal model; it is not off-policy IS for the base target."
        ),
        "cost_interpretation": (
            "FLOPs use identical model-specific token-slot accounting. Full-grid wall "
            "times came from separate runs on the same RTX 3090 and are reported only "
            "as observations, not as a controlled speedup estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "results/gsm8k_3090/gsm8k_3090_aligned_is_passk_validated.json"
        ),
    )
    parser.add_argument(
        "--weighted-smoke",
        type=Path,
        default=Path("results/validation/gsm8k_clipped_smoke_paired.json"),
    )
    parser.add_argument(
        "--uncorrected-smoke",
        type=Path,
        default=Path("results/validation/gsm8k_uncorrected_smoke.json"),
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path(
            "results/gsm8k_3090/gsm8k_3090_aligned_is_uncorrected_validated.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/gsm8k_3090/"
            "gsm8k_3090_aligned_is_rescoring_ablation_validated.json"
        ),
    )
    args = parser.parse_args()
    report = build_report(
        _load(args.reference),
        _load(args.ablation),
        reference_path=args.reference,
        ablation_path=args.ablation,
        weighted_smoke=_load(args.weighted_smoke),
        uncorrected_smoke=_load(args.uncorrected_smoke),
        weighted_smoke_path=args.weighted_smoke,
        uncorrected_smoke_path=args.uncorrected_smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

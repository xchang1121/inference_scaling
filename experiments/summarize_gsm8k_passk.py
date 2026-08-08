"""Combine compatible GSM8K pass@k reports with paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

if __package__:
    from experiments.gsm8k_passk import _estimated_pass_at_k
else:
    from gsm8k_passk import _estimated_pass_at_k
from inference_scaling.rng import SeedStream


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _paired_difference(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    draws: int,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    reference_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in reference["per_problem"]
    }
    candidate_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in candidate["per_problem"]
    }
    if reference_by_problem.keys() != candidate_by_problem.keys():
        raise ValueError("paired pass@k methods do not contain the same problem rows")

    problem_indices = tuple(reference_by_problem)
    comparisons: dict[str, Any] = {}
    for k_text in reference["estimated_pass_at_k"]:
        if k_text not in candidate["estimated_pass_at_k"]:
            raise ValueError("paired pass@k methods do not contain the same k values")
        k = int(k_text)
        differences = [
            _estimated_pass_at_k(candidate_by_problem[index], draws, k)
            - _estimated_pass_at_k(reference_by_problem[index], draws, k)
            for index in problem_indices
        ]
        rng = random.Random(SeedStream(seed).derive("paired", k))
        bootstrap = sorted(
            statistics.fmean(
                differences[rng.randrange(len(differences))] for _ in differences
            )
            for _ in range(replicates)
        )
        comparisons[k_text] = {
            "candidate_minus_reference": statistics.fmean(differences),
            "paired_problem_bootstrap_95": [
                bootstrap[int(0.025 * replicates)],
                bootstrap[int(0.975 * replicates)],
            ],
        }
    return comparisons


def _cost_comparison(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | str]:
    candidate_seconds = float(candidate["seconds_excluding_model_load"])
    candidate_flops = int(candidate["estimated_dense_forward_flops"])
    if candidate_seconds <= 0 or candidate_flops <= 0:
        raise ValueError("candidate pass@k cost must be positive")
    return {
        "reference_over_candidate_inference_wall_time": (
            float(reference["seconds_excluding_model_load"]) / candidate_seconds
        ),
        "reference_over_candidate_inference_flops": (
            int(reference["estimated_dense_forward_flops"]) / candidate_flops
        ),
        "interpretation": (
            "The pair name is candidate_minus_reference, while each cost ratio uses "
            "reference as numerator and candidate as denominator. A cost ratio above "
            "one means the candidate used less of that inference-only resource."
        ),
    }


def _concise_method(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "examples",
        "draws_per_example",
        "generated_answers",
        "single_draw_accuracy",
        "estimated_pass_at_k",
        "estimated_pass_at_k_problem_bootstrap_95",
        "mean_unique_parsed_answers_across_all_draws",
        "mean_unique_full_outputs_across_all_draws",
        "unparseable_fraction",
        "total_forward_token_slots",
        "estimated_dense_forward_flops",
        "estimated_dense_forward_petaflops",
        "seconds_excluding_model_load",
        "seconds_per_generated_answer",
        "continuous_batching",
        "compute_by_model",
        "continuous_batching_by_model",
    )
    return {key: summary[key] for key in keys if key in summary}


def combine_reports(
    reports: list[dict[str, Any]],
    sources: list[dict[str, str]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if not reports or len(reports) != len(sources):
        raise ValueError("at least one report and one matching source are required")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")

    draws = int(reports[0]["draws_per_problem"])
    problem_indices = tuple(int(value) for value in reports[0]["problem_indices"])
    methods: dict[str, dict[str, Any]] = {}
    source_by_method: dict[str, str] = {}
    full_summaries: dict[str, dict[str, Any]] = {}
    method_order: list[str] = []
    for report, source in zip(reports, sources, strict=True):
        if int(report["draws_per_problem"]) != draws:
            raise ValueError("pass@k reports use different draw counts")
        if tuple(int(value) for value in report["problem_indices"]) != problem_indices:
            raise ValueError("pass@k reports use different problem rows or order")
        for method, summary in report["methods"].items():
            if method in methods:
                raise ValueError(f"duplicate pass@k method across reports: {method}")
            per_problem = tuple(
                int(item["problem_index"]) for item in summary["per_problem"]
            )
            if per_problem != problem_indices:
                raise ValueError(f"{method} per-problem rows do not match the report")
            methods[method] = _concise_method(summary)
            full_summaries[method] = summary
            source_by_method[method] = source["path"]
            method_order.append(method)

    paired: dict[str, Any] = {}
    for reference, candidate in combinations(method_order, 2):
        comparison_seed = SeedStream(bootstrap_seed).derive(reference, candidate)
        paired[f"{candidate}_minus_{reference}"] = {
            "pass_at_k": _paired_difference(
                full_summaries[reference],
                full_summaries[candidate],
                draws=draws,
                seed=comparison_seed,
                replicates=bootstrap_replicates,
            ),
            "cost": _cost_comparison(
                full_summaries[reference], full_summaries[candidate]
            ),
        }

    return {
        "schema_version": 1,
        "benchmark": reports[0]["benchmark"],
        "problem_indices": list(problem_indices),
        "draws_per_problem": draws,
        "method_order": method_order,
        "methods": methods,
        "source_by_method": source_by_method,
        "paired_comparisons": paired,
        "source_reports": sources,
        "bootstrap": {
            "unit": "problem",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
        },
        "cost_scope": (
            "Inference only. GRPO training cost is intentionally absent and must be "
            "added separately for end-to-end or amortized comparisons. Each method's "
            "FLOPs already sum model-specific parameter_count * forward_token_slots."
        ),
        "limitations": (
            "Problem bootstrap intervals capture row sampling uncertainty only. They do "
            "not include model, prompt, hyperparameter, finite-chain mixing, or finite "
            "importance-weight approximation uncertainty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    sources = [
        {"path": str(path), "sha256": _file_sha256(path)} for path in args.inputs
    ]
    combined = combine_reports(
        reports,
        sources,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

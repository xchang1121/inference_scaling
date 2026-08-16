"""Summarize the verifier small-rollout rescoring ablation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.summarize_gsm8k import _paired_difference

try:
    from experiments.shared.artifacts import file_sha256 as _sha256
except ModuleNotFoundError:  # direct execution from experiments/
    from shared.artifacts import file_sha256 as _sha256


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _bundle(directory: Path) -> dict[str, Any]:
    paths = {
        "manifest": directory / "manifest.json",
        "summary": directory / "summary.json",
        "records": directory / "records.jsonl",
    }
    return {
        "directory": directory,
        "paths": paths,
        "manifest": _load_json(paths["manifest"]),
        "summary": _load_json(paths["summary"]),
        "records": _load_records(paths["records"]),
    }


def _records_by_problem(bundle: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {
        int(record["problem_index"]): record for record in bundle["records"]
    }
    if len(records) != len(bundle["records"]):
        raise ValueError(f"duplicate problem rows in {bundle['directory']}")
    return records


def _normalized_pair_config(bundle: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(bundle["manifest"]["effective"])
    conditional = effective["config"]["conditional_is"]
    conditional.pop("importance_log_ratio_clip", None)
    conditional.pop("apply_importance_correction", None)
    effective.pop("tag", None)
    return effective


def _validate(
    standard: dict[str, Any],
    weighted: dict[str, Any],
    uncorrected: dict[str, Any],
) -> None:
    expected_method = "verifier_conditional_is_small_proposal"
    if weighted["summary"]["method"] != expected_method:
        raise ValueError("weighted report has the wrong method")
    if uncorrected["summary"]["method"] != expected_method:
        raise ValueError("uncorrected report has the wrong method")
    if _normalized_pair_config(weighted) != _normalized_pair_config(uncorrected):
        raise ValueError("paired verifier runs differ beyond importance correction")
    weighted_effective = weighted["manifest"]["effective"]
    uncorrected_effective = uncorrected["manifest"]["effective"]
    if weighted_effective["implementation_sha256"] != uncorrected_effective[
        "implementation_sha256"
    ]:
        raise ValueError("paired verifier runs used different implementations")
    if weighted_effective["input_weight_sha256"] != uncorrected_effective[
        "input_weight_sha256"
    ]:
        raise ValueError("paired verifier runs used different model weights")
    weighted_conditional = weighted_effective["config"]["conditional_is"]
    uncorrected_conditional = uncorrected_effective["config"]["conditional_is"]
    if weighted_conditional.get("importance_log_ratio_clip") != 10.0:
        raise ValueError("weighted verifier run must use the frozen log-ratio clip")
    if uncorrected_conditional.get("apply_importance_correction") is not False:
        raise ValueError("uncorrected verifier run did not disable importance correction")
    if uncorrected_conditional.get("importance_log_ratio_clip") is not None:
        raise ValueError("uncorrected verifier run retained log-ratio clipping")
    for bundle in (standard, weighted, uncorrected):
        records = _records_by_problem(bundle)
        manifest_indices = tuple(bundle["manifest"]["effective"]["problem_indices"])
        if tuple(sorted(records)) != manifest_indices:
            raise ValueError(f"incomplete problem grid in {bundle['directory']}")
        correct = sum(bool(record["correct"]) for record in records.values())
        if correct != int(bundle["summary"]["correct"]):
            raise ValueError(f"summary accuracy mismatch in {bundle['directory']}")
    paired_indices = tuple(sorted(_records_by_problem(weighted)))
    if tuple(sorted(_records_by_problem(uncorrected))) != paired_indices:
        raise ValueError("weighted and uncorrected runs use different problem rows")
    if tuple(sorted(_records_by_problem(standard))) != paired_indices:
        raise ValueError("standard verifier context uses different problem rows")
    for index in paired_indices:
        weighted_record = _records_by_problem(weighted)[index]
        uncorrected_record = _records_by_problem(uncorrected)[index]
        if weighted_record["question_sha256"] != uncorrected_record["question_sha256"]:
            raise ValueError(f"question hash mismatch for problem {index}")
        if weighted_record["diagnostics"]["reward_source"] != "exact":
            raise ValueError("weighted verifier run did not use exact reward")
        if uncorrected_record["diagnostics"]["reward_source"] != "exact":
            raise ValueError("uncorrected verifier run did not use exact reward")
    if int(uncorrected["summary"]["base_scored_tokens"]) != 0:
        raise ValueError("uncorrected verifier run performed base-model rescoring")
    if int(uncorrected["summary"]["base_score_forward_token_slots"]) != 0:
        raise ValueError("uncorrected verifier run contains base score slots")
    if int(weighted["summary"]["base_scored_tokens"]) <= 0:
        raise ValueError("weighted verifier run contains no base-model scores")


def _compute_breakdown(bundle: dict[str, Any]) -> dict[str, int | float]:
    summary = bundle["summary"]
    manifest = bundle["manifest"]
    base_parameters = int(manifest["model"]["parameter_count"])
    proposal_model = manifest.get("proposal_model")
    proposal_parameters = (
        int(proposal_model["parameter_count"]) if proposal_model is not None else 0
    )
    base_generation = (
        2 * base_parameters * int(summary["base_generation_forward_token_slots"])
    )
    base_scoring = 2 * base_parameters * int(summary["base_score_forward_token_slots"])
    proposal_generation = (
        2
        * proposal_parameters
        * int(summary["proposal_generation_forward_token_slots"])
    )
    return {
        "base_generation_flops": base_generation,
        "base_generation_petaflops": base_generation / 1e15,
        "base_scoring_flops": base_scoring,
        "base_scoring_petaflops": base_scoring / 1e15,
        "proposal_generation_flops": proposal_generation,
        "proposal_generation_petaflops": proposal_generation / 1e15,
        "total_flops": base_generation + base_scoring + proposal_generation,
        "total_petaflops": (
            base_generation + base_scoring + proposal_generation
        )
        / 1e15,
    }


def _compact(bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle["summary"]
    return {
        "accuracy": summary["accuracy"],
        "accuracy_wilson_95": summary["accuracy_wilson_95"],
        "correct": summary["correct"],
        "examples": summary["examples"],
        "seconds_excluding_model_load": summary["sum_example_seconds"],
        "mean_selected_output_tokens": summary["mean_selected_output_tokens"],
        "base_generated_tokens": summary["base_generated_tokens"],
        "base_scored_tokens": summary["base_scored_tokens"],
        "proposal_generated_tokens": summary["proposal_generated_tokens"],
        "base_generation_forward_token_slots": summary[
            "base_generation_forward_token_slots"
        ],
        "base_score_forward_token_slots": summary["base_score_forward_token_slots"],
        "proposal_generation_forward_token_slots": summary[
            "proposal_generation_forward_token_slots"
        ],
        "compute": _compute_breakdown(bundle),
        "manifest_fingerprint": bundle["manifest"]["fingerprint"],
    }


def _paired_behavior(
    weighted: dict[str, Any], uncorrected: dict[str, Any]
) -> dict[str, Any]:
    weighted_records = _records_by_problem(weighted)
    uncorrected_records = _records_by_problem(uncorrected)
    wins = losses = exact_output_matches = prediction_matches = 0
    for index, weighted_record in weighted_records.items():
        uncorrected_record = uncorrected_records[index]
        weighted_correct = bool(weighted_record["correct"])
        uncorrected_correct = bool(uncorrected_record["correct"])
        wins += int(uncorrected_correct and not weighted_correct)
        losses += int(weighted_correct and not uncorrected_correct)
        exact_output_matches += int(weighted_record["output"] == uncorrected_record["output"])
        prediction_matches += int(
            weighted_record["prediction"] == uncorrected_record["prediction"]
        )
    return {
        "uncorrected_wins": wins,
        "uncorrected_losses": losses,
        "correctness_ties": len(weighted_records) - wins - losses,
        "prediction_matches": prediction_matches,
        "exact_output_matches": exact_output_matches,
    }


def build_report(
    standard: dict[str, Any],
    weighted: dict[str, Any],
    uncorrected: dict[str, Any],
) -> dict[str, Any]:
    _validate(standard, weighted, uncorrected)
    weighted_summary = weighted["summary"]
    uncorrected_summary = uncorrected["summary"]
    weighted_flops = int(weighted_summary["estimated_dense_forward_flops"])
    uncorrected_flops = int(uncorrected_summary["estimated_dense_forward_flops"])
    weighted_seconds = float(weighted_summary["sum_example_seconds"])
    uncorrected_seconds = float(uncorrected_summary["sum_example_seconds"])
    sources: dict[str, str] = {}
    for bundle in (standard, weighted, uncorrected):
        for path in bundle["paths"].values():
            sources[str(path).replace("\\", "/")] = _sha256(path)
    return {
        "schema_version": 1,
        "benchmark": "OpenAI GSM8K official test split",
        "problem_indices": sorted(_records_by_problem(weighted)),
        "setting": {
            "candidate_model": "Qwen2.5-1.5B-Instruct",
            "standard_rollout_model": "Qwen2.5-1.5B-Instruct",
            "small_rollout_model": "Qwen2.5-0.5B-Instruct",
            "candidate_count": 8,
            "rollout_count_per_candidate": 3,
            "block_size": 48,
            "maximum_new_tokens": 192,
            "reward": "exact numeric verifier reward",
            "reward_temperature": 0.04,
        },
        "methods": {
            "standard_verifier_is": _compact(standard),
            "small_rollout_with_base_rescoring": _compact(weighted),
            "small_rollout_without_base_rescoring": _compact(uncorrected),
        },
        "quality_comparisons": {
            "weighted_minus_standard": _paired_difference(
                weighted["records"], standard["records"]
            ),
            "uncorrected_minus_weighted": _paired_difference(
                uncorrected["records"], weighted["records"]
            ),
            "uncorrected_minus_standard": _paired_difference(
                uncorrected["records"], standard["records"]
            ),
            "uncorrected_vs_weighted_behavior": _paired_behavior(
                weighted, uncorrected
            ),
        },
        "cost_comparisons": {
            "weighted_over_uncorrected_flops": weighted_flops / uncorrected_flops,
            "uncorrected_flops_reduction_fraction": 1
            - uncorrected_flops / weighted_flops,
            "uncorrected_over_weighted_wall_time": (
                uncorrected_seconds / weighted_seconds
            ),
            "uncorrected_base_generated_token_increase_fraction": (
                int(uncorrected_summary["base_generated_tokens"])
                / int(weighted_summary["base_generated_tokens"])
                - 1
            ),
            "uncorrected_proposal_generated_token_increase_fraction": (
                int(uncorrected_summary["proposal_generated_tokens"])
                / int(weighted_summary["proposal_generated_tokens"])
                - 1
            ),
            "uncorrected_output_length_increase_fraction": (
                float(uncorrected_summary["mean_selected_output_tokens"])
                / float(weighted_summary["mean_selected_output_tokens"])
                - 1
            ),
        },
        "interpretation": {
            "target": (
                "The uncorrected method keeps base-model candidate blocks but estimates "
                "their future verifier energy under 0.5B continuations. It is a biased "
                "small-model lookahead method, not off-policy IS for the full base target."
            ),
            "timing": (
                "Weighted and uncorrected rows were rerun sequentially in the same "
                "session with identical code, hardware, problem rows, seeds, and budgets. "
                "Their selected paths and generated lengths may differ."
            ),
        },
        "source_files": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/gsm8k/gsm8k-3090-aligned")
    parser.add_argument(
        "--standard-dir", default=root / "verifier_conditional_is-validated", type=Path
    )
    parser.add_argument(
        "--weighted-dir",
        default=root
        / "verifier_conditional_is_small_proposal-with-rescore-paired-validated",
        type=Path,
    )
    parser.add_argument(
        "--uncorrected-dir",
        default=root / "verifier_conditional_is_small_proposal-no-rescore-validated",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path(
            "results/gsm8k_3090/"
            "gsm8k_3090_aligned_verifier_rescoring_ablation_validated.json"
        ),
        type=Path,
    )
    args = parser.parse_args()
    report = build_report(
        _bundle(args.standard_dir),
        _bundle(args.weighted_dir),
        _bundle(args.uncorrected_dir),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

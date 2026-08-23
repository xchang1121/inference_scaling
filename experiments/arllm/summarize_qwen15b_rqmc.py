"""Aggregate the paired Qwen2.5-1.5B scrambled-Sobol rollout study."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import statistics
import tomllib
from typing import Any, Sequence

from experiments.shared.artifacts import file_sha256
from experiments.shared.statistics import (
    clustered_paired_binary_difference,
    wilson_interval,
)


RQMC_ARMS = (
    ("iid", "iid"),
    ("sobol", "scrambled_sobol"),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def _record_key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["problem_index"]), int(record["draw_index"])


def _sum_backend(records: Sequence[dict[str, Any]], field: str) -> int:
    return sum(int(record["backend_delta"].get(field, 0)) for record in records)


def _per_draw_rows(
    records: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    draws: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for draw in range(draws):
        current = [
            record for record in records if int(record["draw_index"]) == draw
        ]
        baseline = [
            record for record in reference if int(record["draw_index"]) == draw
        ]
        seconds = sum(float(record["elapsed_seconds"]) for record in current)
        baseline_seconds = sum(
            float(record["elapsed_seconds"]) for record in baseline
        )
        flops = _sum_backend(current, "estimated_dense_forward_flops")
        baseline_flops = _sum_backend(baseline, "estimated_dense_forward_flops")
        tokens = _sum_backend(current, "generated_tokens")
        baseline_tokens = _sum_backend(baseline, "generated_tokens")
        rows.append(
            {
                "draw_index": draw,
                "observations": len(current),
                "correct": sum(bool(record["correct"]) for record in current),
                "accuracy": sum(bool(record["correct"]) for record in current)
                / len(current),
                "seconds_excluding_model_load": seconds,
                "wall_factor_vs_iid_same_draw": seconds / baseline_seconds,
                "main_model_dense_forward_flops": flops,
                "main_model_flops_factor_vs_iid_same_draw": flops / baseline_flops,
                "generated_tokens": tokens,
                "generated_token_factor_vs_iid_same_draw": tokens / baseline_tokens,
            }
        )
    return rows


def _paired_first_step_diagnostics(
    reference: Sequence[dict[str, Any]],
    compared: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reference_by_key = {_record_key(record): record for record in reference}
    compared_by_key = {_record_key(record): record for record in compared}
    if reference_by_key.keys() != compared_by_key.keys():
        raise ValueError("RQMC arms do not contain the same paired observations")
    differences: list[float] = []
    selected_agreement = 0
    ranking_agreement = 0
    observations = 0
    for key in sorted(reference_by_key):
        left = reference_by_key[key]["diagnostics"]
        right = compared_by_key[key]["diagnostics"]
        left_tokens = left["candidate_token_ids_by_step"][0]
        right_tokens = right["candidate_token_ids_by_step"][0]
        if left_tokens != right_tokens:
            raise ValueError(
                "paired iid and scrambled-Sobol runs generated different first-step candidates"
            )
        left_weights = [
            float(value) for value in left["candidate_log_weight_estimates_by_step"][0]
        ]
        right_weights = [
            float(value) for value in right["candidate_log_weight_estimates_by_step"][0]
        ]
        if len(left_weights) != len(right_weights) or not left_weights:
            raise ValueError("paired first-step candidate weights do not align")
        differences.extend(
            right_value - left_value
            for left_value, right_value in zip(
                left_weights,
                right_weights,
                strict=True,
            )
        )
        selected_agreement += (
            int(left["selected_candidate_indices"][0])
            == int(right["selected_candidate_indices"][0])
        )
        ranking_agreement += left_weights.index(max(left_weights)) == right_weights.index(
            max(right_weights)
        )
        observations += 1
    return {
        "paired_observations": observations,
        "first_step_candidates_identical": True,
        "first_step_selected_index_agreement": selected_agreement / observations,
        "first_step_max_weight_candidate_agreement": ranking_agreement / observations,
        "first_step_log_weight_mean_difference": statistics.fmean(differences),
        "first_step_log_weight_mean_absolute_difference": statistics.fmean(
            abs(value) for value in differences
        ),
        "first_step_log_weight_root_mean_squared_difference": sqrt(
            statistics.fmean(value * value for value in differences)
        ),
        "interpretation": (
            "paired estimator difference at matched candidates; this is not an "
            "across-randomization variance estimate"
        ),
    }


def summarize_rqmc_study(
    *,
    config_path: Path,
    raw_root: Path,
    tag: str,
    draws: int,
    questions: int,
    rollout_count: int,
    phase: str = "screen",
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if draws <= 0 or questions <= 0 or rollout_count <= 0:
        raise ValueError("draws, questions and rollout_count must be positive")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if phase not in {"screen", "confirmation"}:
        raise ValueError("unknown RQMC study phase")
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, list[dict[str, str]]] = {}
    environments: list[dict[str, Any]] = []
    implementation_hashes: list[dict[str, str]] = []

    for arm, design in RQMC_ARMS:
        records_by_arm[arm] = []
        sources[arm] = []
        for draw in range(draws):
            directory = raw_root / profile / f"conditional_is-{tag}-{arm}-draw{draw}"
            records_path = directory / "records.jsonl"
            manifest_path = directory / "manifest.json"
            records = _load_jsonl(records_path)
            if len(records) != questions:
                raise ValueError(
                    f"{records_path} contains {len(records)} rows; expected {questions}"
                )
            for record in records:
                diagnostics = record["diagnostics"]
                if diagnostics["rollout_design"] != design:
                    raise ValueError(f"{records_path} has the wrong rollout design")
                if diagnostics["reward_source"] != "frozen_consensus":
                    raise ValueError(f"{records_path} has the wrong reward source")
                if int(diagnostics["configured_rollout_count"]) != rollout_count:
                    raise ValueError(f"{records_path} has the wrong rollout count")
                if bool(diagnostics["uses_test_gold_oracle"]):
                    raise ValueError(f"{records_path} used the test answer during inference")
                if int(record["draw_index"]) != draw:
                    raise ValueError(f"{records_path} has the wrong draw index")
            records_by_arm[arm].extend(records)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            environments.append(dict(manifest["environment"]))
            implementation_hashes.append(
                dict(manifest["effective"]["implementation_sha256"])
            )
            sources[arm].append(
                {
                    "records": records_path.as_posix(),
                    "records_sha256": file_sha256(records_path),
                    "manifest": manifest_path.as_posix(),
                    "manifest_fingerprint": str(manifest["fingerprint"]),
                }
            )

    reference = records_by_arm["iid"]
    reference_seconds = sum(float(record["elapsed_seconds"]) for record in reference)
    reference_flops = _sum_backend(reference, "estimated_dense_forward_flops")
    reference_tokens = _sum_backend(reference, "generated_tokens")
    table: list[dict[str, Any]] = []
    for arm, design in RQMC_ARMS:
        records = records_by_arm[arm]
        rows = len(records)
        correct = sum(bool(record["correct"]) for record in records)
        seconds = sum(float(record["elapsed_seconds"]) for record in records)
        flops = _sum_backend(records, "estimated_dense_forward_flops")
        tokens = _sum_backend(records, "generated_tokens")
        paired = clustered_paired_binary_difference(
            records,
            reference,
            cluster_key="problem_index",
            outcome_key="correct",
            seed=20260823 + len(arm),
            replicates=bootstrap_replicates,
        )
        table.append(
            {
                "arm": arm,
                "rollout_design": design,
                "observations": rows,
                "correct": correct,
                "accuracy": correct / rows,
                "wilson_95_treating_draws_as_observations": wilson_interval(
                    correct, rows
                ),
                "sum_seconds_excluding_model_load": seconds,
                "mean_seconds_per_observation": seconds / rows,
                "main_model_dense_forward_flops": flops,
                "main_model_dense_forward_petaflops": flops / 1e15,
                "generated_tokens": tokens,
                "wall_factor_vs_iid": seconds / reference_seconds,
                "main_model_flops_factor_vs_iid": flops / reference_flops,
                "generated_token_factor_vs_iid": tokens / reference_tokens,
                "mean_rollout_ess_descriptive": statistics.fmean(
                    float(record["diagnostics"]["mean_rollout_ess"])
                    for record in records
                ),
                "mean_within_candidate_log_weight_dispersion": statistics.fmean(
                    float(
                        record["diagnostics"][
                            "mean_within_candidate_log_weight_dispersion"
                        ]
                    )
                    for record in records
                ),
                "paired_vs_iid": {
                    "accuracy_difference": paired["difference"],
                    "clustered_bootstrap_95": paired["clustered_bootstrap_95"],
                    "bootstrap_unit": (
                        "GSM8K question; all draws stay in the same cluster"
                    ),
                },
                "per_draw": _per_draw_rows(records, reference, draws),
            }
        )

    keys = {
        arm: sorted(_record_key(record) for record in records)
        for arm, records in records_by_arm.items()
    }
    if any(value != keys["iid"] for value in keys.values()):
        raise ValueError("RQMC arms do not use the same GSM8K rows and draws")
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("RQMC arms were run in different environments")
    if any(
        hashes != implementation_hashes[0] for hashes in implementation_hashes[1:]
    ):
        raise ValueError("RQMC arms do not use the same implementation")

    paired_weights = _paired_first_step_diagnostics(
        reference,
        records_by_arm["sobol"],
    )
    sobol = next(row for row in table if row["arm"] == "sobol")
    accuracy_difference = float(sobol["paired_vs_iid"]["accuracy_difference"])
    passes = (
        accuracy_difference >= -0.03125
        and (
            sobol["wall_factor_vs_iid"] <= 0.95
            or sobol["main_model_flops_factor_vs_iid"] <= 0.95
        )
    ) or (
        accuracy_difference >= 0.03125
        and sobol["wall_factor_vs_iid"] <= 1.05
        and sobol["main_model_flops_factor_vs_iid"] <= 1.05
    )
    decision = {
        "result": (
            "advance_to_confirmation"
            if phase == "screen" and passes
            else "accepted"
            if phase == "confirmation" and passes
            else "rejected"
        ),
        "passing_arms": ["sobol"] if passes else [],
        "confirmation_run_required": phase == "screen" and passes,
        "reason": (
            "scrambled Sobol met the registered quality-cost gate"
            if passes
            else "scrambled Sobol did not meet the registered quality-cost gate"
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "phase": phase,
        "scope": {
            "model": "Qwen2.5-1.5B-Instruct",
            "model_path": str(config["models"]["base"]),
            "model_revision": str(config["models"]["base_revision"]),
            "model_weight_sha256": str(config["models"]["base_weight_sha256"]),
            "model_family": "arllm",
            "dllm_experiments": False,
            "dataset": "pinned OpenAI GSM8K test split",
            "profile": profile,
            "questions": questions,
            "draws": draws,
            "candidate_count": int(config["conditional_is"]["candidate_count"]),
            "rollouts_per_candidate": rollout_count,
            "block_size": int(config["conditional_is"]["block_size"]),
            "maximum_new_tokens": int(config["generation"]["max_new_tokens"]),
            "reward": "independent-pilot frozen consensus",
            "pilot_samples": int(config["iterated_is"]["pilot_samples"]),
            "environment": environments[0],
        },
        "comparison": (
            "paired iid and digitally scrambled Sobol rollout uniforms at fixed "
            "questions, candidate count, rollout count, proposal, reward and model"
        ),
        "problem_indices": sorted({key[0] for key in keys["iid"]}),
        "table": table,
        "paired_first_step_weight_diagnostics": paired_weights,
        "decision": decision,
        "sources": sources,
        "implementation_sha256": implementation_hashes[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--questions", type=int, required=True)
    parser.add_argument("--rollout-count", type=int, required=True)
    parser.add_argument("--phase", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_rqmc_study(
        config_path=args.config,
        raw_root=args.raw_root,
        tag=args.tag,
        draws=args.draws,
        questions=args.questions,
        rollout_count=args.rollout_count,
        phase=args.phase,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

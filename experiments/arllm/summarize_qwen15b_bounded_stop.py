"""Aggregate the paired Qwen2.5-1.5B exact rollout-stopping study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any, Sequence

from experiments.shared.artifacts import file_sha256
from experiments.shared.statistics import (
    clustered_paired_binary_difference,
    wilson_interval,
)


BOUNDED_STOP_ARMS = (
    ("full", False),
    ("bounded", True),
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


def _key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["problem_index"]), int(record["draw_index"])


def _sum_backend(records: Sequence[dict[str, Any]], field: str) -> int:
    return sum(int(record["backend_delta"].get(field, 0)) for record in records)


def _sum_diagnostic(records: Sequence[dict[str, Any]], field: str) -> int:
    return sum(int(record["diagnostics"].get(field, 0)) for record in records)


def _rollout_accounting(
    records: Sequence[dict[str, Any]],
    *,
    early_stop: bool,
) -> tuple[int, int, int, int]:
    planned = _sum_diagnostic(records, "rollout_evaluations_planned")
    performed = _sum_diagnostic(records, "rollout_evaluations_performed")
    skipped = _sum_diagnostic(records, "rollout_evaluations_skipped")
    batches = _sum_diagnostic(records, "rollout_evaluation_batches")
    if not early_stop:
        legacy_performed = _sum_diagnostic(records, "rollout_evaluations")
        if planned == 0:
            planned = legacy_performed
        if performed == 0:
            performed = legacy_performed
        if batches == 0:
            batches = _sum_diagnostic(records, "guidance_steps")
    return planned, performed, skipped, batches


def _per_draw_rows(
    records: Sequence[dict[str, Any]],
    full: Sequence[dict[str, Any]],
    draws: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for draw in range(draws):
        current = [
            record for record in records if int(record["draw_index"]) == draw
        ]
        reference = [
            record for record in full if int(record["draw_index"]) == draw
        ]
        seconds = sum(float(record["elapsed_seconds"]) for record in current)
        reference_seconds = sum(
            float(record["elapsed_seconds"]) for record in reference
        )
        flops = _sum_backend(current, "estimated_dense_forward_flops")
        reference_flops = _sum_backend(reference, "estimated_dense_forward_flops")
        tokens = _sum_backend(current, "generated_tokens")
        reference_tokens = _sum_backend(reference, "generated_tokens")
        rows.append(
            {
                "draw_index": draw,
                "observations": len(current),
                "seconds_excluding_model_load": seconds,
                "wall_factor_vs_full_same_draw": seconds / reference_seconds,
                "main_model_dense_forward_flops": flops,
                "main_model_flops_factor_vs_full_same_draw": flops / reference_flops,
                "generated_tokens": tokens,
                "generated_token_factor_vs_full_same_draw": tokens / reference_tokens,
            }
        )
    return rows


def _paired_agreement(
    full: Sequence[dict[str, Any]],
    bounded: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    full_by_key = {_key(record): record for record in full}
    bounded_by_key = {_key(record): record for record in bounded}
    if full_by_key.keys() != bounded_by_key.keys():
        raise ValueError("bounded-stop arms do not contain the same observations")
    output_matches = 0
    selected_matches = 0
    candidate_matches = 0
    for key in sorted(full_by_key):
        left = full_by_key[key]
        right = bounded_by_key[key]
        output_matches += left["output"] == right["output"]
        selected_matches += (
            left["diagnostics"]["selected_candidate_indices"]
            == right["diagnostics"]["selected_candidate_indices"]
        )
        candidate_matches += (
            left["diagnostics"]["candidate_token_ids_by_step"]
            == right["diagnostics"]["candidate_token_ids_by_step"]
        )
    observations = len(full_by_key)
    return {
        "paired_observations": observations,
        "exact_output_matches": output_matches,
        "exact_output_match_fraction": output_matches / observations,
        "selected_index_matches": selected_matches,
        "selected_index_match_fraction": selected_matches / observations,
        "candidate_token_matches": candidate_matches,
        "candidate_token_match_fraction": candidate_matches / observations,
    }


def summarize_bounded_stop_study(
    *,
    config_path: Path,
    raw_root: Path,
    tag: str,
    draws: int,
    questions: int,
    rollout_count: int,
    evaluation_batch_size: int,
    log_weight_lower: float,
    log_weight_upper: float,
    phase: str = "screen",
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if min(draws, questions, rollout_count, evaluation_batch_size) <= 0:
        raise ValueError("study sizes must be positive")
    if log_weight_lower > log_weight_upper:
        raise ValueError("log-weight bounds must be ordered")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if phase not in {"screen", "confirmation"}:
        raise ValueError("unknown bounded-stop study phase")
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, list[dict[str, str]]] = {}
    environments: list[dict[str, Any]] = []
    implementation_hashes: list[dict[str, str]] = []
    for arm, early_stop in BOUNDED_STOP_ARMS:
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
                if bool(diagnostics["exact_rollout_early_stop_enabled"]) != early_stop:
                    raise ValueError(f"{records_path} has the wrong stopping mode")
                if diagnostics["reward_source"] != "frozen_consensus":
                    raise ValueError(f"{records_path} has the wrong reward source")
                if int(diagnostics["configured_rollout_count"]) != rollout_count:
                    raise ValueError(f"{records_path} has the wrong rollout count")
                if int(record["draw_index"]) != draw:
                    raise ValueError(f"{records_path} has the wrong draw index")
                if bool(diagnostics["uses_test_gold_oracle"]):
                    raise ValueError(f"{records_path} used the test answer during inference")
                if early_stop and int(
                    diagnostics["rollout_evaluation_batch_size"]
                ) != evaluation_batch_size:
                    raise ValueError(f"{records_path} has the wrong evaluation batch size")
                if early_stop and diagnostics[
                    "declared_rollout_log_weight_bounds"
                ] != [log_weight_lower, log_weight_upper]:
                    raise ValueError(f"{records_path} has the wrong log-weight bounds")
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

    full = records_by_arm["full"]
    reference_seconds = sum(float(record["elapsed_seconds"]) for record in full)
    reference_flops = _sum_backend(full, "estimated_dense_forward_flops")
    reference_tokens = _sum_backend(full, "generated_tokens")
    table: list[dict[str, Any]] = []
    for arm, early_stop in BOUNDED_STOP_ARMS:
        records = records_by_arm[arm]
        rows = len(records)
        correct = sum(bool(record["correct"]) for record in records)
        seconds = sum(float(record["elapsed_seconds"]) for record in records)
        flops = _sum_backend(records, "estimated_dense_forward_flops")
        tokens = _sum_backend(records, "generated_tokens")
        planned, performed, skipped, evaluation_batches = _rollout_accounting(
            records,
            early_stop=early_stop,
        )
        paired = clustered_paired_binary_difference(
            records,
            full,
            cluster_key="problem_index",
            outcome_key="correct",
            seed=20260823 + len(arm),
            replicates=bootstrap_replicates,
        )
        table.append(
            {
                "arm": arm,
                "exact_rollout_early_stop": early_stop,
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
                "wall_factor_vs_full": seconds / reference_seconds,
                "main_model_flops_factor_vs_full": flops / reference_flops,
                "generated_token_factor_vs_full": tokens / reference_tokens,
                "rollout_evaluations_planned": planned,
                "rollout_evaluations_performed": performed,
                "rollout_evaluations_skipped": skipped,
                "rollout_skip_fraction": skipped / planned if planned else 0.0,
                "rollout_evaluation_batches": evaluation_batches,
                "exact_early_stop_steps": _sum_diagnostic(
                    records, "exact_early_stop_steps"
                ),
                "selection_invariant_verified_steps": _sum_diagnostic(
                    records, "selection_invariant_verified_steps"
                ),
                "paired_vs_full": {
                    "accuracy_difference": paired["difference"],
                    "clustered_bootstrap_95": paired["clustered_bootstrap_95"],
                    "bootstrap_unit": (
                        "GSM8K question; all draws stay in the same cluster"
                    ),
                },
                "per_draw": _per_draw_rows(records, full, draws),
            }
        )

    keys = {
        arm: sorted(_key(record) for record in records)
        for arm, records in records_by_arm.items()
    }
    if keys["full"] != keys["bounded"]:
        raise ValueError("bounded-stop arms do not use the same rows and draws")
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("bounded-stop arms were run in different environments")
    if any(
        hashes != implementation_hashes[0] for hashes in implementation_hashes[1:]
    ):
        raise ValueError("bounded-stop arms do not use the same implementation")
    agreement = _paired_agreement(full, records_by_arm["bounded"])
    bounded = next(row for row in table if row["arm"] == "bounded")
    exact = (
        agreement["exact_output_match_fraction"] == 1.0
        and agreement["selected_index_match_fraction"] == 1.0
        and agreement["candidate_token_match_fraction"] == 1.0
    )
    passes = exact and (
        bounded["wall_factor_vs_full"] <= 0.95
        or bounded["main_model_flops_factor_vs_full"] <= 0.95
    )
    decision = {
        "result": (
            "advance_to_confirmation"
            if phase == "screen" and passes
            else "accepted"
            if phase == "confirmation" and passes
            else "rejected"
        ),
        "passing_arms": ["bounded"] if passes else [],
        "confirmation_run_required": phase == "screen" and passes,
        "reason": (
            "bounded stopping preserved every paired output and reduced registered cost"
            if passes
            else "bounded stopping failed exact agreement or the registered cost gate"
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
            "evaluation_batch_size": evaluation_batch_size,
            "declared_rollout_log_weight_bounds": [
                log_weight_lower,
                log_weight_upper,
            ],
            "reward": "independent-pilot frozen consensus",
            "reward_temperature": float(
                config["conditional_is"]["reward_temperature"]
            ),
            "pilot_samples": int(config["iterated_is"]["pilot_samples"]),
            "block_size": int(config["conditional_is"]["block_size"]),
            "maximum_new_tokens": int(config["generation"]["max_new_tokens"]),
            "environment": environments[0],
        },
        "comparison": (
            "complete finite rollout evaluation versus exact interval-certified "
            "stopping with paired candidates, rollout seeds and selection uniforms"
        ),
        "problem_indices": sorted({key[0] for key in keys["full"]}),
        "table": table,
        "paired_exact_agreement": agreement,
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
    parser.add_argument("--evaluation-batch-size", type=int, required=True)
    parser.add_argument("--log-weight-lower", type=float, required=True)
    parser.add_argument("--log-weight-upper", type=float, required=True)
    parser.add_argument("--phase", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_bounded_stop_study(
        config_path=args.config,
        raw_root=args.raw_root,
        tag=args.tag,
        draws=args.draws,
        questions=args.questions,
        rollout_count=args.rollout_count,
        evaluation_batch_size=args.evaluation_batch_size,
        log_weight_lower=args.log_weight_lower,
        log_weight_upper=args.log_weight_upper,
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

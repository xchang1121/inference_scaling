"""Aggregate the paired Qwen2.5-1.5B iterated-SIR screening grid."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tomllib
from pathlib import Path
from typing import Any, Sequence

from experiments.shared.artifacts import file_sha256
from experiments.shared.statistics import quantile, wilson_interval


ISIR_ARMS = (
    ("n9-u1", 9, 1),
    ("n5-u2", 5, 2),
    ("n3-u4", 3, 4),
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


def _clustered_paired_difference(
    arm: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    def grouped(records: Sequence[dict[str, Any]]) -> dict[int, list[float]]:
        result: dict[int, list[float]] = {}
        for record in records:
            result.setdefault(int(record["problem_index"]), []).append(
                float(bool(record["correct"]))
            )
        return result

    left = grouped(arm)
    right = grouped(reference)
    if left.keys() != right.keys():
        raise ValueError("paired i-SIR arms must contain the same GSM8K rows")
    indices = sorted(left)
    differences: list[float] = []
    for index in indices:
        if len(left[index]) != len(right[index]):
            raise ValueError("paired i-SIR arms must contain the same draw count")
        differences.append(
            statistics.fmean(left[index]) - statistics.fmean(right[index])
        )
    rng = random.Random(seed)
    bootstrap = [
        statistics.fmean(
            differences[rng.randrange(len(differences))] for _ in differences
        )
        for _ in range(replicates)
    ]
    return {
        "accuracy_difference": statistics.fmean(differences),
        "clustered_bootstrap_95": [
            quantile(bootstrap, 0.025),
            quantile(bootstrap, 0.975),
        ],
        "bootstrap_unit": "GSM8K question; all draws stay in the same cluster",
    }


def summarize_isir_screen(
    *,
    config_path: Path,
    raw_root: Path,
    tag: str,
    draws: int,
    bootstrap_replicates: int = 10_000,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    if draws <= 0 or bootstrap_replicates <= 0:
        raise ValueError("draws and bootstrap_replicates must be positive")
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    expected = int(config["run"]["sample_count"])
    records_by_arm: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, list[dict[str, str]]] = {}
    environments: list[dict[str, Any]] = []
    implementation_hashes: list[dict[str, str]] = []
    for arm, pool_size, updates in ISIR_ARMS:
        records_by_arm[arm] = []
        sources[arm] = []
        for draw in range(draws):
            directory = (
                raw_root
                / profile
                / f"iterated_conditional_is-{tag}-{arm}-draw{draw}"
            )
            records_path = directory / "records.jsonl"
            manifest_path = directory / "manifest.json"
            records = _load_jsonl(records_path)
            if len(records) != expected:
                raise ValueError(
                    f"{records_path} contains {len(records)} rows; expected {expected}"
                )
            for record in records:
                diagnostics = record["diagnostics"]
                if int(diagnostics["pool_size"]) != pool_size:
                    raise ValueError(f"{records_path} has the wrong pool size")
                if int(diagnostics["updates_per_block"]) != updates:
                    raise ValueError(f"{records_path} has the wrong update count")
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

    reference = records_by_arm["n9-u1"]
    reference_seconds = sum(float(record["elapsed_seconds"]) for record in reference)
    reference_flops = sum(
        int(record["backend_delta"]["estimated_dense_forward_flops"])
        for record in reference
    )
    table: list[dict[str, Any]] = []
    for arm, pool_size, updates in ISIR_ARMS:
        records = records_by_arm[arm]
        correct = sum(bool(record["correct"]) for record in records)
        seconds = sum(float(record["elapsed_seconds"]) for record in records)
        flops = sum(
            int(record["backend_delta"]["estimated_dense_forward_flops"])
            for record in records
        )
        rows = len(records)
        table.append(
            {
                "arm": arm,
                "pool_size": pool_size,
                "updates_per_block": updates,
                "distinct_candidate_rollout_states_per_block": (
                    1 + updates * (pool_size - 1)
                ),
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
                "wall_factor_vs_n9_u1": seconds / reference_seconds,
                "main_model_flops_factor_vs_n9_u1": flops / reference_flops,
                "mean_rollout_ess": statistics.fmean(
                    float(record["diagnostics"]["mean_rollout_ess"])
                    for record in records
                ),
                "mean_rollout_reward": statistics.fmean(
                    float(record["diagnostics"]["mean_rollout_reward"])
                    for record in records
                ),
                "paired_vs_n9_u1": _clustered_paired_difference(
                    records,
                    reference,
                    seed=20260822 + pool_size * 100 + updates,
                    replicates=bootstrap_replicates,
                ),
            }
        )
    indices = {
        arm: sorted({int(record["problem_index"]) for record in records})
        for arm, records in records_by_arm.items()
    }
    if any(value != indices["n9-u1"] for value in indices.values()):
        raise ValueError("i-SIR arms do not use the same GSM8K rows")
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("i-SIR arms were run in different environments")
    if any(
        hashes != implementation_hashes[0] for hashes in implementation_hashes[1:]
    ):
        raise ValueError("i-SIR arms do not use the same implementation")
    baseline: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    if baseline_path is not None:
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_rows = [
            row
            for row in baseline_report["table"]
            if row["method"] == "conditional_is"
        ]
        if len(baseline_rows) != 1:
            raise ValueError("baseline report must contain one conditional_is row")
        if list(baseline_report["problem_indices"]) != indices["n9-u1"]:
            raise ValueError("baseline report uses different GSM8K rows")
        row = baseline_rows[0]
        baseline = {
            "source": baseline_path.as_posix(),
            "source_sha256": file_sha256(baseline_path),
            "method": "conditional_is",
            "reward": "cumulative self-consistency",
            "observations": int(row["examples"]),
            "draws": 1,
            "accuracy": float(row["accuracy"]),
            "seconds_excluding_model_load": float(row["seconds_excluding_model_load"]),
            "main_model_dense_forward_flops": int(row["estimated_dense_forward_flops"]),
        }
        for result in table:
            result["wall_factor_per_draw_vs_existing_conditional_is"] = (
                result["sum_seconds_excluding_model_load"]
                / draws
                / baseline["seconds_excluding_model_load"]
            )
            result["main_model_flops_factor_per_draw_vs_existing_conditional_is"] = (
                result["main_model_dense_forward_flops"]
                / draws
                / baseline["main_model_dense_forward_flops"]
            )
        passing_arms = [
            result["arm"]
            for result in table
            if (
                result["accuracy"] - baseline["accuracy"] >= -0.03125
                and (
                    result["wall_factor_per_draw_vs_existing_conditional_is"] <= 0.95
                    or result[
                        "main_model_flops_factor_per_draw_vs_existing_conditional_is"
                    ]
                    <= 0.95
                )
            )
            or (
                result["accuracy"] - baseline["accuracy"] >= 0.03125
                and result["wall_factor_per_draw_vs_existing_conditional_is"] <= 1.05
                and result[
                    "main_model_flops_factor_per_draw_vs_existing_conditional_is"
                ]
                <= 1.05
            )
        ]
        decision = {
            "result": "advance_to_confirmation" if passing_arms else "rejected",
            "passing_arms": passing_arms,
            "reason": (
                "at least one arm passed the registered screen gate"
                if passing_arms
                else "no arm met the quality-cost gate against existing conditional IS"
            ),
            "confirmation_run_required": bool(passing_arms),
        }
    return {
        "schema_version": 1,
        "status": "complete" if decision is not None else "screening",
        "scope": {
            "model": "Qwen2.5-1.5B-Instruct",
            "model_path": str(config["models"]["base"]),
            "model_revision": str(config["models"]["base_revision"]),
            "model_weight_sha256": str(config["models"]["base_weight_sha256"]),
            "model_family": "arllm",
            "dllm_experiments": False,
            "dataset": "pinned OpenAI GSM8K test split",
            "profile": profile,
            "questions": expected,
            "draws": draws,
            "reward": "independent-pilot frozen consensus",
            "pilot_samples": int(config["iterated_is"]["pilot_samples"]),
            "rollouts_per_candidate": int(config["conditional_is"]["rollout_count"]),
            "block_size": int(config["conditional_is"]["block_size"]),
            "maximum_new_tokens": int(config["generation"]["max_new_tokens"]),
            "environment": environments[0],
        },
        "comparison": (
            "all arms evaluate nine distinct candidate-rollout states per block; "
            "n9-u1 is one-shot SIR and the other arms reuse the selected extended state"
        ),
        "problem_indices": indices["n9-u1"],
        "table": table,
        "existing_conditional_is_baseline": baseline,
        "decision": decision,
        "sources": sources,
        "implementation_sha256": implementation_hashes[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_isir_screen(
        config_path=args.config,
        raw_root=args.raw_root,
        tag=args.tag,
        draws=args.draws,
        bootstrap_replicates=args.bootstrap_replicates,
        baseline_path=args.baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Aggregate the paired Qwen2.5-1.5B MH suffix-schedule screen."""

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


MH_SUFFIX_ARMS = (
    ("uniform", "uniform"),
    ("inverse", "inverse_length"),
    ("multiscale", "multiscale"),
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


def _weighted_mean(
    records: Sequence[dict[str, Any]],
    value: str,
    weight: str = "attempts",
) -> float:
    denominator = sum(int(record["diagnostics"][weight]) for record in records)
    if denominator <= 0:
        return 0.0
    return sum(
        float(record["diagnostics"][value])
        * int(record["diagnostics"][weight])
        for record in records
    ) / denominator


def summarize_mh_suffix_screen(
    *,
    config_path: Path,
    raw_root: Path,
    tag: str,
    draws: int,
    bootstrap_replicates: int = 10_000,
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

    for arm, schedule in MH_SUFFIX_ARMS:
        records_by_arm[arm] = []
        sources[arm] = []
        for draw in range(draws):
            directory = raw_root / profile / f"mh-{tag}-{arm}-draw{draw}"
            records_path = directory / "records.jsonl"
            manifest_path = directory / "manifest.json"
            records = _load_jsonl(records_path)
            if len(records) != expected:
                raise ValueError(
                    f"{records_path} contains {len(records)} rows; expected {expected}"
                )
            for record in records:
                diagnostics = record["diagnostics"]
                if diagnostics["suffix_schedule"] != schedule:
                    raise ValueError(f"{records_path} has the wrong suffix schedule")
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

    reference = records_by_arm["uniform"]
    reference_seconds = sum(float(record["elapsed_seconds"]) for record in reference)
    reference_flops = sum(
        int(record["backend_delta"]["estimated_dense_forward_flops"])
        for record in reference
    )
    reference_tokens = sum(
        int(record["backend_delta"]["generated_tokens"]) for record in reference
    )
    table: list[dict[str, Any]] = []
    for arm, schedule in MH_SUFFIX_ARMS:
        records = records_by_arm[arm]
        rows = len(records)
        correct = sum(bool(record["correct"]) for record in records)
        seconds = sum(float(record["elapsed_seconds"]) for record in records)
        flops = sum(
            int(record["backend_delta"]["estimated_dense_forward_flops"])
            for record in records
        )
        generated_tokens = sum(
            int(record["backend_delta"]["generated_tokens"]) for record in records
        )
        attempts = sum(int(record["diagnostics"]["attempts"]) for record in records)
        accepted = sum(int(record["diagnostics"]["accepted"]) for record in records)
        paired = clustered_paired_binary_difference(
            records,
            reference,
            cluster_key="problem_index",
            outcome_key="correct",
            seed=20260822 + len(arm),
            replicates=bootstrap_replicates,
        )
        table.append(
            {
                "arm": arm,
                "suffix_schedule": schedule,
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
                "generated_tokens": generated_tokens,
                "attempts": attempts,
                "accepted": accepted,
                "acceptance_rate": accepted / attempts if attempts else 0.0,
                "mean_proposed_suffix_length": _weighted_mean(
                    records, "mean_proposed_suffix_length"
                ),
                "mean_proposed_token_changes": _weighted_mean(
                    records, "mean_proposed_token_changes"
                ),
                "mean_accepted_token_changes": _weighted_mean(
                    records, "mean_accepted_token_changes"
                ),
                "wall_factor_vs_uniform": seconds / reference_seconds,
                "main_model_flops_factor_vs_uniform": flops / reference_flops,
                "generated_token_factor_vs_uniform": (
                    generated_tokens / reference_tokens
                ),
                "paired_vs_uniform": {
                    "accuracy_difference": paired["difference"],
                    "clustered_bootstrap_95": paired["clustered_bootstrap_95"],
                    "bootstrap_unit": (
                        "GSM8K question; all draws stay in the same cluster"
                    ),
                },
            }
        )

    indices = {
        arm: sorted({int(record["problem_index"]) for record in records})
        for arm, records in records_by_arm.items()
    }
    if any(value != indices["uniform"] for value in indices.values()):
        raise ValueError("MH suffix arms do not use the same GSM8K rows")
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("MH suffix arms were run in different environments")
    if any(
        hashes != implementation_hashes[0] for hashes in implementation_hashes[1:]
    ):
        raise ValueError("MH suffix arms do not use the same implementation")

    passing = [
        row["arm"]
        for row in table
        if row["arm"] != "uniform"
        and (
            (
                row["paired_vs_uniform"]["accuracy_difference"] >= -0.03125
                and (
                    row["wall_factor_vs_uniform"] <= 0.95
                    or row["main_model_flops_factor_vs_uniform"] <= 0.95
                )
            )
            or (
                row["paired_vs_uniform"]["accuracy_difference"] >= 0.03125
                and row["wall_factor_vs_uniform"] <= 1.05
                and row["main_model_flops_factor_vs_uniform"] <= 1.05
            )
        )
    ]
    decision = {
        "result": "advance_to_confirmation" if passing else "rejected",
        "passing_arms": passing,
        "confirmation_run_required": bool(passing),
        "reason": (
            "at least one nonuniform schedule passed the registered screen gate"
            if passing
            else "no nonuniform schedule met the registered quality-cost gate"
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
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
            "alpha": float(config["mh"]["alpha"]),
            "steps_per_block": int(config["mh"]["steps_per_block"]),
            "block_size": int(config["mh"]["block_size"]),
            "maximum_new_tokens": int(config["generation"]["max_new_tokens"]),
            "environment": environments[0],
        },
        "comparison": (
            "all arms use the same questions, draws, alpha, block size and MH update "
            "count; only the fixed full-support suffix-length distribution changes"
        ),
        "problem_indices": indices["uniform"],
        "table": table,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_mh_suffix_screen(
        config_path=args.config,
        raw_root=args.raw_root,
        tag=args.tag,
        draws=args.draws,
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

"""Post-process a replay benchmark without changing its raw-run provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tomllib
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _wilson(
    correct: int, count: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    proportion = correct / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_quality(
    records: Sequence[dict[str, Any]], *, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    if not records:
        raise ValueError("replay quality comparison requires at least one record")
    fresh = [bool(item["fresh"]["correct"]) for item in records]
    warm = [bool(item["warm_replay"]["correct"]) for item in records]
    deltas = [float(right) - float(left) for left, right in zip(fresh, warm)]
    rng = random.Random(0)
    bootstrap = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    ]
    fresh_correct = sum(fresh)
    warm_correct = sum(warm)
    fresh_interval = _wilson(fresh_correct, len(records))
    warm_interval = _wilson(warm_correct, len(records))
    same_numeric_prediction = sum(
        item["fresh"]["prediction"] == item["warm_replay"]["prediction"]
        for item in records
    )
    return {
        "fresh_correct": fresh_correct,
        "fresh_accuracy": fresh_correct / len(records),
        "fresh_accuracy_wilson_95": list(fresh_interval),
        "warm_correct": warm_correct,
        "warm_accuracy": warm_correct / len(records),
        "warm_accuracy_wilson_95": list(warm_interval),
        "warm_minus_fresh_accuracy": sum(deltas) / len(deltas),
        "warm_minus_fresh_paired_bootstrap_95": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "same_numeric_prediction_count": same_numeric_prediction,
        "same_numeric_prediction_rate": same_numeric_prediction / len(records),
    }


def _minimum_warm_online_uses(
    cache_build_cost: float, fresh_cost: float, warm_online_cost: float
) -> int | None:
    saving_per_use = fresh_cost - warm_online_cost
    if saving_per_use <= 0.0:
        return None
    return max(1, math.ceil(cache_build_cost / saving_per_use))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--tag", default="default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    run_dir = (
        args.results_root
        / str(config["run"]["name"])
        / f"replay-comparison-{args.tag}"
    )
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    source_summary_path = run_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _load_records(records_path)
    report = json.loads(source_summary_path.read_text(encoding="utf-8"))
    fingerprint = str(manifest["fingerprint"])
    if report["manifest_fingerprint"] != fingerprint:
        raise ValueError("replay summary and manifest fingerprints differ")
    if any(item["manifest_fingerprint"] != fingerprint for item in records):
        raise ValueError("a replay record does not belong to the run manifest")
    record_indices = [int(item["problem_index"]) for item in records]
    if record_indices != [int(index) for index in report["problem_indices"]]:
        raise ValueError("replay records and summary contain different problem rows")
    if len(records) != int(report["examples"]):
        raise ValueError("replay record count does not match the source summary")
    if args.limit is not None and len(records) != args.limit:
        raise ValueError("replay record count does not match --limit")

    fresh_flops = float(report["fresh_total_estimated_dense_forward_flops"])
    warm_flops = float(report["warm_online_total_estimated_dense_forward_flops"])
    cache_flops = float(report["cache_build_total_estimated_dense_forward_flops"])
    fresh_seconds = float(report["fresh_total_seconds"])
    warm_seconds = float(report["warm_online_total_seconds"])
    cache_seconds = float(report["cache_build_total_seconds"])
    report["schema_version"] = 4
    report["postprocessing_provenance"] = {
        "postprocessor": "experiments/summarize_gsm8k_replay.py",
        "postprocessor_sha256": _sha256(Path(__file__)),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_records_sha256": _sha256(records_path),
        "source_summary_sha256": _sha256(source_summary_path),
    }
    report["quality_comparison"] = _paired_quality(records)
    report["aggregate_fresh_over_warm_online_wall_time_factor"] = (
        fresh_seconds / warm_seconds
    )
    report["aggregate_fresh_over_warm_one_shot_wall_time_factor"] = (
        fresh_seconds / (cache_seconds + warm_seconds)
    )
    report["cache_amortization"] = {
        "minimum_warm_online_uses_for_flop_break_even": _minimum_warm_online_uses(
            cache_flops, fresh_flops, warm_flops
        ),
        "minimum_warm_online_uses_for_wall_time_break_even": (
            _minimum_warm_online_uses(cache_seconds, fresh_seconds, warm_seconds)
        ),
        "all_candidate_batches_reproduced": all(
            bool(item["warm_replay"]["candidates_reproduced"]) for item in records
        ),
        "definition": (
            "minimum integer q such that cache construction plus q warm-online "
            "evaluations costs no more than q matched fresh-only evaluations"
        ),
        "scope": (
            "the break-even assumes every later evaluation has the same measured "
            "replay-key coverage; it does not apply to unrelated prompts or changed "
            "candidate keys"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

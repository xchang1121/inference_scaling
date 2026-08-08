"""Run and summarize public-GSM8K pass@k/diversity replicates."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

if __package__:
    from experiments.gsm8k_reproduction import IMPLEMENTATION_FILES, _file_sha256
else:
    from gsm8k_reproduction import IMPLEMENTATION_FILES, _file_sha256

PASSK_METHODS = ("base", "mh", "rl_sample")
PASSK_IMPLEMENTATION_FILES = (
    *IMPLEMENTATION_FILES,
    "experiments/gsm8k_passk.py",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _estimated_pass_at_k(correct: int, draws: int, k: int) -> float:
    if draws <= 0 or not 1 <= k <= draws:
        raise ValueError("k must lie between one and the number of draws")
    if draws - correct < k:
        return 1.0
    return 1.0 - math.comb(draws - correct, k) / math.comb(draws, k)


def _summarize_method(
    records_by_draw: list[list[dict[str, Any]]],
    summaries: list[dict[str, Any]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    by_draw = [
        {int(record["problem_index"]): record for record in records}
        for records in records_by_draw
    ]
    index_sets = [set(records) for records in by_draw]
    if not index_sets or len({tuple(sorted(indices)) for indices in index_sets}) != 1:
        raise ValueError("pass@k draws do not contain identical public benchmark rows")
    indices = sorted(index_sets[0])
    pass_at_k = {}
    for k in ks:
        estimates = []
        for index in indices:
            correct = sum(bool(draw[index]["correct"]) for draw in by_draw)
            estimates.append(_estimated_pass_at_k(correct, len(by_draw), k))
        pass_at_k[str(k)] = statistics.fmean(estimates)

    unique_parsed = []
    unparseable = 0
    for index in indices:
        predictions = [draw[index]["prediction"] for draw in by_draw]
        parsed = {prediction for prediction in predictions if prediction is not None}
        unique_parsed.append(len(parsed))
        unparseable += sum(prediction is None for prediction in predictions)

    total_flops = sum(int(summary["estimated_dense_forward_flops"]) for summary in summaries)
    total_slots = sum(int(summary["total_forward_token_slots"]) for summary in summaries)
    total_samples = len(indices) * len(by_draw)
    return {
        "examples": len(indices),
        "draws_per_example": len(by_draw),
        "single_draw_accuracy": sum(
            bool(draw[index]["correct"]) for draw in by_draw for index in indices
        )
        / total_samples,
        "estimated_pass_at_k": pass_at_k,
        "mean_unique_parsed_answers_across_all_draws": statistics.fmean(unique_parsed),
        "unparseable_fraction": unparseable / total_samples,
        "total_forward_token_slots": total_slots,
        "estimated_dense_forward_flops": total_flops,
        "estimated_dense_forward_petaflops": total_flops / 1e15,
        "estimated_dense_flops_per_generated_answer": total_flops / total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--tag", default="passk")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--methods", default=",".join(PASSK_METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.draws <= 0 or args.limit <= 0:
        raise ValueError("draws and limit must be positive")

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(PASSK_METHODS))
    if unknown:
        raise ValueError(f"unsupported pass@k methods: {', '.join(unknown)}")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    profile = str(config["run"]["name"])
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = "src" + (os.pathsep + existing if existing else "")

    if not args.summarize_only:
        for method in methods:
            for draw in range(args.draws):
                draw_tag = f"{args.tag}-draw-{draw}"
                command = [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    "--config",
                    str(args.config),
                    "--method",
                    method,
                    "--tag",
                    draw_tag,
                    "--draw-index",
                    str(draw),
                    "--limit",
                    str(args.limit),
                ]
                print("RUN", subprocess.list2cmdline(command), flush=True)
                subprocess.run(command, check=True, env=environment)

    ks = tuple(k for k in (1, 2, 4, 8, 16, 32) if k <= args.draws)
    table = {}
    for method in methods:
        records_by_draw = []
        summaries = []
        for draw in range(args.draws):
            directory = (
                args.results_root
                / profile
                / f"{method}-{args.tag}-draw-{draw}"
            )
            records_by_draw.append(_load_jsonl(directory / "records.jsonl"))
            summaries.append(
                json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            )
        table[method] = _summarize_method(records_by_draw, summaries, ks)

    report = {
        "schema_version": 1,
        "benchmark": "OpenAI GSM8K official test split",
        "profile": profile,
        "methods": table,
        "implementation_sha256": {
            path: _file_sha256(Path(path)) for path in PASSK_IMPLEMENTATION_FILES
        },
        "pass_at_k_definition": (
            "for n independent draws with c correct answers, each problem contributes "
            "1 - choose(n-c,k)/choose(n,k); the report averages this estimator over "
            "the same fixed public rows"
        ),
        "diversity_scope": (
            "number of distinct parsed final numeric answers; this is not full token-sequence diversity"
        ),
        "compute_definition": (
            "2 * model parameter count * observed padded forward token slots, summed "
            "over every independent draw"
        ),
    }
    output = args.output or Path(f"results/{profile}_{args.tag}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Integrity checks and paired analysis for the Stage-4B Online Uno run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .hf_online_uno import summarize_online_runs
from .stage2_analysis import bootstrap_interval, exact_two_sided_sign_p


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def analyze(
    benchmark: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if benchmark.get("execution_backend") != (
        "huggingface_pytorch_kv_cache_online_fast_residual"
    ):
        raise ValueError("input is not a Stage-4B Online Uno benchmark.")
    runs = benchmark.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("benchmark contains no runs.")
    expected_tokens = int(benchmark["design"]["max_new_tokens"])
    prompts = benchmark["design"]["prompts"]
    pairs: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (int(run["repetition"]), int(run["prompt_index"]))
        label = str(run["label"])
        if label in pairs[key]:
            raise ValueError(f"duplicate {label} run for paired key {key}.")
        pairs[key][label] = run
    labels = list(dict.fromkeys(str(run["label"]) for run in runs))
    required_labels = {"static", "online_s10", "online_s20"}
    if set(labels) != required_labels:
        raise ValueError(f"unexpected method labels: {labels}.")
    if any(set(methods) != required_labels for methods in pairs.values()):
        raise ValueError("every paired key must contain all three methods.")

    output_lengths_pass = all(
        int(run["result"]["metrics"]["output_tokens"]) == expected_tokens
        for run in runs
    )
    finite_pass = _all_finite(runs)
    isolation_records = [
        run["result"]["diagnostics"]["parameter_isolation"]
        for run in runs
        if run["label"] != "static"
    ]
    isolation_pass = all(
        int(record["trainable_base_parameter_tensors"]) == 0
        and int(record["base_optimizer_overlap"]) == 0
        and int(record["fast_trainable_parameters"]) == 526_336
        for record in isolation_records
    )
    routing = benchmark["routing_probe"]
    routing_pass = bool(routing["clean_rows_match"] and routing["noise_rows_changed"])
    safety_pass = output_lengths_pass and finite_pass and isolation_pass and routing_pass

    summary = summarize_online_runs(
        runs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    prompt_breakdown: dict[str, Any] = {}
    for label_index, label in enumerate(("online_s10", "online_s20")):
        per_prompt = {}
        for prompt_index, prompt in enumerate(prompts):
            selected = [
                methods
                for (repetition, index), methods in sorted(pairs.items())
                if index == prompt_index
            ]
            tpf_ratios = np.asarray(
                [
                    methods[label]["result"]["metrics"]["decoder_tokens_per_forward"]
                    / methods["static"]["result"]["metrics"][
                        "decoder_tokens_per_forward"
                    ]
                    for methods in selected
                ],
                dtype=np.float64,
            )
            speed_ratios = np.asarray(
                [
                    methods[label]["result"]["metrics"]["decode_tokens_per_second"]
                    / methods["static"]["result"]["metrics"][
                        "decode_tokens_per_second"
                    ]
                    for methods in selected
                ],
                dtype=np.float64,
            )
            acceptance_deltas = np.asarray(
                [
                    methods[label]["result"]["metrics"]["spec_acceptance_rate"]
                    - methods["static"]["result"]["metrics"]["spec_acceptance_rate"]
                    for methods in selected
                ],
                dtype=np.float64,
            )
            local_seed = bootstrap_seed + 100_000 * label_index + 1_000 * prompt_index
            per_prompt[str(prompt_index)] = {
                "prompt": prompt,
                "pairs": len(selected),
                "paired_tpf_ratio": bootstrap_interval(
                    tpf_ratios,
                    samples=bootstrap_samples,
                    seed=local_seed + 1,
                ),
                "paired_decode_speed_ratio": bootstrap_interval(
                    speed_ratios,
                    samples=bootstrap_samples,
                    seed=local_seed + 2,
                ),
                "paired_acceptance_rate_delta": bootstrap_interval(
                    acceptance_deltas,
                    samples=bootstrap_samples,
                    seed=local_seed + 3,
                ),
            }
        prompt_breakdown[label] = per_prompt

    pairwise_counts = {}
    for label in ("online_s10", "online_s20"):
        speed_wins = 0
        tpf_wins = 0
        for methods in pairs.values():
            online_metrics = methods[label]["result"]["metrics"]
            static_metrics = methods["static"]["result"]["metrics"]
            speed_wins += int(
                online_metrics["decode_tokens_per_second"]
                > static_metrics["decode_tokens_per_second"]
            )
            tpf_wins += int(
                online_metrics["decoder_tokens_per_forward"]
                > static_metrics["decoder_tokens_per_forward"]
            )
        pairwise_counts[label] = {
            "pairs": len(pairs),
            "speed_wins": speed_wins,
            "tpf_wins": tpf_wins,
            "speed_sign_test_two_sided_p": exact_two_sided_sign_p(
                speed_wins,
                len(pairs),
            ),
            "tpf_ties_or_losses": len(pairs) - tpf_wins,
        }

    primary = summary["online_s10"]
    exploratory = summary["online_s20"]
    return {
        "schema_version": 1,
        "input_backend": benchmark["execution_backend"],
        "input_checkpoint": benchmark["checkpoint"],
        "integrity": {
            "runs": len(runs),
            "paired_workloads": len(pairs),
            "expected_output_tokens_per_run": expected_tokens,
            "output_lengths_pass": output_lengths_pass,
            "all_numeric_values_finite": finite_pass,
            "parameter_isolation_records": len(isolation_records),
            "parameter_isolation_pass": isolation_pass,
            "routing_pass": routing_pass,
            "safety_gate_pass": safety_pass,
        },
        "bootstrap": {
            "method": "paired percentile bootstrap of the median",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "summary": summary,
        "prompt_breakdown": prompt_breakdown,
        "pairwise_counts": pairwise_counts,
        "decision": {
            "preregistered_primary": "online_s10",
            "safety_gate_pass": safety_pass,
            "primary_real_model_learning_gate_pass": (
                primary["paired_tpf_ratio"]["ci_95_low"] > 1.0
            ),
            "primary_hf_system_gate_pass": (
                primary["paired_decode_speed_ratio"]["ci_95_low"] > 1.0
            ),
            "primary_hf_system_significant_slowdown": (
                primary["paired_decode_speed_ratio"]["ci_95_high"] < 1.0
            ),
            "exploratory_s20_tpf_interval_excludes_one": not (
                exploratory["paired_tpf_ratio"]["ci_95_low"]
                <= 1.0
                <= exploratory["paired_tpf_ratio"]["ci_95_high"]
            ),
            "exploratory_s20_speed_interval_excludes_one": not (
                exploratory["paired_decode_speed_ratio"]["ci_95_low"]
                <= 1.0
                <= exploratory["paired_decode_speed_ratio"]["ci_95_high"]
            ),
            "official_runtime_tested": False,
            "full_uno_adapter_updated_online": False,
        },
        "scope_warning": (
            "Fifteen pairs cover five seeds on three fixed prompts and one HF fallback "
            "backend. The fast residual is not the full diffusion LoRA, and prompt-level "
            "heterogeneity prevents generalizing the result to arbitrary workloads."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(
        benchmark,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

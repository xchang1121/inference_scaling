"""Paired bootstrap analysis for the Stage-2 Uno checkpoint benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def bootstrap_interval(
    values: np.ndarray,
    *,
    statistic: Callable[..., np.ndarray] = np.median,
    samples: int = 50_000,
    seed: int = 20260905,
) -> dict[str, float]:
    """Return a deterministic percentile interval over paired observations."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires a one-dimensional sample of size >= 2.")
    if samples < 1_000:
        raise ValueError("use at least 1,000 bootstrap resamples.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    resampled = statistic(values[indices], axis=1)
    return {
        "estimate": float(statistic(values)),
        "ci_95_low": float(np.percentile(resampled, 2.5)),
        "ci_95_high": float(np.percentile(resampled, 97.5)),
    }


def exact_two_sided_sign_p(wins: int, pairs: int) -> float:
    """Exact two-sided sign-test p-value under equal win/loss probability."""

    if not 0 <= wins <= pairs or pairs < 1:
        raise ValueError("wins must lie in [0, pairs] and pairs must be positive.")
    tail = min(wins, pairs - wins)
    probability = sum(math.comb(pairs, k) for k in range(tail + 1)) / (2**pairs)
    return min(1.0, 2.0 * probability)


def analyze(
    benchmark: dict[str, object],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    if benchmark.get("execution_backend") != "huggingface_pytorch_kv_cache_fallback":
        raise ValueError("input is not the Stage-2 Hugging Face fallback benchmark.")
    raw_runs = benchmark.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("benchmark contains no runs.")

    pairs: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    for run in raw_runs:
        key = (int(run["repetition"]), int(run["prompt_index"]))
        pairs.setdefault(key, {})[str(run["label"])] = run["metrics"]
    labels = sorted(
        {label for methods in pairs.values() for label in methods if label.startswith("uno_b")},
        key=lambda label: int(label.removeprefix("uno_b")),
    )
    if not labels or any("ar" not in methods for methods in pairs.values()):
        raise ValueError("every pair must contain AR and at least one Uno method.")

    methods: dict[str, object] = {}
    for label_index, label in enumerate(labels):
        if any(label not in pair for pair in pairs.values()):
            raise ValueError(f"method {label} is missing from one or more pairs.")
        ar_tps = np.asarray(
            [pair["ar"]["decode_tokens_per_second"] for pair in pairs.values()],
            dtype=np.float64,
        )
        uno_tps = np.asarray(
            [pair[label]["decode_tokens_per_second"] for pair in pairs.values()],
            dtype=np.float64,
        )
        tpf = np.asarray(
            [pair[label]["decoder_tokens_per_forward"] for pair in pairs.values()],
            dtype=np.float64,
        )
        acceptance = np.asarray(
            [pair[label]["spec_acceptance_rate"] for pair in pairs.values()],
            dtype=np.float64,
        )
        ratios = uno_tps / ar_tps
        deltas = uno_tps - ar_tps
        wins = int(np.count_nonzero(deltas > 0))
        local_seed = seed + 10_000 * label_index
        methods[label] = {
            "block_size": int(label.removeprefix("uno_b")),
            "paired_observations": int(ratios.size),
            "tpf_median_bootstrap": bootstrap_interval(
                tpf,
                samples=bootstrap_samples,
                seed=local_seed + 1,
            ),
            "paired_decode_speedup_median_bootstrap": bootstrap_interval(
                ratios,
                samples=bootstrap_samples,
                seed=local_seed + 2,
            ),
            "paired_decode_tps_delta_median_bootstrap": bootstrap_interval(
                deltas,
                samples=bootstrap_samples,
                seed=local_seed + 3,
            ),
            "spec_acceptance_rate_median_bootstrap": bootstrap_interval(
                acceptance,
                samples=bootstrap_samples,
                seed=local_seed + 4,
            ),
            "paired_speed_wins": wins,
            "paired_speed_losses_or_ties": int(ratios.size - wins),
            "two_sided_sign_test_p": exact_two_sided_sign_p(wins, int(ratios.size)),
        }

    best_label = max(
        labels,
        key=lambda label: methods[label]["paired_decode_speedup_median_bootstrap"][
            "estimate"
        ],
    )
    algorithmic_pass = any(
        method["tpf_median_bootstrap"]["ci_95_low"] > 1.0
        for method in methods.values()
    )
    fallback_speed_pass = any(
        method["paired_decode_speedup_median_bootstrap"]["ci_95_low"] > 1.0
        for method in methods.values()
    )
    return {
        "schema_version": 1,
        "input_backend": benchmark["execution_backend"],
        "input_checkpoint": benchmark["checkpoint"],
        "paired_keys": [
            {"repetition": repetition, "prompt_index": prompt_index}
            for repetition, prompt_index in sorted(pairs)
        ],
        "bootstrap": {
            "method": "paired percentile bootstrap of the median",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "methods": methods,
        "decision": {
            "best_fallback_method": best_label,
            "algorithmic_tpf_reproduction_pass": algorithmic_pass,
            "hf_fallback_wallclock_speedup_pass": fallback_speed_pass,
            "official_runtime_speedup_tested": False,
            "paper_h200_throughput_reproduced": False,
        },
        "scope_warning": (
            "Intervals resample ten seeds/repetitions of one prompt on one HF fallback "
            "backend. They quantify run variation, not prompt/domain generalization, and "
            "are not evidence for Nano-vLLM or H200 throughput."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(
        benchmark,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
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

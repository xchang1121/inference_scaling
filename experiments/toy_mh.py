"""Enumerated smoke experiment for the suffix-MH implementation."""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import prod

from inference_scaling.algorithms.mh import run_mh_chains
from inference_scaling.backends import TabularAutoregressiveBackend
from inference_scaling.config import MHConfig, SamplingConfig
from inference_scaling.metrics import empirical_distribution, total_variation
from inference_scaling.rng import SeedStream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base = (0.65, 0.35)
    alpha = 2.0
    length = 2
    backend = TabularAutoregressiveBackend({}, fallback=base)
    config = MHConfig(
        alpha=alpha,
        total_length=length,
        block_size=length,
        steps_per_block=args.steps,
        chains=args.chains,
    )
    results = run_mh_chains(
        backend,
        (),
        config,
        SamplingConfig(temperature=0.7),
        SeedStream(args.seed),
    )

    exact_weights = {
        sequence: prod(base[token] for token in sequence) ** alpha
        for sequence in product(range(len(base)), repeat=length)
    }
    normalizer = sum(exact_weights.values())
    exact = {sequence: weight / normalizer for sequence, weight in exact_weights.items()}
    empirical = empirical_distribution(result.token_ids for result in results)
    report = {
        "acceptance_rate": sum(result.accepted for result in results)
        / sum(result.attempts for result in results),
        "chains": args.chains,
        "empirical": {"".join(map(str, key)): value for key, value in sorted(empirical.items())},
        "exact": {"".join(map(str, key)): value for key, value in sorted(exact.items())},
        "seed": args.seed,
        "steps": args.steps,
        "total_variation": total_variation(empirical, exact),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


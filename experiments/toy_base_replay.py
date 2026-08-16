"""Finite-state check of the replay correction and data lifecycle."""

from __future__ import annotations

import argparse
import json
from math import exp, log

import numpy as np

from inference_scaling.algorithms.base_replay import (
    ProbabilityObservation,
    corrected_replay_log_weight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100_000)
    args = parser.parse_args()

    base = np.asarray([0.8, 0.2])
    behavior = np.asarray([0.55, 0.45])
    rewards = np.asarray([0.0, 1.0])
    truncation = 1.25
    rng = np.random.default_rng(2026)
    estimates = np.empty(args.trials, dtype=np.float64)
    for index in range(args.trials):
        history_token = int(rng.choice(2, p=behavior))
        fresh_token = int(rng.choice(2, p=base))
        log_weight, _, _ = corrected_replay_log_weight(
            [
                ProbabilityObservation(
                    log(base[history_token]),
                    log(behavior[history_token]),
                    float(rewards[history_token]),
                )
            ],
            [
                ProbabilityObservation(
                    log(base[fresh_token]),
                    log(behavior[fresh_token]),
                    float(rewards[fresh_token]),
                )
            ],
            truncation=truncation,
            reward_temperature=1.0,
        )
        estimates[index] = exp(log_weight)

    exact = float(np.dot(base, np.exp(rewards)))
    report = {
        "absolute_error": abs(float(estimates.mean()) - exact),
        "exact_weight": exact,
        "mean_corrected_estimate": float(estimates.mean()),
        "standard_error": float(estimates.std(ddof=1) / np.sqrt(args.trials)),
        "trials": args.trials,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

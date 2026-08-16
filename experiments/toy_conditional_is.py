"""Enumerated smoke experiment for on-policy and off-policy conditional IS."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import exp

from inference_scaling.algorithms.conditional_is import conditional_is_step
from inference_scaling.backends import TabularAutoregressiveBackend
from inference_scaling.config import ConditionalISConfig, SamplingConfig
from inference_scaling.metrics import total_variation
from inference_scaling.rng import SeedStream


def reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def exact_target() -> dict[int, float]:
    first = (0.7, 0.3)
    second = ((0.9, 0.1), (0.2, 0.8))
    conditional_weights = {
        candidate: sum(
            second[candidate][token] * exp(reward((), (candidate, token)))
            for token in (0, 1)
        )
        for candidate in (0, 1)
    }
    weights = {
        candidate: first[candidate] * conditional_weights[candidate]
        for candidate in (0, 1)
    }
    normalizer = sum(weights.values())
    return {candidate: value / normalizer for candidate, value in weights.items()}


def run(trials: int, proposal_temperature: float) -> dict[str, object]:
    backend = TabularAutoregressiveBackend(
        {(): [0.7, 0.3], (0,): [0.9, 0.1], (1,): [0.2, 0.8]},
        fallback=[0.5, 0.5],
    )
    config = ConditionalISConfig(
        candidate_count=12, rollout_count=8, block_size=1, total_length=2
    )
    counts: Counter[int] = Counter()
    ess_total = 0.0
    for trial in range(trials):
        step = conditional_is_step(
            base_backend=backend,
            rollout_backend=backend,
            prompt=(),
            generated_prefix=(),
            config=config,
            base_sampling=SamplingConfig(),
            rollout_sampling=SamplingConfig(temperature=proposal_temperature),
            reward=reward,
            seeds=SeedStream(50_000 + trial),
            step_index=0,
        )
        counts[step.selected.token_ids[0]] += 1
        for candidate in step.candidates:
            weights = [exp(item.log_weight - candidate.log_weight) for item in candidate.rollouts]
            ess_total += sum(weights) ** 2 / sum(weight * weight for weight in weights)
    empirical = {token: count / trials for token, count in counts.items()}
    exact = exact_target()
    return {
        "average_completion_ess": ess_total / (trials * config.candidate_count),
        "empirical": {str(key): value for key, value in sorted(empirical.items())},
        "proposal_temperature": proposal_temperature,
        "total_variation": total_variation(empirical, exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1800)
    args = parser.parse_args()
    report = {
        "exact": {str(key): value for key, value in sorted(exact_target().items())},
        "off_policy": run(args.trials, 0.55),
        "on_policy": run(args.trials, 1.0),
        "trials": args.trials,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

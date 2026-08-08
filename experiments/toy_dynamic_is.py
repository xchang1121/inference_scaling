"""Finite-state experiment for dynamic candidates and the outer IS ratio."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import exp

from inference_scaling.algorithms.dynamic_is import CandidateProposal, dynamic_is_step
from inference_scaling.backends import TabularAutoregressiveBackend
from inference_scaling.config import DynamicISConfig, SamplingConfig
from inference_scaling.metrics import total_variation
from inference_scaling.replay import BehaviorRegistry, InMemoryReplayStore
from inference_scaling.rng import SeedStream


def reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def exact_target() -> dict[int, float]:
    first = (0.7, 0.3)
    second = ((0.9, 0.1), (0.2, 0.8))
    energies = {
        candidate: sum(
            second[candidate][completion]
            * exp(reward((), (candidate, completion)))
            for completion in (0, 1)
        )
        for candidate in (0, 1)
    }
    weights = {
        candidate: first[candidate] * energies[candidate] for candidate in (0, 1)
    }
    normalizer = sum(weights.values())
    return {candidate: weight / normalizer for candidate, weight in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1200)
    args = parser.parse_args()
    base = TabularAutoregressiveBackend(
        {(): [0.7, 0.3], (0,): [0.9, 0.1], (1,): [0.2, 0.8]},
        fallback=[0.5, 0.5],
        model_id="base",
    )
    auxiliary = CandidateProposal.for_backend(
        TabularAutoregressiveBackend({}, fallback=[0.08, 0.92], model_id="auxiliary"),
        SamplingConfig(),
        label="high-reward-candidate-proposal",
    )
    config = DynamicISConfig(
        candidate_count=32,
        block_size=1,
        total_length=2,
        rollout_budget=64.0,
        auxiliary_mixture=0.75,
    )
    counts: Counter[int] = Counter()
    outer_ess = 0.0
    for trial in range(args.trials):
        step = dynamic_is_step(
            base_backend=base,
            registry=BehaviorRegistry(),
            store=InMemoryReplayStore(),
            prompt=(),
            generated_prefix=(),
            config=config,
            base_sampling=SamplingConfig(),
            reward=reward,
            reward_version="reward-v1",
            seeds=SeedStream(80_000 + trial),
            step_index=0,
            auxiliary_proposal=auxiliary,
        )
        counts[step.selected.token_ids[0]] += 1
        weights = [exp(candidate.log_weight) for candidate in step.candidates]
        outer_ess += sum(weights) ** 2 / sum(weight * weight for weight in weights)
    empirical = {token: count / args.trials for token, count in counts.items()}
    exact = exact_target()
    print(
        json.dumps(
            {
                "average_candidate_ess": outer_ess / args.trials,
                "empirical": {str(key): value for key, value in sorted(empirical.items())},
                "exact": {str(key): value for key, value in sorted(exact.items())},
                "total_variation": total_variation(empirical, exact),
                "trials": args.trials,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

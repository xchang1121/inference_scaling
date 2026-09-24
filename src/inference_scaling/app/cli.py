"""``python -m inference_scaling``: one inference run chosen by four fields.

Every other parameter lives in ``settings/inference.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from inference_scaling.app.rewards import REWARDS
from inference_scaling.app.run import Choices, run
from inference_scaling.app.settings import load_settings
from inference_scaling.datasets import DATASETS

ALGORITHMS = ("sample", "greedy", "beam", "best_of_n", "mh", "reward_mh", "is")
MODELS = ("ar", "dllm")
# Algorithms whose target reweights the base model by exp(reward / temperature) or selects by reward.
REWARD_ALGORITHMS = frozenset({"best_of_n", "reward_mh", "is"})
DEFAULT_REWARD = "vote"


def parse(argv: Sequence[str] | None = None) -> tuple[Choices, Path]:
    parser = argparse.ArgumentParser(
        prog="python -m inference_scaling",
        description="One inference run chosen by four fields; every other parameter lives in settings/inference.json.",
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="is")
    parser.add_argument("--model", choices=MODELS, default="ar", help="model family")
    parser.add_argument("--reward", choices=REWARDS,
                        help=f"reward of {', '.join(sorted(REWARD_ALGORITHMS))} (default: {DEFAULT_REWARD})")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="gsm8k")
    parser.add_argument("--output", type=Path, default=Path("results"), help="root of the results directories")
    args = parser.parse_args(argv)
    reward = args.reward
    if args.algorithm in REWARD_ALGORITHMS:
        reward = reward or DEFAULT_REWARD
    elif reward is not None:
        parser.error(f"--algorithm {args.algorithm} does not use a reward")
    return Choices(args.algorithm, args.model, reward, args.dataset), args.output


def main(argv: Sequence[str] | None = None) -> None:
    choices, output = parse(argv)
    print(json.dumps(run(choices, load_settings(), output), ensure_ascii=False, indent=2))


__all__ = ["ALGORITHMS", "MODELS", "REWARD_ALGORITHMS", "main", "parse"]

"""Resumable orchestrator for the aligned GSM8K experiment matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

DEFAULT_METHODS = (
    "base",
    "beam",
    "best_of_n",
    "mh",
    "conditional_is",
    "conditional_is_small_proposal",
    "rl_sample",
    "rl_greedy",
)


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--tag", default="default")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--with-replay", action="store_true")
    parser.add_argument("--with-async", action="store_true")
    parser.add_argument("--with-ablations", action="store_true")
    parser.add_argument("--with-budget-curve", action="store_true")
    parser.add_argument("--with-length-ablation", action="store_true")
    parser.add_argument("--with-matched-target", action="store_true")
    parser.add_argument("--with-passk", action="store_true")
    parser.add_argument("--ablation-limit", type=int, default=32)
    parser.add_argument("--passk-limit", type=int, default=32)
    parser.add_argument("--passk-draws", type=int, default=8)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        loaded_config = tomllib.load(source)
    configured_length = int(loaded_config["generation"]["max_new_tokens"])
    configured_beams = int(loaded_config["beam"]["num_beams"])
    configured_best_of_n = int(loaded_config["best_of_n"]["samples"])
    configured_candidates = int(loaded_config["conditional_is"]["candidate_count"])
    configured_rollouts = int(loaded_config["conditional_is"]["rollout_count"])
    configured_conditional_block = int(
        loaded_config["conditional_is"]["block_size"]
    )

    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = "src" + (os.pathsep + existing if existing else "")
    common = ["--config", str(args.config), "--tag", args.tag]
    if args.limit is not None:
        common.extend(["--limit", str(args.limit)])

    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    for method in methods:
        _run(
            [
                sys.executable,
                "experiments/gsm8k_reproduction.py",
                *common,
                "--method",
                method,
            ],
            environment,
        )

    if args.with_matched_target:
        for method in (
            "verifier_mh",
            "verifier_conditional_is",
            "verifier_conditional_is_small_proposal",
        ):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *common,
                    "--method",
                    method,
                ],
                environment,
            )

    if args.with_replay:
        replay_output = f"results/{args.config.stem}_replay_{args.tag}.json"
        _run(
            [
                sys.executable,
                "experiments/gsm8k_replay_benchmark.py",
                *common,
                "--aggregate-output",
                replay_output,
            ],
            environment,
        )
        _run(
            [
                sys.executable,
                "experiments/summarize_gsm8k_replay.py",
                *common,
                "--output",
                replay_output,
            ],
            environment,
        )
    if args.with_async:
        async_limit = args.limit if args.limit is not None else min(args.ablation_limit, 32)
        _run(
            [
                sys.executable,
                "experiments/gsm8k_async_benchmark.py",
                "--config",
                str(args.config),
                "--limit",
                str(async_limit),
                "--output",
                f"results/{args.config.stem}_async_grouped_{args.tag}.json",
            ],
            environment,
        )

    if args.with_passk:
        _run(
            [
                sys.executable,
                "experiments/gsm8k_passk.py",
                "--config",
                str(args.config),
                "--tag",
                f"{args.tag}-passk",
                "--limit",
                str(args.passk_limit),
                "--draws",
                str(args.passk_draws),
            ],
            environment,
        )

    if args.with_ablations:
        ablation_common = [
            "--config",
            str(args.config),
            "--limit",
            str(args.ablation_limit),
        ]
        for alpha in (1.0, 2.0, 4.0, 8.0):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "mh",
                    "--tag",
                    f"alpha-{alpha:g}",
                    "--mh-alpha",
                    str(alpha),
                ],
                environment,
            )
        for steps in (1, 2, 5, 10):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "mh",
                    "--tag",
                    f"steps-{steps}",
                    "--mh-steps",
                    str(steps),
                ],
                environment,
            )
        _run(
            [
                sys.executable,
                "experiments/gsm8k_reproduction.py",
                *ablation_common,
                "--method",
                "conditional_is",
                "--tag",
                "conditional-reference",
                "--candidate-count",
                str(configured_candidates),
                "--rollout-count",
                str(configured_rollouts),
                "--block-size",
                str(configured_conditional_block),
                "--sampling-temperature",
                "1.0",
            ],
            environment,
        )
        for tag, clip in (
            ("conditional-small-proposal-reference", "10"),
            ("conditional-small-proposal-unclipped", "none"),
        ):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is_small_proposal",
                    "--tag",
                    tag,
                    "--candidate-count",
                    str(configured_candidates),
                    "--rollout-count",
                    str(configured_rollouts),
                    "--block-size",
                    str(configured_conditional_block),
                    "--sampling-temperature",
                    "1.0",
                    "--importance-log-ratio-clip",
                    clip,
                ],
                environment,
            )
        _run(
            [
                sys.executable,
                "experiments/gsm8k_reproduction.py",
                *ablation_common,
                "--method",
                "best_of_n",
                "--tag",
                "best-of-n-reference",
                "--best-of-n-samples",
                str(configured_best_of_n),
                "--sampling-temperature",
                "1.0",
            ],
            environment,
        )
        for candidates in (3, 5, 10):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is",
                    "--tag",
                    f"candidates-{candidates}-rollouts-3",
                    "--candidate-count",
                    str(candidates),
                    "--rollout-count",
                    "3",
                ],
                environment,
            )
        for rollouts in (1, 5):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is",
                    "--tag",
                    f"candidates-10-rollouts-{rollouts}",
                    "--candidate-count",
                    "10",
                    "--rollout-count",
                    str(rollouts),
                ],
                environment,
            )
        for guidance_steps in (2, 8, 16):
            block_size = configured_length // guidance_steps
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    "conditional_is",
                    "--tag",
                    f"guidance-steps-{guidance_steps}",
                    "--max-new-tokens",
                    str(configured_length),
                    "--block-size",
                    str(block_size),
                    "--candidate-count",
                    str(configured_candidates),
                    "--rollout-count",
                    str(configured_rollouts),
                ],
                environment,
            )
        for reward_source in (
            "log_probability",
            "negative_entropy",
            "self_certainty",
            "exact",
        ):
            for method in (
                "best_of_n",
                "conditional_is",
                "conditional_is_small_proposal",
            ):
                command = [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    method,
                    "--tag",
                    f"{method}-reward-{reward_source}",
                    "--conditional-reward",
                    reward_source,
                ]
                if method == "best_of_n":
                    command.extend(
                        [
                            "--best-of-n-samples",
                            str(configured_best_of_n),
                            "--sampling-temperature",
                            "1.0",
                        ]
                    )
                else:
                    command.extend(
                        [
                            "--candidate-count",
                            str(configured_candidates),
                            "--rollout-count",
                            str(configured_rollouts),
                            "--block-size",
                            str(configured_conditional_block),
                            "--sampling-temperature",
                            "1.0",
                        ]
                    )
                _run(command, environment)
        for temperature in (0.7, 1.5):
            for method in (
                "beam",
                "best_of_n",
                "conditional_is",
                "conditional_is_small_proposal",
            ):
                command = [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *ablation_common,
                    "--method",
                    method,
                    "--tag",
                    f"{method}-temperature-{temperature:g}",
                    "--sampling-temperature",
                    str(temperature),
                ]
                if method == "beam":
                    command.extend(["--num-beams", str(configured_beams)])
                elif method == "best_of_n":
                    command.extend(
                        ["--best-of-n-samples", str(configured_best_of_n)]
                    )
                else:
                    command.extend(
                        [
                            "--candidate-count",
                            str(configured_candidates),
                            "--rollout-count",
                            str(configured_rollouts),
                            "--block-size",
                            str(configured_conditional_block),
                        ]
                    )
                _run(command, environment)

    if args.with_budget_curve:
        curve_common = [
            "--config",
            str(args.config),
            "--limit",
            str(args.ablation_limit),
        ]
        _run(
            [
                sys.executable,
                "experiments/gsm8k_reproduction.py",
                *curve_common,
                "--method",
                "beam",
                "--tag",
                "beam-reference",
                "--num-beams",
                str(configured_beams),
            ],
            environment,
        )
        _run(
            [
                sys.executable,
                "experiments/gsm8k_reproduction.py",
                *curve_common,
                "--method",
                "best_of_n",
                "--tag",
                "best-of-n-reference",
                "--best-of-n-samples",
                str(configured_best_of_n),
                "--sampling-temperature",
                "1.0",
            ],
            environment,
        )
        for method, tag in (
            ("conditional_is", "conditional-reference"),
            (
                "conditional_is_small_proposal",
                "conditional-small-proposal-reference",
            ),
        ):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *curve_common,
                    "--method",
                    method,
                    "--tag",
                    tag,
                    "--candidate-count",
                    str(configured_candidates),
                    "--rollout-count",
                    str(configured_rollouts),
                    "--block-size",
                    str(configured_conditional_block),
                    "--sampling-temperature",
                    "1.0",
                ],
                environment,
            )
        for beams in (4, 8):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *curve_common,
                    "--method",
                    "beam",
                    "--tag",
                    f"budget-beam-{beams}",
                    "--num-beams",
                    str(beams),
                ],
                environment,
            )
        for samples in (4, 8):
            _run(
                [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *curve_common,
                    "--method",
                    "best_of_n",
                    "--tag",
                    f"budget-best-of-n-{samples}",
                    "--best-of-n-samples",
                    str(samples),
                ],
                environment,
            )
        for candidates in (3, 5, 10):
            for method in ("conditional_is", "conditional_is_small_proposal"):
                _run(
                    [
                        sys.executable,
                        "experiments/gsm8k_reproduction.py",
                        *curve_common,
                        "--method",
                        method,
                        "--tag",
                        f"budget-{method}-m{candidates}-k3",
                        "--candidate-count",
                        str(candidates),
                        "--rollout-count",
                        "3",
                    ],
                    environment,
                )

    if args.with_length_ablation:
        length_common = [
            "--config",
            str(args.config),
            "--limit",
            str(args.ablation_limit),
        ]
        for length in (128, 256, 512):
            block_size = max(1, length // 4)
            for method in (
                "base",
                "best_of_n",
                "beam",
                "conditional_is",
                "conditional_is_small_proposal",
                "rl_greedy",
            ):
                command = [
                    sys.executable,
                    "experiments/gsm8k_reproduction.py",
                    *length_common,
                    "--method",
                    method,
                    "--tag",
                    f"length-{length}",
                    "--max-new-tokens",
                    str(length),
                ]
                if method.startswith("conditional_is"):
                    command.extend(["--block-size", str(block_size)])
                _run(command, environment)


if __name__ == "__main__":
    main()

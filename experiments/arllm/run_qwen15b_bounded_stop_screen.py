"""Resumable Qwen2.5-1.5B exact bounded-rollout stopping study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.arllm.summarize_qwen15b_bounded_stop import BOUNDED_STOP_ARMS
from experiments.shared.suite_runner import run_manifested_commands


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    reproduction = REPOSITORY_ROOT / "experiments" / "arllm" / "gsm8k_reproduction.py"
    summarizer = (
        REPOSITORY_ROOT
        / "experiments"
        / "arllm"
        / "summarize_qwen15b_bounded_stop.py"
    )
    commands: list[list[str]] = []
    for draw in range(args.draws):
        arms = (
            BOUNDED_STOP_ARMS
            if draw % 2 == 0
            else tuple(reversed(BOUNDED_STOP_ARMS))
        )
        for arm, early_stop in arms:
            command = [
                sys.executable,
                str(reproduction),
                "--config",
                str(args.config),
                "--method",
                "conditional_is",
                "--conditional-reward",
                "frozen_consensus",
                "--rollout-count",
                str(args.rollout_count),
                "--draw-index",
                str(draw),
                "--limit",
                str(args.limit),
                "--tag",
                f"{args.tag}-{arm}-draw{draw}",
                "--output-root",
                str(args.raw_root),
            ]
            if early_stop:
                command.extend(
                    [
                        "--exact-rollout-early-stop",
                        "--rollout-log-weight-lower",
                        str(args.log_weight_lower),
                        "--rollout-log-weight-upper",
                        str(args.log_weight_upper),
                        "--rollout-evaluation-batch-size",
                        str(args.evaluation_batch_size),
                    ]
                )
            commands.append(command)
    commands.append(
        [
            sys.executable,
            str(summarizer),
            "--config",
            str(args.config),
            "--raw-root",
            str(args.raw_root),
            "--tag",
            args.tag,
            "--draws",
            str(args.draws),
            "--questions",
            str(args.limit),
            "--rollout-count",
            str(args.rollout_count),
            "--evaluation-batch-size",
            str(args.evaluation_batch_size),
            "--log-weight-lower",
            str(args.log_weight_lower),
            "--log-weight-upper",
            str(args.log_weight_upper),
            "--phase",
            args.phase,
            "--output",
            str(args.output),
        ]
    )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsm8k_quick.toml"),
    )
    parser.add_argument("--tag", default="qwen15b-bounded-stop-screen")
    parser.add_argument("--phase", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rollout-count", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--log-weight-lower", type=float, default=0.0)
    parser.add_argument("--log-weight-upper", type=float, default=10.0)
    parser.add_argument("--raw-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    with args.config.open("rb") as source:
        config = tomllib.load(source)
    configured = int(config["run"]["sample_count"])
    args.limit = configured if args.limit is None else args.limit
    if args.draws <= 0 or args.rollout_count <= 0 or args.evaluation_batch_size <= 0:
        raise ValueError("draws, rollout-count and evaluation-batch-size must be positive")
    if args.evaluation_batch_size > args.rollout_count:
        raise ValueError("evaluation-batch-size cannot exceed rollout-count")
    if args.log_weight_lower > args.log_weight_upper:
        raise ValueError("log-weight bounds must be ordered")
    if args.phase == "screen" and (args.limit != 8 or args.draws != 2):
        raise ValueError("the registered screen requires 8 questions and 2 draws")
    if args.phase == "confirmation" and (args.limit != 32 or args.draws != 4):
        raise ValueError(
            "the registered confirmation requires 32 questions and 4 draws"
        )
    if args.output is None:
        filename = (
            "bounded_stop_screen.json"
            if args.phase == "screen"
            else "bounded_stop_confirmation.json"
        )
        args.output = Path("results/arllm/qwen15b_optimization") / filename
    commands = build_commands(args)
    manifest = args.output.with_name(f"{args.tag}_manifest.json")
    run_manifested_commands(
        commands=commands,
        root=REPOSITORY_ROOT,
        manifest_path=manifest,
        metadata={
            "study": "qwen15b_exact_bounded_rollout_stopping",
            "phase": args.phase,
            "model": "Qwen2.5-1.5B-Instruct",
            "dllm_experiments": False,
            "tag": args.tag,
            "draws": args.draws,
            "questions": args.limit,
            "rollouts_per_candidate": args.rollout_count,
            "evaluation_batch_size": args.evaluation_batch_size,
            "log_weight_bounds": [args.log_weight_lower, args.log_weight_upper],
            "reward": "independent-pilot frozen consensus",
            "execution_order": "full then bounded on even draws; reverse on odd draws",
        },
        dry_run=args.dry_run,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()

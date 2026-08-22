"""Resumable Qwen2.5-1.5B MH suffix-schedule screening entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.arllm.summarize_qwen15b_mh_suffix import MH_SUFFIX_ARMS
from experiments.shared.suite_runner import run_manifested_commands


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    reproduction = REPOSITORY_ROOT / "experiments" / "arllm" / "gsm8k_reproduction.py"
    summarizer = (
        REPOSITORY_ROOT
        / "experiments"
        / "arllm"
        / "summarize_qwen15b_mh_suffix.py"
    )
    commands: list[list[str]] = []
    for draw in range(args.draws):
        if getattr(args, "phase", "screen") == "confirmation":
            offset = draw % len(MH_SUFFIX_ARMS)
            arms = MH_SUFFIX_ARMS[offset:] + MH_SUFFIX_ARMS[:offset]
        else:
            arms = (
                MH_SUFFIX_ARMS
                if draw % 2 == 0
                else tuple(reversed(MH_SUFFIX_ARMS))
            )
        for arm, schedule in arms:
            commands.append(
                [
                    sys.executable,
                    str(reproduction),
                    "--config",
                    str(args.config),
                    "--method",
                    "mh",
                    "--mh-suffix-schedule",
                    schedule,
                    "--draw-index",
                    str(draw),
                    "--limit",
                    str(args.limit),
                    "--tag",
                    f"{args.tag}-{arm}-draw{draw}",
                    "--output-root",
                    str(args.raw_root),
                ]
            )
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
            "--phase",
            getattr(args, "phase", "screen"),
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
    parser.add_argument("--tag", default="qwen15b-mh-suffix-screen")
    parser.add_argument(
        "--phase",
        choices=("screen", "confirmation"),
        default="screen",
    )
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--raw-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    with args.config.open("rb") as source:
        config = tomllib.load(source)
    configured = int(config["run"]["sample_count"])
    args.limit = configured if args.limit is None else args.limit
    if args.draws <= 0:
        raise ValueError("draws must be positive")
    if args.phase == "screen" and (args.limit != 8 or args.draws != 2):
        raise ValueError("the registered screen requires 8 questions and 2 draws")
    if args.phase == "confirmation" and (args.limit != 32 or args.draws != 4):
        raise ValueError(
            "the registered confirmation requires 32 questions and 4 draws"
        )
    if args.output is None:
        filename = (
            "mh_suffix_screen.json"
            if args.phase == "screen"
            else "mh_suffix_confirmation.json"
        )
        args.output = Path("results/arllm/qwen15b_optimization") / filename
    commands = build_commands(args)
    manifest = args.output.with_name(f"{args.tag}_manifest.json")
    run_manifested_commands(
        commands=commands,
        root=REPOSITORY_ROOT,
        manifest_path=manifest,
        metadata={
            "study": "qwen15b_mh_suffix_schedule",
            "phase": args.phase,
            "model": "Qwen2.5-1.5B-Instruct",
            "dllm_experiments": False,
            "tag": args.tag,
            "draws": args.draws,
            "questions": args.limit,
            "execution_order": (
                "cyclic Latin order across draws"
                if args.phase == "confirmation"
                else "forward arms on even draws; reverse arms on odd draws"
            ),
        },
        dry_run=args.dry_run,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()

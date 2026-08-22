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
        arms = MH_SUFFIX_ARMS if draw % 2 == 0 else tuple(reversed(MH_SUFFIX_ARMS))
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
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--raw-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/arllm/qwen15b_optimization/mh_suffix_screen.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    with args.config.open("rb") as source:
        config = tomllib.load(source)
    configured = int(config["run"]["sample_count"])
    args.limit = configured if args.limit is None else args.limit
    if args.limit != configured:
        raise ValueError(
            "the tracked screen summary requires the configured question count"
        )
    if args.draws <= 0:
        raise ValueError("draws must be positive")
    commands = build_commands(args)
    manifest = args.output.with_name(f"{args.tag}_manifest.json")
    run_manifested_commands(
        commands=commands,
        root=REPOSITORY_ROOT,
        manifest_path=manifest,
        metadata={
            "study": "qwen15b_mh_suffix_schedule_screen",
            "model": "Qwen2.5-1.5B-Instruct",
            "dllm_experiments": False,
            "tag": args.tag,
            "draws": args.draws,
            "questions": args.limit,
            "execution_order": "forward arms on even draws; reverse arms on odd draws",
        },
        dry_run=args.dry_run,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()

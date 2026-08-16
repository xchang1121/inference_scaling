"""Single resumable entry point for the paired LLaDA experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.gsm8k_reproduction import METHODS
from experiments.shared.paired_protocol import load_pairing
from experiments.shared.suite_runner import run_manifested_commands

DEFAULT_METHODS = tuple(method for method in METHODS if not method.startswith("vrpo_"))
ALIGNED_METHODS = ("vrpo_sample", "vrpo_greedy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--tag")
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dllm/gsm8k")
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--draw-index", type=int, default=0)
    parser.add_argument("--with-aligned", action="store_true")
    parser.add_argument(
        "--vrpo",
        choices=("skip", "preflight", "train"),
        default="preflight",
        help=(
            "skip VRPO, run a CPU-only implementation preflight, or prepare "
            "preferences and train the resumable adapter before evaluation"
        ),
    )
    parser.add_argument(
        "--with-replay", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    methods = list(args.methods or DEFAULT_METHODS)
    include_aligned = args.with_aligned or args.vrpo == "train"
    if include_aligned and args.vrpo != "train":
        adapter = Path(str(config["alignment"]["adapter"]))
        if not adapter.is_dir():
            raise FileNotFoundError(
                f"aligned LLaDA adapter is absent: {adapter}; run the VRPO stage first"
            )
    if include_aligned:
        for method in ALIGNED_METHODS:
            if method not in methods:
                methods.append(method)
    elif any(method in ALIGNED_METHODS for method in methods):
        raise ValueError("aligned methods require --with-aligned")

    tag = args.tag or f"llada-{args.profile}"
    root = REPOSITORY_ROOT
    runner = root / "experiments" / "dllm" / "gsm8k_reproduction.py"
    replay_runner = root / "experiments" / "dllm" / "gsm8k_replay_benchmark.py"
    prepare_vrpo = root / "experiments" / "dllm" / "prepare_gsm8k_vrpo.py"
    train_vrpo = root / "experiments" / "dllm" / "train_gsm8k_vrpo.py"
    common = [
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--output-root",
        str(args.output_root),
        "--tag",
        tag,
        "--profile",
        args.profile,
    ]
    if args.limit is not None:
        common.extend(("--limit", str(args.limit)))

    commands: list[list[str]] = []
    if args.vrpo == "preflight":
        commands.append(
            [sys.executable, str(train_vrpo), "--config", str(args.config), "--preflight"]
        )
    elif args.vrpo == "train":
        commands.extend(
            (
                [
                    sys.executable,
                    str(prepare_vrpo),
                    "--config",
                    str(args.config),
                    "--profile",
                    "full",
                ],
                [
                    sys.executable,
                    str(train_vrpo),
                    "--config",
                    str(args.config),
                    "--resume",
                    "auto",
                ],
            )
        )
    commands.extend([
        [
            sys.executable,
            str(runner),
            *common,
            "--method",
            method,
            "--draw-index",
            str(args.draw_index),
        ]
        for method in methods
    ])
    if args.with_replay:
        commands.append([sys.executable, str(replay_runner), *common])

    run_manifested_commands(
        commands=commands,
        root=root,
        manifest_path=args.output_root / tag / "suite_manifest.json",
        metadata={
            "family": "dllm",
            "profile": args.profile,
            "tag": tag,
            "methods": methods,
            "with_replay": args.with_replay,
            "with_aligned": include_aligned,
            "vrpo": args.vrpo,
        },
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

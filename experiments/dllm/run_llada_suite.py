"""Single resumable entry point for the paired LLaDA experiment suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from experiments.dllm.gsm8k_reproduction import METHODS
from experiments.shared.paired_protocol import load_pairing

DEFAULT_METHODS = tuple(method for method in METHODS if not method.startswith("vrpo_"))
ALIGNED_METHODS = ("vrpo_sample", "vrpo_greedy")


def _command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


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
    root = Path(__file__).resolve().parents[2]
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

    suite_dir = args.output_root / tag
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "suite_manifest.json"
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "tag": tag,
        "methods": methods,
        "with_replay": args.with_replay,
        "with_aligned": include_aligned,
        "vrpo": args.vrpo,
        "commands": [_command_text(command) for command in commands],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_commands": 0,
        "status": "dry_run" if args.dry_run else "running",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for command in commands:
        print(_command_text(command), flush=True)
    if args.dry_run:
        return

    environment = os.environ.copy()
    python_paths = (str(root / "src"), str(root))
    environment["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    try:
        for index, command in enumerate(commands, start=1):
            subprocess.run(command, cwd=root, env=environment, check=True)
            manifest["completed_commands"] = index
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except BaseException:
        manifest["status"] = "failed"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

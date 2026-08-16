"""Unified CLI for AR-LLM and diffusion-LLM reproduction pipelines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.arllm.run_arllm_suite import AR_METHODS, COMPONENTS
from experiments.dllm.gsm8k_reproduction import METHODS as DLLM_METHODS


def _command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _default_python(environment_variable: str) -> str:
    """Select a family interpreter from the environment or current process."""

    return os.environ.get(environment_variable) or sys.executable


def build_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    if args.family in {"arllm", "both"}:
        command = [
            args.ar_python,
            str(root / "experiments" / "arllm" / "run_arllm_suite.py"),
            "--stage",
            args.stage,
            "--profile",
            args.profile,
            "--tag",
            f"{args.tag}-arllm",
            "--training-config",
            str(args.ar_training_config),
            "--summary-root",
            str(args.output_root / "arllm"),
            "--methods",
            *args.ar_methods,
            "--components",
            *args.components,
        ]
        if args.ar_config is not None:
            command.extend(("--config", str(args.ar_config)))
        if args.backend is not None:
            command.extend(("--backend", args.backend))
        for flag, value in (
            ("--limit", args.limit),
            ("--train-limit", args.train_limit),
            ("--training-output", args.ar_training_output),
            ("--max-train-steps", args.max_train_steps),
            ("--num-generations", args.num_generations),
            ("--max-completion-length", args.max_completion_length),
            ("--ablation-limit", args.ablation_limit),
            ("--passk-limit", args.passk_limit),
            ("--passk-draws", args.passk_draws),
        ):
            if value is not None:
                command.extend((flag, str(value)))
        if args.dry_run:
            command.append("--dry-run")
        commands.append(command)

    if args.family in {"dllm", "both"}:
        dllm_config = str(args.dllm_config)
        if args.stage == "prepare":
            commands.append(
                [
                    args.dllm_python,
                    str(root / "experiments" / "dllm" / "download_llada.py"),
                    "--config",
                    dllm_config,
                ]
            )
        elif args.stage == "train":
            if args.profile == "smoke":
                commands.append(
                    [
                        args.dllm_python,
                        str(root / "experiments" / "dllm" / "train_gsm8k_vrpo.py"),
                        "--config",
                        dllm_config,
                        "--preflight",
                    ]
                )
            else:
                commands.extend(
                    (
                        [
                            args.dllm_python,
                            str(root / "experiments" / "dllm" / "prepare_gsm8k_vrpo.py"),
                            "--config",
                            dllm_config,
                            "--profile",
                            "full",
                        ],
                        [
                            args.dllm_python,
                            str(root / "experiments" / "dllm" / "train_gsm8k_vrpo.py"),
                            "--config",
                            dllm_config,
                            "--resume",
                            "auto",
                        ],
                    )
                )
        else:
            if args.stage == "all":
                commands.append(
                    [
                        args.dllm_python,
                        str(root / "experiments" / "dllm" / "download_llada.py"),
                        "--config",
                        dllm_config,
                    ]
                )
            vrpo = "skip"
            if args.stage == "all":
                vrpo = "preflight" if args.profile == "smoke" else "train"
            command = [
                args.dllm_python,
                str(root / "experiments" / "dllm" / "run_llada_suite.py"),
                "--config",
                dllm_config,
                "--profile",
                args.profile,
                "--tag",
                f"{args.tag}-dllm",
                "--output-root",
                str(args.output_root / "dllm"),
                "--vrpo",
                vrpo,
                "--methods",
                *args.dllm_methods,
            ]
            if args.limit is not None:
                command.extend(("--limit", str(args.limit)))
            if (
                any(method.startswith("vrpo_") for method in args.dllm_methods)
                and vrpo != "train"
            ):
                command.append("--with-aligned")
            if "replay" not in args.components:
                command.append("--no-with-replay")
            if args.dry_run:
                command.append("--dry-run")
            commands.append(command)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("arllm", "dllm", "both"), default="both")
    parser.add_argument("--stage", choices=("prepare", "train", "inference", "all"), default="inference")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--tag", default="reproduction")
    parser.add_argument("--ar-config", type=Path)
    parser.add_argument("--ar-training-config", type=Path, default=Path("configs/gsm8k_grpo.toml"))
    parser.add_argument(
        "--dllm-config",
        type=Path,
        default=Path("configs/gsm8k_llada_moe_3090.toml"),
    )
    parser.add_argument("--ar-methods", nargs="+", choices=AR_METHODS)
    parser.add_argument("--dllm-methods", nargs="+", choices=DLLM_METHODS)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS)
    parser.add_argument("--backend", choices=("transformers", "vllm", "vllm-sync"))
    parser.add_argument(
        "--ar-python",
        default=_default_python("AR_PYTHON"),
        help="AR executable name/path; defaults to AR_PYTHON, then current Python",
    )
    parser.add_argument(
        "--dllm-python",
        default=_default_python("DLLM_PYTHON"),
        help="dLLM executable name/path; defaults to DLLM_PYTHON, then current Python",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--ar-training-output", type=Path)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--ablation-limit", type=int)
    parser.add_argument("--passk-limit", type=int)
    parser.add_argument("--passk-draws", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("results/reproduction"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.ar_methods = tuple(args.ar_methods or AR_METHODS)
    args.dllm_methods = tuple(
        args.dllm_methods
        or (method for method in DLLM_METHODS if not method.startswith("vrpo_"))
    )
    args.components = tuple(
        args.components
        or (("quality", "replay") if args.profile == "smoke" else COMPONENTS[:-1])
    )
    root = REPOSITORY_ROOT
    commands = build_commands(args, root)
    manifest_dir = args.output_root / args.tag
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "reproduction_manifest.json"
    manifest = {
        "schema_version": 1,
        "family": args.family,
        "stage": args.stage,
        "profile": args.profile,
        "tag": args.tag,
        "ar_methods": args.ar_methods,
        "dllm_methods": args.dllm_methods,
        "components": args.components,
        "python_executables": {
            "controller": sys.executable,
            "arllm": args.ar_python,
            "dllm": args.dllm_python,
        },
        "commands": [_command_text(command) for command in commands],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_commands": 0,
        "status": "dry_run" if args.dry_run else "running",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for command in commands:
        print(_command_text(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    if args.dry_run:
        return
    try:
        for index, command in enumerate(commands, start=1):
            subprocess.run(command, cwd=root, env=environment, check=True)
            manifest["completed_commands"] = index
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException:
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

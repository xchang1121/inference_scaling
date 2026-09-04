"""Collect a small, reproducible host manifest without changing the machine."""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(command: Sequence[str], cwd: Path | None = None) -> CommandResult:
    """Run a read-only probe and retain enough information to diagnose failures."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return CommandResult(list(command), 127, "", str(error))
    return CommandResult(
        list(command),
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def parse_nvidia_smi_csv(text: str) -> list[dict[str, object]]:
    """Parse the stable no-header CSV emitted by the preflight GPU query."""

    rows: list[dict[str, object]] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 5:
            continue
        index, name, memory_mib, driver, compute_capability = (
            value.strip() for value in row
        )
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_mib.split()[0])
        except ValueError:
            continue
        rows.append(
            {
                "index": parsed_index,
                "name": name,
                "memory_total_mib": parsed_memory,
                "driver_version": driver,
                "compute_capability": compute_capability,
            }
        )
    return rows


def system_memory_bytes() -> int | None:
    """Return physical RAM using the standard Windows API or POSIX sysconf."""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_torch() -> dict[str, object]:
    try:
        import torch
    except Exception as error:  # pragma: no cover - depends on the host environment
        return {"importable": False, "error": type(error).__name__}

    available = bool(torch.cuda.is_available())
    result: dict[str, object] = {
        "importable": True,
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": available,
    }
    if available:
        result.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_0": torch.cuda.get_device_name(0),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return result


def collect_git(repo_root: Path) -> dict[str, object]:
    probes = {
        "revision": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
        "origin": ["git", "remote", "get-url", "origin"],
    }
    results = {name: run_command(command, repo_root) for name, command in probes.items()}
    return {
        "root": str(repo_root.resolve()),
        "revision": results["revision"].stdout or None,
        "branch": results["branch"].stdout or None,
        "origin": results["origin"].stdout or None,
        "working_tree_clean": results["status"].returncode == 0
        and not results["status"].stdout,
        "status_entries": results["status"].stdout.splitlines(),
    }


def collect_manifest(repo_root: Path, upstream_commit: str) -> dict[str, object]:
    gpu_probe = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    wsl_probe = run_command(["wsl.exe", "--list", "--quiet"])
    nvcc_probe = run_command(["nvcc", "--version"])
    memory = system_memory_bytes()
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "datasets",
            "triton",
            "flash-attn",
            "vllm",
        )
    }
    torch_info = collect_torch()
    python_310 = sys.version_info[:2] == (3, 10)
    official_runtime_ready = (
        platform.system() == "Linux"
        and python_310
        and packages["torch"] == "2.11.0"
        and packages["triton"] == "3.6.0"
        and packages["flash-attn"] == "2.8.3"
        and bool(torch_info.get("cuda_available"))
    )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": memory,
            "physical_memory_gib": round(memory / 2**30, 3) if memory else None,
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "gpu": {
            "probe": asdict(gpu_probe),
            "devices": parse_nvidia_smi_csv(gpu_probe.stdout),
        },
        "torch": torch_info,
        "packages": packages,
        "toolchain": {
            "nvcc": asdict(nvcc_probe),
            "git": asdict(run_command(["git", "--version"])),
        },
        "wsl": {
            "executable_present": shutil.which("wsl.exe") is not None,
            "ready": wsl_probe.returncode == 0,
            "distribution_names": [
                line.strip("\x00 \t")
                for line in wsl_probe.stdout.splitlines()
                if line.strip("\x00 \t")
            ],
            "returncode": wsl_probe.returncode,
        },
        "repository": collect_git(repo_root),
        "upstream": {
            "uno_repository": "https://github.com/ifm-ai/uno",
            "pinned_commit": upstream_commit,
            "paper": "https://arxiv.org/abs/2609.04010v1",
        },
        "readiness": {
            "official_uno_runtime_ready": official_runtime_ready,
            "prototype_cuda_ready": bool(torch_info.get("cuda_available")),
            "official_runtime_requirements": {
                "platform": "Linux x86_64",
                "python": ">=3.10,<3.11",
                "torch": "2.11.0 (cu128)",
                "triton": "3.6.0",
                "flash_attn": "2.8.3",
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository whose revision is recorded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON manifest.",
    )
    parser.add_argument(
        "--upstream-commit",
        default="ed2ee36bb7a3aea8732ebc635b3f09490a032ea3",
        help="Audited ifm-ai/uno revision.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = collect_manifest(args.repo_root, args.upstream_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

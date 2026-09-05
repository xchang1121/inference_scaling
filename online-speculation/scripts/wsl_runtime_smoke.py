"""Verify the pinned Uno CUDA/FA2 runtime and write an auditable manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED = {
    "python": (3, 10),
    "torch": "2.11.0",
    "torch_cuda": "12.8",
    "triton": "3.6.0",
    "flash_attn": "2.8.3",
    "transformers": "4.55.0",
    "compute_capability": (8, 6),
    "uno_commit": "ed2ee36bb7a3aea8732ebc635b3f09490a032ea3",
}


def run_text(command: list[str]) -> dict[str, Any]:
    """Run a read-only inventory command and retain its exact outcome."""
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout.strip(),
    }


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uno-source", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--flash-wheel-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected": EXPECTED,
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "uname": list(os.uname()),
        },
        "paths": {
            "uno_source": str(args.uno_source.resolve()),
            "base_model": str(args.base_model.resolve()),
            "adapter": str(args.adapter.resolve()),
        },
        "flash_wheel_sha256": args.flash_wheel_sha256,
        "commands": {
            "os_release": run_text(["cat", "/etc/os-release"]),
            "nvidia_smi": run_text(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "uno_revision": run_text(
                ["git", "-C", str(args.uno_source), "rev-parse", "HEAD"]
            ),
            "pip_freeze": run_text([sys.executable, "-m", "pip", "freeze"]),
            "linux_nvidia_packages": run_text(
                [
                    "bash",
                    "-lc",
                    "dpkg-query -W -f='${Package}\\n' 'nvidia-driver-*' "
                    "'cuda-drivers*' 2>/dev/null || true",
                ]
            ),
        },
        "packages": {},
        "cuda": {},
        "smoke": {},
        "checks": {},
        "success": False,
        "error": None,
    }

    try:
        import flash_attn
        import torch
        import transformers
        import triton
        from flash_attn import flash_attn_func

        # Importing the package catches missing native/runtime dependencies.
        import nano_vllm_uno  # noqa: F401

        result["packages"] = {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "flash_attn": flash_attn.__version__,
            "transformers": transformers.__version__,
            "nano_vllm_uno": package_version("nano-vllm-uno"),
        }
        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }

        torch.manual_seed(20260905)
        q = torch.randn(
            (2, 128, 8, 64),
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        result["smoke"] = {
            "fa2_forward_shape": list(output.shape),
            "fa2_output_finite": bool(torch.isfinite(output).all().item()),
            "fa2_backward_finite": bool(
                all(
                    tensor.grad is not None
                    and torch.isfinite(tensor.grad).all().item()
                    for tensor in (q, k, v)
                )
            ),
            "loss": float(loss.item()),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }

        packages = result["packages"]
        cuda = result["cuda"]
        commands = result["commands"]
        result["checks"] = {
            "python_3_10": sys.version_info[:2] == EXPECTED["python"],
            "linux_x86_64": platform.machine() == "x86_64",
            "torch_version": packages["torch"].startswith(EXPECTED["torch"]),
            "torch_cuda_version": cuda["runtime"] == EXPECTED["torch_cuda"],
            "triton_version": packages["triton"] == EXPECTED["triton"],
            "flash_attn_version": packages["flash_attn"].startswith(
                EXPECTED["flash_attn"]
            ),
            "transformers_version": packages["transformers"]
            == EXPECTED["transformers"],
            "cuda_available": cuda["available"],
            "rtx_3090": "RTX 3090" in cuda["device_name"],
            "compute_capability": tuple(cuda["compute_capability"])
            == EXPECTED["compute_capability"],
            "uno_revision": commands["uno_revision"]["output"]
            == EXPECTED["uno_commit"],
            "no_linux_nvidia_driver": not commands["linux_nvidia_packages"][
                "output"
            ].strip(),
            "fa2_forward": result["smoke"]["fa2_output_finite"],
            "fa2_backward": result["smoke"]["fa2_backward_finite"],
        }
        result["success"] = all(result["checks"].values())
    except Exception as exc:  # keep a manifest even when a binary import fails
        result["error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


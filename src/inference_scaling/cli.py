"""Small dependency-free command line interface."""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np


ALGORITHMS = {
    "mh": "suffix-resampling Metropolis--Hastings",
    "conditional-is": "on-policy conditional-energy importance sampling",
    "base-replay": "base-candidate off-policy rollout replay",
    "dynamic-is": "dynamic candidates, outer IS, and variance--cost allocation",
}


def _environment() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch
    except ImportError:
        result["torch"] = None
        result["cuda_available"] = False
    else:
        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(0)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inference-scaling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("algorithms", help="list algorithm families")
    subparsers.add_parser("environment", help="print a machine-readable environment summary")
    args = parser.parse_args(argv)

    if args.command == "algorithms":
        for name, description in ALGORITHMS.items():
            print(f"{name}: {description}")
        return 0
    if args.command == "environment":
        print(json.dumps(_environment(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

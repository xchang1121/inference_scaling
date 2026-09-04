from __future__ import annotations

from online_speculation.preflight import parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv() -> None:
    parsed = parse_nvidia_smi_csv(
        "0, NVIDIA GeForce RTX 3090, 24576, 596.49, 8.6\n"
    )

    assert parsed == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mib": 24576,
            "driver_version": "596.49",
            "compute_capability": "8.6",
        }
    ]


def test_parse_nvidia_smi_csv_ignores_diagnostics() -> None:
    assert parse_nvidia_smi_csv("NVIDIA-SMI has failed\n") == []

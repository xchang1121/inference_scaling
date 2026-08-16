from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.dllm.run_llada_suite import main


def _arguments(output: Path) -> list[str]:
    return [
        "run_llada_suite.py",
        "--config",
        "configs/gsm8k_llada_moe_3090.toml",
        "--profile",
        "smoke",
        "--tag",
        "unit",
        "--output-root",
        str(output),
        "--vrpo",
        "preflight",
        "--methods",
        "base",
    ]


def test_llada_suite_executes_preflight_quality_and_replay_then_completes(
    monkeypatch, tmp_path
):
    calls = []

    def complete(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sys, "argv", _arguments(tmp_path))
    monkeypatch.setattr(subprocess, "run", complete)
    main()

    assert [Path(call[0][1]).name for call in calls] == [
        "train_gsm8k_vrpo.py",
        "gsm8k_reproduction.py",
        "gsm8k_replay_benchmark.py",
    ]
    manifest = json.loads(
        (tmp_path / "unit" / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["completed_commands"] == 3


def test_llada_suite_records_subprocess_failure(monkeypatch, tmp_path):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(sys, "argv", _arguments(tmp_path))
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        main()

    manifest = json.loads(
        (tmp_path / "unit" / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["completed_commands"] == 0

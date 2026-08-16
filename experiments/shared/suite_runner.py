"""Shared execution and manifest handling for resumable experiment suites."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


def command_text(command: Sequence[str]) -> str:
    """Render a command using the quoting rules of the current platform."""

    return subprocess.list2cmdline(list(command))


def repository_environment(
    root: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment in which repository modules are importable."""

    environment = dict(os.environ if base is None else base)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root / "src"), str(root), existing)
    ).rstrip(os.pathsep)
    return environment


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run_manifested_commands(
    *,
    commands: Sequence[Sequence[str]],
    root: Path,
    manifest_path: Path,
    metadata: Mapping[str, Any],
    dry_run: bool,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute commands in order while atomically recording suite progress."""

    manifest: dict[str, Any] = {
        "schema_version": 1,
        **metadata,
        "commands": [command_text(command) for command in commands],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_commands": 0,
        "status": "dry_run" if dry_run else "running",
    }
    _write_manifest(manifest_path, manifest)
    for command in commands:
        print(command_text(command), flush=True)
    if dry_run:
        return manifest

    process_environment = repository_environment(root, base=environment)
    try:
        for index, command in enumerate(commands, start=1):
            subprocess.run(
                list(command),
                cwd=root,
                env=process_environment,
                check=True,
            )
            manifest["completed_commands"] = index
            _write_manifest(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    return manifest


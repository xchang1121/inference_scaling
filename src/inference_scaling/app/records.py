"""What a run writes: identities in the manifest, one record per draw, the summary.

A run directory holds ``manifest.json`` (choices, settings, code, environment,
model and dataset identities), ``records.jsonl`` (one line per problem and
draw) and ``summary.json`` (aggregates over the records of the selected
problems and draws).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from inference_scaling.datasets.base import file_sha256
from inference_scaling.shared.metrics import importance_effective_sample_size

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES = ("torch", "transformers", "accelerate", "peft", "vllm", "numpy", "math-verify", "huggingface-hub")
# The z value of a two-sided 95% interval.
_WILSON_Z = 1.959963984540054


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cached_file_sha256(path: Path, *, cache_dir: Path, expected: str | None = None) -> str:
    """Hash a large immutable file once; the cache is keyed by path, size and times."""

    artifact = path.resolve()
    stat = artifact.stat()
    identity = {"path": str(artifact), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    cache = cache_dir.resolve() / f"{hashlib.sha256(str(artifact).encode('utf-8')).hexdigest()}.json"
    digest: str | None = None
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if isinstance(cached, dict) and all(cached.get(name) == value for name, value in identity.items()):
            digest = cached.get("sha256")
    if not isinstance(digest, str):
        digest = file_sha256(artifact)
        write_json_atomic(cache, {**identity, "sha256": digest})
    if expected is not None and digest != expected:
        raise ValueError(f"hash mismatch for {artifact}: expected {expected}, got {digest}")
    return digest


def directory_hashes(path: Path, *, names: Iterable[str] | None = None, suffixes: Iterable[str] = ()) -> dict[str, str]:
    """Hash the files below a directory selected by exact name prefix or suffix."""

    directory = path.resolve()
    prefixes, endings = tuple(names or ()), tuple(suffix.lower() for suffix in suffixes)
    files = sorted(
        candidate for candidate in directory.rglob("*")
        if candidate.is_file() and (candidate.name.startswith(prefixes) or candidate.suffix.lower() in endings)
    )
    return {candidate.relative_to(directory).as_posix(): file_sha256(candidate) for candidate in files}


def checkpoint_metadata_hashes(path: Path) -> dict[str, str]:
    """Tokenizer, configuration and custom-code files of a checkpoint."""

    return directory_hashes(path, suffixes=(".json", ".jinja", ".model", ".py", ".tiktoken", ".txt"))


def adapter_hashes(path: Path) -> dict[str, str]:
    files = directory_hashes(path, names=("adapter_config.json", "adapter_model."))
    if "adapter_config.json" not in files or not any(name.startswith("adapter_model.") for name in files):
        raise FileNotFoundError(f"{path} is not a PEFT adapter directory")
    return files


def source_sha256() -> dict[str, str]:
    """Hash of every Python file of the package: a code change starts a new run."""

    return {path.relative_to(PACKAGE_ROOT).as_posix(): file_sha256(path)
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))}


def git_state() -> dict[str, Any] | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=PACKAGE_ROOT).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
                                cwd=PACKAGE_ROOT).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import torch

        if torch.cuda.is_available():
            hardware["cuda"] = torch.version.cuda
            hardware["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages,
            "hardware": hardware}


def write_json_atomic(path: Path, value: Any) -> None:
    """Publish a complete JSON file; a failed write keeps the old one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as sink:
        temporary = Path(sink.name)
        try:
            json.dump(value, sink, ensure_ascii=False, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        except BaseException:
            sink.close()
            temporary.unlink()
            raise
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            records.append(value)
    return records


def snapshot_delta(before: Any, after: Any) -> dict[str, Any]:
    """Field-wise difference of the numeric counters of two backend snapshots."""

    left, right = asdict(before), asdict(after)
    return {key: right[key] - left[key] for key, value in right.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


class Meter:
    """Backend counter deltas of one problem, by phase and model role.

    Without backends (concurrent problems share the counters) there is no
    per-problem cost.
    """

    def __init__(self, backends: Mapping[str, Any]) -> None:
        self.backends = {role: backend for role, backend in backends.items() if backend is not None}
        self.phases: dict[str, dict[str, dict[str, Any]]] = {}
        self.extra: Counter[str] = Counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        before = {role: backend.snapshot() for role, backend in self.backends.items()}
        try:
            yield
        finally:
            deltas = {role: snapshot_delta(before[role], backend.snapshot()) for role, backend in self.backends.items()}
            if deltas:
                for key, value in self.extra.items():
                    deltas["base"][key] = deltas["base"].get(key, 0) + value
                self.phases[name] = deltas
            self.extra.clear()

    def add(self, values: Mapping[str, int]) -> None:
        """Charge base-model work the counters do not see to the current phase."""

        self.extra.update(values)

    def cost(self, slots: Callable[[Mapping[str, Any]], int],
             flops: Callable[[str, Mapping[str, Any]], float]) -> dict[str, Any] | None:
        if not self.backends:
            return None
        deltas = [(role, delta) for phase in self.phases.values() for role, delta in phase.items()]
        return {"phases": self.phases, "forward_token_slots": sum(slots(delta) for _, delta in deltas),
                "flops": sum(flops(role, delta) for role, delta in deltas)}


def importance_trace(steps: Sequence[Any]) -> dict[str, Any]:
    """Weights, rewards and selections of conditional IS steps (either model family)."""

    rollouts = [rollout for step in steps for candidate in step.candidates for rollout in candidate.rollouts]
    rewards = [float(rollout.reward) for rollout in rollouts]
    optional = ("completion_index", "retained_candidate", "rollout_evaluations_performed")
    return {
        "steps": [{
            "prefix_length": step.generated_length_before,
            "candidate_log_weights": [candidate.log_weight for candidate in step.candidates],
            "selected_index": step.selected_index,
            **{name: getattr(step, name) for name in optional if hasattr(step, name)},
        } for step in steps],
        "rollout_evaluations": len(rollouts),
        "mean_rollout_ess": statistics.fmean(
            importance_effective_sample_size([rollout.log_weight for rollout in candidate.rollouts])
            for step in steps for candidate in step.candidates
        ) if steps else 0.0,
        "rollout_reward": _statistics(rewards),
    }


def wilson_interval(successes: int, trials: int) -> list[float]:
    proportion, z = successes / trials, _WILSON_Z
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def pass_at_k(correct: int, draws: int, k: int) -> float:
    """Unbiased pass@k from ``draws`` samples of which ``correct`` are correct."""

    return 1.0 if draws - correct < k else 1.0 - math.comb(draws - correct, k) / math.comb(draws, k)


def _statistics(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "sum": math.fsum(values)}


def _add(total: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, Mapping):
            _add(total.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


def summarize(records: Sequence[Mapping[str, Any]], *, draws: int) -> dict[str, Any]:
    """Aggregate the records of one run; pass@k needs every draw of a problem."""

    count = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    summary: dict[str, Any] = {
        "records": count,
        "problems": len({record["problem_id"] for record in records}),
        "draws": draws,
        "correct": correct,
        "accuracy": correct / count if count else None,
        "accuracy_wilson_95": wilson_interval(correct, count) if count else None,
    }
    if draws > 1:
        outcomes: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            outcomes[record["problem_id"]].append(bool(record["correct"]))
        complete = [values for values in outcomes.values() if len(values) == draws]
        ks = sorted({*(2 ** power for power in range(draws.bit_length()) if 2 ** power <= draws), draws})
        summary["pass_at_k"] = {
            str(k): statistics.fmean(pass_at_k(sum(values), draws, k) for values in complete) if complete else None
            for k in ks
        }
        summary["pass_at_k_problems"] = len(complete)
    cost: dict[str, Any] = {}
    for record in records:
        if record["cost"] is not None:
            _add(cost, record["cost"])
    summary["cost_total"] = cost
    summary["reward"] = _statistics([float(record["reward"]) for record in records if record["reward"] is not None])
    summary["seconds"] = _statistics([float(record["elapsed_seconds"]) for record in records])
    summary["output_tokens"] = _statistics([float(record["output"]["tokens"]) for record in records])
    summary["failures"] = {
        "unparseable": sum(not record["parseable"] for record in records),
        "length_exhausted": sum(bool(record["output"]["length_exhausted"]) for record in records),
        "fallbacks": dict(Counter(reason for record in records for reason in record["fallbacks"])),
    }
    return summary


__all__ = [
    "adapter_hashes",
    "cached_file_sha256",
    "checkpoint_metadata_hashes",
    "environment",
    "git_state",
    "importance_trace",
    "json_sha256",
    "Meter",
    "load_jsonl",
    "pass_at_k",
    "snapshot_delta",
    "source_sha256",
    "summarize",
    "wilson_interval",
    "write_json_atomic",
]

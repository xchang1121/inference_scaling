"""One run: choices + settings -> a resumable results directory.

The directory is ``<output>/<dataset>/<model>/<algorithm>[-<reward>]/<fingerprint>``.
The fingerprint hashes everything that determines the records: the choices,
the settings they read, the model and dataset identities and the package
source. Rerunning with the same fingerprint appends only missing
(problem, draw) records; the draw count is not part of it, so draws can be
extended later.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from inference_scaling.app.records import (
    environment,
    git_state,
    json_sha256,
    load_jsonl,
    source_sha256,
    summarize,
    write_json_atomic,
)
from inference_scaling.datasets import load_dataset
from inference_scaling.datasets.base import Dataset, Problem
from inference_scaling.shared.rng import SeedStream


@dataclass(frozen=True)
class Choices:
    algorithm: str
    model: str
    reward: str | None
    dataset: str

    @property
    def label(self) -> str:
        return self.algorithm if self.reward is None else f"{self.algorithm}-{self.reward}"


def _family(settings: Mapping[str, Any], choices: Choices, dataset: Dataset) -> Any:
    # Imported lazily: each family pulls in its own model stack.
    if choices.model == "ar":
        from inference_scaling.app.ar import ARFamily

        return ARFamily(settings, choices, dataset)
    from inference_scaling.app.dllm import DLLMFamily

    return DLLMFamily(settings, choices, dataset)


def effective_settings(settings: Mapping[str, Any], choices: Choices) -> dict[str, Any]:
    """The settings a run reads; the draw count and the hash cache location are excluded."""

    family = settings[choices.model]
    return {
        "seed": settings["run"]["seed"],
        "dataset": settings["datasets"][choices.dataset],
        "reward": None if choices.reward is None else settings["rewards"][choices.reward],
        "model": {key: value for key, value in family.items() if key != "algorithms"},
        "algorithm": family["algorithms"][choices.algorithm],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _solve(family: Any, dataset: Dataset, problem: Problem, draw: int, seed: int) -> dict[str, Any]:
    family.synchronize()
    started = time.perf_counter()
    solution = family.solve(problem, SeedStream(SeedStream(seed).derive("draw", draw)))
    family.synchronize()
    elapsed = time.perf_counter() - started
    grade = dataset.grade(solution.pop("answer_text"), problem)
    return {
        "problem_id": problem.id,
        "draw": draw,
        "question": problem.question,
        "reference": problem.answer,
        **solution,
        "answer": grade.answer,
        "parseable": grade.parseable,
        "correct": grade.correct,
        "elapsed_seconds": elapsed,
    }


def run(choices: Choices, settings: Mapping[str, Any], output: Path) -> dict[str, Any]:
    draws, seed = int(settings["run"]["draws"]), int(settings["run"]["seed"])
    if draws < 1:
        raise ValueError("run.draws must be positive")
    dataset = load_dataset(choices.dataset, settings["datasets"][choices.dataset])
    family = None
    try:
        family = _family(settings, choices, dataset)
        identity = {
            "choices": asdict(choices),
            "effective_settings": effective_settings(settings, choices),
            "models": family.artifacts(),
            "dataset": dataset.describe(),
            "source_sha256": json_sha256(source_sha256()),
        }
        fingerprint = json_sha256(identity)
        directory = output / choices.dataset / choices.model / choices.label / fingerprint[:16]
        manifest_path, records_path = directory / "manifest.json", directory / "records.jsonl"
        if not manifest_path.is_file():
            write_json_atomic(manifest_path, {
                "fingerprint": fingerprint, **identity, "settings": settings,
                "settings_sha256": json_sha256(settings), "git": git_state(), "environment": environment(),
                "created": _now(),
            })
        elif json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError(f"{directory} belongs to another run")
        done = {(record["problem_id"], record["draw"]) for record in load_jsonl(records_path)}
        pending = [(problem, draw) for draw in range(draws) for problem in dataset.problems
                   if (problem.id, draw) not in done]
        if pending:
            family.load()
            lock = threading.Lock()
            with records_path.open("a", encoding="utf-8") as sink:

                def work(item: tuple[Problem, int]) -> None:
                    problem, draw = item
                    record = _solve(family, dataset, problem, draw, seed)
                    with lock:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                        done.add((problem.id, draw))
                        print(f"[{len(done)}/{len(dataset.problems) * draws}] {choices.label} problem={problem.id} "
                              f"draw={draw} correct={record['correct']} seconds={record['elapsed_seconds']:.1f}",
                              flush=True)

                with ThreadPoolExecutor(max_workers=family.workers) as pool:
                    for future in [pool.submit(work, item) for item in pending]:
                        future.result()
        selected = {problem.id for problem in dataset.problems}
        records = [record for record in load_jsonl(records_path)
                   if record["problem_id"] in selected and record["draw"] < draws]
        summary = {"fingerprint": fingerprint, "directory": str(directory), "updated": _now(),
                   **summarize(records, draws=draws)}
        write_json_atomic(directory / "summary.json", summary)
        return summary
    finally:
        if family is not None:
            family.close()
        dataset.close()


__all__ = ["Choices", "effective_settings", "run"]

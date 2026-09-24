"""MATH-500: the pinned Hugging Face release, graded by Math-Verify in a worker process."""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from inference_scaling.datasets.base import Dataset, Grade, Problem, file_sha256


def _judge_worker(connection: Any) -> None:
    # One persistent process gives a real timeout on Windows and Unix; the
    # library's own signal-based timeouts are disabled inside it.
    from math_verify import LatexExtractionConfig, parse, verify

    @lru_cache(maxsize=4096)
    def parsed(text: str, reference: bool) -> Any:
        if reference:
            text = "\\boxed{" + text + "}"
        return parse(text, extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback",
                     parsing_timeout=0)

    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            operation, values = request
            try:
                result: Any
                if operation == "grade":
                    prediction, reference = values
                    gold, answer = parsed(reference, True), parsed(prediction, False)
                    result = {"correct": bool(gold and answer and verify(gold, answer, timeout_seconds=0)),
                              "parseable": bool(answer)}
                elif operation == "equivalent":
                    left, right = (parsed(value, False) for value in values)
                    result = bool(left and right and verify(left, right, timeout_seconds=0)
                                  and verify(right, left, timeout_seconds=0))
                elif operation == "parse":
                    value = parsed(values[0], False)
                    result = None if not value else str(value[0])
                else:
                    raise ValueError(f"unknown judge operation: {operation}")
                connection.send((True, result))
            except Exception as error:
                connection.send((False, f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class MathJudge:
    """Math-Verify equivalence and grading behind a timeout."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._process: Any = None
        self._connection: Any = None

    def _request(self, operation: str, *values: str) -> Any:
        if self._process is None:
            context = mp.get_context("spawn")
            self._connection, child = context.Pipe()
            self._process = context.Process(target=_judge_worker, args=(child,), daemon=True)
            self._process.start()
            child.close()
        self._connection.send((operation, values))
        if not self._connection.poll(self.timeout_seconds):
            self.close()
            raise TimeoutError("math answer evaluation timed out")
        success, result = self._connection.recv()
        if not success:
            raise ValueError(result)
        return result

    def grade(self, prediction: str, reference: str) -> dict[str, bool]:
        return self._request("grade", prediction, reference)

    def equivalent(self, left: str, right: str) -> bool:
        return bool(self._request("equivalent", left, right))

    def answer_key(self, text: str) -> str | None:
        return self._request("parse", text)

    def close(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=self.timeout_seconds)
            self._process.close()
            self._connection.close()
            self._process = self._connection = None


def stratified_order(problems: list[Problem], *, seed: int, minimum_level: int,
                     excluded_ids: frozenset[str]) -> list[Problem]:
    """Round-robin over shuffled subject/level strata, independent of the answers."""

    groups: dict[tuple[str, int], list[Problem]] = defaultdict(list)
    for problem in sorted(problems, key=lambda item: item.id):
        if int(problem.metadata["level"]) >= minimum_level and problem.id not in excluded_ids:
            groups[(str(problem.metadata["subject"]), int(problem.metadata["level"]))].append(problem)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    ordered = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


class MATH500(Dataset):
    name = "math500"

    def __init__(self, settings: Mapping[str, Any]) -> None:
        source = settings["source"]
        path = Path(str(settings["path"]))
        if settings["download"] and not path.is_file():
            from huggingface_hub import hf_hub_download

            downloaded = Path(hf_hub_download(str(source["repository"]), str(source["filename"]), repo_type="dataset",
                                              revision=str(source["revision"]), local_dir=path.parent))
            if downloaded != path:
                downloaded.replace(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        problems = [Problem(str(row["unique_id"]), str(row["problem"]), str(row["answer"]),
                            {"subject": str(row["subject"]), "level": int(row["level"])}) for row in rows]
        if len({problem.id for problem in problems}) != len(problems):
            raise ValueError("MATH-500 contains duplicate problem identifiers")
        selection = settings["selection"]
        ordered = stratified_order(problems, seed=int(selection["seed"]), minimum_level=int(selection["minimum_level"]),
                                   excluded_ids=frozenset(map(str, selection["excluded_ids"])))
        # The first ``skip`` problems are held out (e.g. for development); the next ``count`` are evaluated.
        skip, count = int(selection["skip"]), int(selection["count"])
        if skip < 0 or count <= 0 or skip + count > len(ordered):
            raise ValueError(f"selection.skip + selection.count must lie within the {len(ordered)} eligible problems")
        super().__init__(settings, tuple(ordered[skip:skip + count]), file_sha256(path))
        self.judge = MathJudge(float(settings["judge_timeout_seconds"]))
        for problem in self.problems:
            if not self.judge.grade("\\boxed{" + problem.answer + "}", problem.answer)["correct"]:
                self.close()
                raise ValueError(f"the grader cannot verify the reference answer of {problem.id}")

    # Math-Verify parses answers inside ``same``; a text stands for its own answer.
    def answer(self, text: str) -> str | None:
        return text or None

    def same(self, left: Any, right: Any) -> bool:
        return self.judge.equivalent(str(left), str(right))

    def grade(self, text: str, problem: Problem) -> Grade:
        result = self.judge.grade(text, problem.answer)
        return Grade(self.judge.answer_key(text), bool(result["parseable"]), bool(result["correct"]))

    def close(self) -> None:
        self.judge.close()


__all__ = ["MATH500", "MathJudge", "stratified_order"]

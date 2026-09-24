"""GSM8K: the official split pinned by URL and checksum, graded on the final number."""

from __future__ import annotations

import json
import random
import re
import urllib.request
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from inference_scaling.datasets.base import Dataset, Grade, Problem, file_sha256

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_FRACTION = rf"(?:{_NUMBER})\s*/\s*(?:{_NUMBER})"
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(" + _FRACTION + "|" + _NUMBER + r")\s*\}")
_HASH_RE = re.compile(r"####\s*(" + _FRACTION + "|" + _NUMBER + r")")
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|answer)\s*(?:is|:|=)?\s*\$?\s*(" + _FRACTION + "|" + _NUMBER + r")",
    flags=re.IGNORECASE,
)
_ANY_NUMBER_RE = re.compile(_FRACTION + "|" + _NUMBER)


def _as_fraction(value: str) -> Fraction | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", 1)
            return Fraction(Decimal(numerator.strip())) / Fraction(Decimal(denominator.strip()))
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def extract_numeric_answer(text: str) -> Fraction | None:
    """The final number of a text: ``####``, then ``\\boxed{}``, then "answer", then the last number."""

    for pattern in (_HASH_RE, _BOXED_RE, _ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            parsed = _as_fraction(matches[-1])
            if parsed is not None:
                return parsed
    matches = _ANY_NUMBER_RE.findall(text)
    return _as_fraction(matches[-1]) if matches else None


def fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _download(path: Path, url: str, sha256: str) -> None:
    """Fetch the pinned file atomically unless an identical copy is present."""

    if path.is_file() and file_sha256(path) == sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = file_sha256(temporary)
        if actual != sha256:
            raise ValueError(f"GSM8K checksum mismatch: expected {sha256}, downloaded {actual}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class GSM8K(Dataset):
    name = "gsm8k"

    def __init__(self, settings: Mapping[str, Any]) -> None:
        source = settings["source"]
        path = Path(str(settings["path"]))
        if settings["download"]:
            _download(path, str(source["url"]), str(source["sha256"]))
        digest = file_sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"{path} is not the pinned GSM8K file")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != int(source["rows"]):
            raise ValueError(f"expected {source['rows']} GSM8K rows, found {len(rows)}")
        problems = []
        for index, row in enumerate(rows):
            answer = extract_numeric_answer(str(row["answer"]))
            if answer is None:
                raise ValueError(f"could not parse the GSM8K reference answer of row {index}")
            problems.append(Problem(str(index), str(row["question"]), str(fraction_text(answer)),
                                    {"solution": str(row["answer"])}))
        count = settings["selection"]["count"]
        if count is not None:
            # Indices are drawn before any inference and kept in dataset order.
            if not 0 < int(count) <= len(problems):
                raise ValueError(f"selection.count must lie in [1, {len(problems)}]")
            chosen = sorted(random.Random(int(settings["selection"]["seed"])).sample(range(len(problems)), int(count)))
            problems = [problems[index] for index in chosen]
        super().__init__(settings, tuple(problems), digest)

    def answer(self, text: str) -> Fraction | None:
        return extract_numeric_answer(text)

    def same(self, left: Any, right: Any) -> bool:
        return bool(left == right)

    def grade(self, text: str, problem: Problem) -> Grade:
        predicted = extract_numeric_answer(text)
        return Grade(fraction_text(predicted), predicted is not None,
                     predicted is not None and predicted == Fraction(problem.answer))


__all__ = ["GSM8K", "extract_numeric_answer", "fraction_text"]

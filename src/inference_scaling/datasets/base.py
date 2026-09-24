"""The interface every evaluation dataset implements."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Problem:
    id: str
    question: str
    answer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Grade:
    answer: str | None
    parseable: bool
    correct: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Dataset(ABC):
    """Problems, their prompt text, an answer rule and a grader.

    ``answer``/``same`` form the answer rule the vote reward compares answers
    with; ``grade`` checks a text against the reference answer and is used for
    evaluation and by the dataset verifier.
    """

    name: str

    def __init__(self, settings: Mapping[str, Any], problems: tuple[Problem, ...], source_sha256: str) -> None:
        if str(settings["prompt_template"]).count("{question}") != 1:
            raise ValueError("prompt_template must contain {question} exactly once")
        self.settings = settings
        self.problems = problems
        self.source_sha256 = source_sha256

    def prompt(self, problem: Problem) -> str:
        # Plain substitution keeps LaTeX braces such as \boxed{} literal.
        return str(self.settings["prompt_template"]).replace("{question}", problem.question)

    @abstractmethod
    def answer(self, text: str) -> Any | None:
        """The final answer of a text, or ``None`` when it has none."""

    @abstractmethod
    def same(self, left: Any, right: Any) -> bool:
        """Whether two answers returned by :meth:`answer` agree."""

    @abstractmethod
    def grade(self, text: str, problem: Problem) -> Grade:
        """Compare the final answer of ``text`` with the problem's reference."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.settings["path"]),
            "source_sha256": self.source_sha256,
            "problem_ids": [problem.id for problem in self.problems],
        }

    def close(self) -> None:
        """Release resources such as a judge process."""


__all__ = ["Dataset", "Grade", "Problem", "file_sha256"]

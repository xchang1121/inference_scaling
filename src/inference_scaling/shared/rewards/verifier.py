"""External reward sources: information the model alone cannot obtain.

A verifier scores decoded ``(prompt, completion)`` text. The ``source`` setting
selects one of

- ``dataset``: the dataset's grader against the problem's reference answer,
  mapped to the configured correct / incorrect / unparseable values;
- ``python``: a trusted factory ``package.module:function`` called as
  ``factory(context=context, **options)``. It returns a callable
  ``(prompt, completion) -> reward`` or an object with ``score`` and optionally
  ``score_batch(prompt, completions)``, for example a reward model r = f(x, y);
- ``constant``: one fixed value, for integration tests and reward-free controls.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any

from inference_scaling.datasets.base import Grade

VERIFIER_SOURCES = ("dataset", "python", "constant")


@dataclass(frozen=True, slots=True)
class VerifierContext:
    """What one verifier instance sees: the prompt and, when it asks, the reference answer."""

    prompt: str
    reference: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


def _finite(value: object, source: str) -> float:
    try:
        reward = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"the {source} verifier returned a non-numeric reward {value!r}") from error
    if not isfinite(reward):
        raise ValueError(f"the {source} verifier returned a non-finite reward {reward!r}")
    return reward


class Verifier:
    """One verifier bound to one prompt; every reward is checked to be finite."""

    def __init__(
        self,
        description: Mapping[str, object],
        score: Callable[[str, str], object],
        score_batch: Callable[[str, Sequence[str]], Sequence[object]] | None = None,
    ) -> None:
        self.description = dict(description)
        self.source = str(self.description["source"])
        self._score = score
        self._score_batch = score_batch

    def score(self, prompt: str, completion: str) -> float:
        return _finite(self._score(prompt, completion), self.source)

    def score_batch(self, prompt: str, completions: Sequence[str]) -> tuple[float, ...]:
        if self._score_batch is None:
            return tuple(self.score(prompt, completion) for completion in completions)
        values = tuple(self._score_batch(prompt, completions))
        if len(values) != len(completions):
            raise ValueError(f"the {self.source} verifier returned {len(values)} rewards for {len(completions)} texts")
        return tuple(_finite(value, self.source) for value in values)

    def describe(self) -> dict[str, object]:
        return dict(self.description)


def _load_factory(reference: str) -> Callable[..., object]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("verifier.python.factory must use the form 'package.module:function'")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError(f"verifier factory {reference!r} is not callable")
    return factory


def build_verifier(
    settings: Mapping[str, Any],
    *,
    context: VerifierContext,
    grade: Callable[[str], Grade] | None,
) -> Verifier:
    """Construct the verifier selected by ``settings["source"]`` for one prompt."""

    source = settings["source"]
    if source == "dataset":
        if grade is None:
            raise ValueError("the dataset verifier needs the dataset's grader")
        values = {name: _finite(settings["dataset"][name], source) for name in ("correct", "incorrect", "unparseable")}

        def score(_prompt: str, completion: str) -> float:
            result = grade(completion)
            return values["correct" if result.correct else "incorrect" if result.parseable else "unparseable"]

        return Verifier({"source": source, **values}, score)
    if source == "constant":
        value = _finite(settings["constant"]["value"], source)
        return Verifier({"source": source, "value": value}, lambda _prompt, _completion: value)
    if source != "python":
        raise ValueError(f"unknown verifier source {source!r}; choose one of {VERIFIER_SOURCES}")
    python = settings["python"]
    options = dict(python["options"])
    if "context" in options:
        raise ValueError("verifier.python.options.context is reserved for the runtime")
    if python["requires_reference"] and context.reference is None:
        raise ValueError("this verifier requires a reference answer")
    visible = context if python["requires_reference"] else replace(context, reference=None)
    constructed = _load_factory(str(python["factory"]))(context=visible, **options)
    description = {"source": source, "factory": str(python["factory"]), "options": options,
                   "requires_reference": bool(python["requires_reference"])}
    point = getattr(constructed, "score", None)
    if callable(point):
        batch = getattr(constructed, "score_batch", None)
        return Verifier(description, point, batch if callable(batch) else None)
    if callable(constructed):
        return Verifier(description, constructed)
    raise TypeError(f"verifier factory {python['factory']!r} returned unsupported {type(constructed).__name__}")


__all__ = ["VERIFIER_SOURCES", "Verifier", "VerifierContext", "build_verifier"]

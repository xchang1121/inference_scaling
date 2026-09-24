"""Evaluation datasets behind one interface.

A dataset supplies its problems, the prompt text of a problem, an answer rule
(the vote reward compares answers with it) and a grader against the reference
answer (used for evaluation and by the dataset verifier). Sources, selection
and prompts come from the dataset's section of ``settings/inference.json``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from inference_scaling.datasets.base import Dataset, Grade, Problem
from inference_scaling.datasets.gsm8k import GSM8K, extract_numeric_answer, fraction_text
from inference_scaling.datasets.math500 import MATH500

DATASETS: dict[str, Callable[[Mapping[str, Any]], Dataset]] = {"gsm8k": GSM8K, "math500": MATH500}


def load_dataset(name: str, settings: Mapping[str, Any]) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose one of {sorted(DATASETS)}")
    return DATASETS[name](settings)


__all__ = ["DATASETS", "Dataset", "Grade", "Problem", "extract_numeric_answer", "fraction_text", "load_dataset"]

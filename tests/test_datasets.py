from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from inference_scaling.app.records import write_json_atomic
from inference_scaling.app.settings import load_settings
from inference_scaling.datasets import load_dataset
from inference_scaling.datasets.base import Problem
from inference_scaling.datasets.gsm8k import GSM8K, extract_numeric_answer
from inference_scaling.datasets.math500 import MATH500, MathJudge, stratified_order


def _gsm8k_settings(tmp_path: Path, rows: int, *, count: int | None, seed: int = 17) -> dict:
    path = tmp_path / "gsm8k.jsonl"
    path.write_text("".join(json.dumps({"question": f"q{index}", "answer": f"work\n#### {index}"}) + "\n"
                            for index in range(rows)), encoding="utf-8")
    settings = copy.deepcopy(load_settings()["datasets"]["gsm8k"])
    settings.update(path=str(path), download=False)
    settings["source"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows)
    settings["selection"].update(count=count, seed=seed)
    return settings


def test_extract_numeric_answer_prefers_explicit_final_markers() -> None:
    assert extract_numeric_answer("2 + 3 = 5\n#### 5") == Fraction(5)
    assert extract_numeric_answer(r"work 100 then \boxed{3/4}") == Fraction(3, 4)
    assert extract_numeric_answer("The final answer is $1,250.50") == Fraction(2501, 2)
    assert extract_numeric_answer("intermediate 7, last 9") == Fraction(9)
    assert extract_numeric_answer("no digits") is None


def test_gsm8k_selection_is_seeded_ordered_and_grades_the_final_number(tmp_path) -> None:
    dataset = GSM8K(_gsm8k_settings(tmp_path, 20, count=6))
    again = GSM8K(_gsm8k_settings(tmp_path, 20, count=6))
    ids = [problem.id for problem in dataset.problems]
    assert ids == [problem.id for problem in again.problems]
    assert [int(value) for value in ids] == sorted(int(value) for value in ids)
    problem = dataset.problems[0]
    assert problem.metadata["solution"].endswith(f"#### {problem.id}")
    assert dataset.grade(f"so #### {problem.id}", problem).correct
    wrong = dataset.grade("#### 999", problem)
    assert wrong.parseable and not wrong.correct and wrong.answer == "999"
    assert not dataset.grade("no answer", problem).parseable
    assert dataset.same(dataset.answer("#### 1.50"), dataset.answer("#### 3/2"))
    assert len(GSM8K(_gsm8k_settings(tmp_path, 20, count=None)).problems) == 20


def test_gsm8k_rejects_an_unpinned_file_and_a_bad_count(tmp_path) -> None:
    settings = _gsm8k_settings(tmp_path, 5, count=None)
    settings["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned"):
        GSM8K(settings)
    with pytest.raises(ValueError, match="selection.count"):
        GSM8K(_gsm8k_settings(tmp_path, 5, count=6))


def test_prompt_template_keeps_latex_braces_and_needs_the_question(tmp_path) -> None:
    settings = _gsm8k_settings(tmp_path, 3, count=None)
    settings["prompt_template"] = "{question}\nAnswer in \\boxed{}."
    dataset = load_dataset("gsm8k", settings)
    assert dataset.prompt(dataset.problems[0]) == "q0\nAnswer in \\boxed{}."
    settings["prompt_template"] = "no placeholder"
    with pytest.raises(ValueError, match="exactly once"):
        GSM8K(settings)


def _math_problems() -> list[Problem]:
    return [Problem(str(index), "question", str(index), {"subject": str(index % 3), "level": 3 + index % 2})
            for index in range(50)]


def test_stratified_order_is_reproducible_and_independent_of_answers() -> None:
    problems = _math_problems()
    ordered = stratified_order(problems, seed=17, minimum_level=3, excluded_ids=frozenset())
    assert ordered == stratified_order(list(reversed(problems)), seed=17, minimum_level=3, excluded_ids=frozenset())
    changed = [Problem(problem.id, problem.question, "different", problem.metadata) for problem in problems]
    assert [problem.id for problem in ordered] == [
        problem.id for problem in stratified_order(changed, seed=17, minimum_level=3, excluded_ids=frozenset())]
    assert len({(problem.metadata["subject"], problem.metadata["level"]) for problem in ordered[:6]}) == 6
    filtered = stratified_order(problems, seed=17, minimum_level=4, excluded_ids=frozenset({ordered[0].id}))
    assert ordered[0].id not in {problem.id for problem in filtered}
    assert all(problem.metadata["level"] >= 4 for problem in filtered)


def _math_settings(tmp_path: Path, rows: list[dict]) -> dict:
    path = tmp_path / "math.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    settings = copy.deepcopy(load_settings()["datasets"]["math500"])
    settings.update(path=str(path), download=False)
    settings["selection"].update(minimum_level=1, excluded_ids=[], skip=1, count=2)
    return settings


def test_math500_rejects_duplicate_ids(tmp_path) -> None:
    row = {"unique_id": "x", "problem": "x", "answer": "1", "subject": "a", "level": 3}
    with pytest.raises(ValueError, match="duplicate"):
        MATH500(_math_settings(tmp_path, [row, row]))


def test_math500_skips_held_out_problems_and_grades_with_math_verify(tmp_path) -> None:
    pytest.importorskip("math_verify")
    rows = [{"unique_id": f"p{index}", "problem": f"q{index}", "answer": "\\frac{1}{2}", "subject": "a", "level": 3}
            for index in range(4)]
    dataset = MATH500(_math_settings(tmp_path, rows))
    try:
        assert len(dataset.problems) == 2
        problem = dataset.problems[0]
        assert dataset.grade(r"Therefore $\boxed{0.5}$.", problem).correct
        assert not dataset.grade("unfinished thought", problem).parseable
        # Equal parsed answers agree at once; other pairs are judged once and cached.
        assert dataset.same(dataset.answer(r"\boxed{2/4}"), dataset.answer(r"so \boxed{0.5}")) and not dataset._verdicts
        assert not dataset.same(dataset.answer(r"\boxed{2}"), dataset.answer(r"\boxed{3}")) and len(dataset._verdicts) == 1
        assert dataset.answer("unfinished thought") is None
    finally:
        dataset.close()


def test_math_judge_cleans_up_its_worker() -> None:
    pytest.importorskip("math_verify")
    judge = MathJudge(timeout_seconds=20)
    try:
        assert not judge.grade(r"$\boxed{3}$", "4")["correct"]
        assert judge.answer_key(r"\boxed{2}") == "2"
    finally:
        judge.close()
    assert judge._process is None


def test_atomic_json_keeps_the_previous_file_after_a_failed_write(tmp_path) -> None:
    path = tmp_path / "summary.json"
    write_json_atomic(path, {"completed": 2})
    previous = path.read_bytes()
    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": object()})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))

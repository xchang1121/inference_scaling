from copy import deepcopy

import pytest

from experiments.shared.reasoning_results import comparison_coverage, summarize_reasoning


def row(problem="one", method="base_enabled", draw=0, correct=True):
    return {"problem_id": problem, "draw": draw, "method": method, "reward": "none",
            "budget_forward_tokens": 128, "used_forward_tokens": 10, "parameter_count": 7,
            "cost": {"generation_forward_token_slots": 8, "score_forward_token_slots": 2,
                     "estimated_dense_forward_flops": 140, "generated_tokens": 6},
            "correct": correct, "parseable": True, "selected_tokens": 6, "thinking_status": "complete",
            "manifest_fingerprint": "same"}


def test_coverage_counts_missing_conditions_and_duplicate_records():
    records = [row(), row(method="base_disabled")]
    options = dict(problem_ids=["one", "two"], budgets=[128], draws=1, methods=["base"], rewards=[])
    result = comparison_coverage(records, **options)
    assert result["expected_records"] == 4 and result["missing_records"] == 2
    assert result["completed_problems"] == 1 and not result["complete"]
    records += [row("two"), row("two", method="base_disabled")]
    assert comparison_coverage(records, **options)["complete"]
    with pytest.raises(ValueError, match="duplicate"):
        comparison_coverage(records + [records[0]], **options)


def test_summary_validates_cost_and_keeps_paired_problem_differences():
    records = [row(), row("two", correct=False), row(method="vote"), row("two", method="vote")]
    summary = summarize_reasoning(records)
    base, vote = summary
    assert base["accuracy"] == 0.5
    assert base["mean_used_forward_tokens"] == 10
    assert base["mean_pfLOPs"] == 140 / 1e15
    assert vote["paired_vs_thinking_base"]["difference"] == 0.5
    broken = deepcopy(records)
    broken[0]["used_forward_tokens"] = 11
    with pytest.raises(ValueError, match="token accounting"):
        summarize_reasoning(broken)
    broken = deepcopy(records)
    broken[0]["cost"]["estimated_dense_forward_flops"] = 141
    with pytest.raises(ValueError, match="FLOPs"):
        summarize_reasoning(broken)
    broken = deepcopy(records)
    broken[0]["manifest_fingerprint"] = "another"
    with pytest.raises(ValueError, match="mix"):
        summarize_reasoning(broken)


def test_repeated_draws_use_problem_bootstrap_and_partial_pairs_are_omitted():
    records = [row(draw=0), row(draw=1, correct=False), row(method="vote")]
    base, vote = summarize_reasoning(records)
    assert base["problems"] == 1 and base["trials"] == 2
    assert base["interval_method"] == "problem_bootstrap"
    assert base["accuracy_95"] == [0.5, 0.5]
    assert "paired_vs_thinking_base" not in vote

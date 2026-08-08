from __future__ import annotations

import pytest

from experiments.gsm8k_is_passk import (
    _combine_batching_snapshots,
    _combine_numeric_deltas,
    _paired_pass_at_k_comparison,
    _summarize_batching_by_model,
    _summarize_model_compute,
)


def test_is_passk_combines_base_and_proposal_compute() -> None:
    base = {
        "generation_forward_token_slots": 10,
        "score_forward_token_slots": 20,
        "estimated_dense_forward_flops": 300,
    }
    proposal = {
        "generation_forward_token_slots": 7,
        "score_forward_token_slots": 0,
        "estimated_dense_forward_flops": 40,
    }
    assert _combine_numeric_deltas(base, proposal) == {
        "generation_forward_token_slots": 17,
        "score_forward_token_slots": 20,
        "estimated_dense_forward_flops": 340,
    }
    assert _combine_numeric_deltas(base, None) == base


def test_is_passk_combines_batching_by_sum_and_max() -> None:
    base = {
        "sample_batches": 2,
        "score_batches": 3,
        "sample_requests": 8,
        "score_sequences": 12,
        "maximum_sample_batch": 6,
        "maximum_score_batch": 7,
    }
    proposal = {
        "sample_batches": 4,
        "score_batches": 0,
        "sample_requests": 10,
        "score_sequences": 0,
        "maximum_sample_batch": 8,
        "maximum_score_batch": 0,
    }
    assert _combine_batching_snapshots(base, proposal) == {
        "sample_batches": 6,
        "score_batches": 3,
        "sample_requests": 18,
        "score_sequences": 12,
        "maximum_sample_batch": 8,
        "maximum_score_batch": 7,
    }


def test_is_passk_keeps_model_specific_compute_and_batching() -> None:
    base_delta = {
        "generation_forward_token_slots": 10,
        "score_forward_token_slots": 5,
        "estimated_dense_forward_flops": 30,
    }
    proposal_delta = {
        "generation_forward_token_slots": 7,
        "score_forward_token_slots": 0,
        "estimated_dense_forward_flops": 8,
    }
    base_batching = {
        "sample_batches": 2,
        "score_batches": 1,
        "sample_requests": 5,
        "score_sequences": 3,
        "maximum_sample_batch": 4,
        "maximum_score_batch": 3,
    }
    proposal_batching = {
        "sample_batches": 3,
        "score_batches": 0,
        "sample_requests": 7,
        "score_sequences": 0,
        "maximum_sample_batch": 6,
        "maximum_score_batch": 0,
    }
    chunks = [
        {
            "base_backend_delta": base_delta,
            "proposal_backend_delta": proposal_delta,
            "continuous_batching_by_model": {
                "base": base_batching,
                "proposal": proposal_batching,
            },
        },
        {
            "base_backend_delta": base_delta,
            "proposal_backend_delta": proposal_delta,
            "continuous_batching_by_model": {
                "base": base_batching,
                "proposal": proposal_batching,
            },
        },
    ]
    assert _summarize_model_compute(chunks, "base_backend_delta") == {
        "estimated_dense_forward_flops": 60,
        "generation_forward_token_slots": 20,
        "score_forward_token_slots": 10,
        "total_forward_token_slots": 30,
        "estimated_dense_forward_petaflops": 60 / 1e15,
    }
    assert _summarize_batching_by_model(chunks, "proposal") == {
        "sample_batches": 6,
        "score_batches": 0,
        "sample_requests": 14,
        "score_sequences": 0,
        "maximum_sample_batch": 6,
        "maximum_score_batch": 0,
    }


def test_is_passk_paired_bootstrap_uses_problem_level_differences() -> None:
    standard = {
        "estimated_pass_at_k": {"1": 0.5, "2": 0.75},
        "per_problem": [
            {"problem_index": 1, "correct_draws": 0},
            {"problem_index": 2, "correct_draws": 2},
        ],
    }
    small = {
        "estimated_pass_at_k": {"1": 0.75, "2": 1.0},
        "per_problem": [
            {"problem_index": 1, "correct_draws": 1},
            {"problem_index": 2, "correct_draws": 2},
        ],
    }
    comparison = _paired_pass_at_k_comparison(
        standard, small, draws=2, seed=3, replicates=1_000
    )
    assert comparison["1"]["small_proposal_minus_standard"] == pytest.approx(0.25)
    assert comparison["2"]["small_proposal_minus_standard"] == pytest.approx(0.5)
    assert comparison["1"]["paired_problem_bootstrap_95"] == [0.0, 0.5]
    assert comparison["2"]["paired_problem_bootstrap_95"] == [0.0, 1.0]

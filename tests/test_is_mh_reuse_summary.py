from copy import deepcopy
from pathlib import Path

import pytest

from experiments.arllm.summarize_is_mh_reuse import _load, build_summary


ROOT = Path(__file__).resolve().parents[1]
REPORTS = tuple(
    ROOT / "results" / "infra" / f"rtx3090_transformers_is_mh_seed{seed}.json"
    for seed in (20260812, 20260813, 20260814)
)


def test_committed_is_mh_reuse_runs_build_paired_factors() -> None:
    summary = build_summary(_load(REPORTS))
    comparisons = summary["comparisons"]

    assert summary["runs"] == 3
    assert summary["machine"]["gpu"] == "NVIDIA GeForce RTX 3090"
    assert comparisons["partial_resume_over_discard"]["generated_token_factor"][
        "mean"
    ] == pytest.approx(0.7692307692)
    assert comparisons["streaming_delayed_over_wait"]["main_model_flops_factor"][
        "mean"
    ] == pytest.approx(1.0)
    assert comparisons["prefetch_delayed_over_ordinary"]["wall_time_factor"][
        "mean"
    ] < 0.9
    assert comparisons["delayed_acceptance_over_ordinary"][
        "exact_reward_call_factor"
    ]["mean"] < 0.8
    assert comparisons["replay_proposal_over_base"]["main_model_flops_factor"][
        "mean"
    ] == pytest.approx(1.003, abs=0.002)


def test_is_mh_reuse_summary_rejects_setting_drift() -> None:
    reports = deepcopy(_load(REPORTS))
    reports[0]["setting"]["chunk_tokens"] = 17

    with pytest.raises(ValueError, match="same non-seed setting"):
        build_summary(reports)

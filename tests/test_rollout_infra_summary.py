from copy import deepcopy
from pathlib import Path

import pytest

from experiments.summarize_rollout_infra import _load, build_summary


ROOT = Path(__file__).resolve().parents[1]
DECODE = tuple(
    ROOT / "results" / "infra" / name
    for name in (
        "rtx3090_transformers_decode_seed20260812.json",
        "rtx3090_transformers_decode_seed20260813.json",
        "rtx3090_transformers_decode_seed20260814.json",
    )
)
ALGORITHM = tuple(
    ROOT / "results" / "infra" / name
    for name in (
        "rtx3090_transformers_algorithms_seed20260812.json",
        "rtx3090_transformers_algorithms_seed20260813.json",
        "rtx3090_transformers_algorithms_seed20260814.json",
    )
)


def test_committed_rtx3090_runs_build_the_reported_paired_comparisons() -> None:
    summary = build_summary(_load(DECODE), _load(ALGORITHM))

    assert summary["runs"] == 3
    assert summary["machine"]["gpu"] == "NVIDIA GeForce RTX 3090"
    comparisons = summary["comparisons"]
    assert comparisons["static_tree_over_baseline"]["wall_time_factor"][
        "mean"
    ] == pytest.approx(2.162, abs=0.001)
    assert comparisons["load_aware_tree_over_baseline"]["wall_time_factor"][
        "mean"
    ] == pytest.approx(0.986, abs=0.001)
    assert comparisons["smc_reuse_over_no_reuse"]["main_model_flops_factor"][
        "mean"
    ] == pytest.approx(0.963, abs=0.001)
    assert comparisons["smc_reuse_over_no_reuse"]["wall_time_factor"][
        "mean"
    ] == pytest.approx(0.856, abs=0.001)


def test_rollout_summary_rejects_unpaired_seeds() -> None:
    decode = _load(DECODE)
    algorithm = deepcopy(_load(ALGORITHM))
    algorithm[0]["setting"]["seed"] = 17

    with pytest.raises(ValueError, match="same unique seeds"):
        build_summary(decode, algorithm)

from __future__ import annotations

import pytest

from online_speculation.hf_recycling_benchmark import _method, summarize


def _row(method, prompt, seconds, seed=1):
    return {"method": method, "workload": prompt, "seed": seed,
            "metrics": {"end_to_end_seconds": seconds, "output_tokens": 20,
                        "output_token_ids": [1, 2], "decoder_tokens_per_forward": 1.5}}


def test_ratio_of_sums_is_distinct_from_average_paired_ratio():
    data = [_row("static:8", "a", 10), _row("static:8", "b", 1),
            _row("tree:8:16", "a", 5), _row("tree:8:16", "b", 0.8)]
    result = summarize(data, samples=1000, seed=72)["tree:8:16"]
    assert result["ratio_of_total_e2e_seconds"] == pytest.approx(11/5.8)
    assert result["paired_e2e_speedup"]["mean"]["estimate"] == pytest.approx((2+1.25)/2)
    assert result["prompt_cluster_ratio_of_sums_speedup"]["estimate"] == pytest.approx(11/5.8)


def test_unpaired_rows_are_not_silently_included_in_paired_denominator():
    data = [_row("static:8", "a", 10), _row("tree:8:16", "a", 5), _row("tree:8:16", "missing", 100)]
    result = summarize(data, samples=100, seed=1)["tree:8:16"]
    assert result["pairs"] == 1
    assert result["ratio_of_total_e2e_seconds"] == 2
    assert result["absolute_e2e_tps"] == 4


def test_static_tree_secondary_baseline_and_ar_parser():
    data = [_row("static:8", "a", 10), _row("tree:8:16", "a", 5), _row("treebudget:8:16", "a", 4)]
    result = summarize(data, samples=100, seed=1, baseline_name="tree:8:16")
    assert result["treebudget:8:16"]["ratio_of_total_e2e_seconds"] == 1.25
    assert _method("ar") == ("ar", None, 0)

from __future__ import annotations

import numpy as np

from online_speculation.stage2_analysis import (
    analyze,
    bootstrap_interval,
    exact_two_sided_sign_p,
)


def test_bootstrap_interval_is_deterministic_and_contains_median() -> None:
    first = bootstrap_interval(np.array([1.1, 1.2, 1.3]), samples=2_000, seed=7)
    second = bootstrap_interval(np.array([1.1, 1.2, 1.3]), samples=2_000, seed=7)
    assert first == second
    assert first["estimate"] == 1.2
    assert first["ci_95_low"] <= first["estimate"] <= first["ci_95_high"]


def test_exact_sign_test_for_ten_wins() -> None:
    assert exact_two_sided_sign_p(10, 10) == 2 / 1024
    assert exact_two_sided_sign_p(5, 10) == 1.0


def test_analysis_preserves_pairing_and_applies_decision_gates() -> None:
    runs = []
    for repetition in range(4):
        common = {"repetition": repetition, "prompt_index": 0}
        runs.append(
            {
                **common,
                "label": "ar",
                "metrics": {
                    "decode_tokens_per_second": 10.0 + repetition,
                    "decoder_tokens_per_forward": 1.0,
                    "spec_acceptance_rate": 0.0,
                },
            }
        )
        runs.append(
            {
                **common,
                "label": "uno_b4",
                "metrics": {
                    "decode_tokens_per_second": 15.0 + 1.5 * repetition,
                    "decoder_tokens_per_forward": 1.4,
                    "spec_acceptance_rate": 0.5,
                },
            }
        )
    result = analyze(
        {
            "execution_backend": "huggingface_pytorch_kv_cache_fallback",
            "checkpoint": {"base_revision": "test"},
            "runs": runs,
        },
        bootstrap_samples=2_000,
        seed=11,
    )
    method = result["methods"]["uno_b4"]
    assert method["paired_speed_wins"] == 4
    assert method["tpf_median_bootstrap"]["ci_95_low"] > 1
    assert result["decision"]["algorithmic_tpf_reproduction_pass"]
    assert result["decision"]["hf_fallback_wallclock_speedup_pass"]
    assert not result["decision"]["official_runtime_speedup_tested"]

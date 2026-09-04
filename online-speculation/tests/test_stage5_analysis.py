from __future__ import annotations

from online_speculation.hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
)
from online_speculation.stage5_analysis import EXPECTED_PROMPTS, analyze


def _run(label: str, repetition: int, prompt_index: int, seed: int) -> dict:
    online = label != "static"
    diagnostics = None
    if online:
        diagnostics = {
            "parameter_isolation": {
                "trainable_base_parameter_tensors": 0,
                "base_optimizer_overlap": 0,
                "fast_trainable_parameters": 526_336,
            },
            "feedback_cycles": 2,
            "candidate_evaluation_cycles": 1,
            "active_head_evaluation_cycles": 5,
            "static_head_skip_cycles": 5,
            "candidate_promotion_attempts": 1,
            "candidate_promotions": 1,
            "candidate_rejections": 0,
            "future_static_resets": 0,
            "promotion_events": [
                {
                    "cycle": 8,
                    "action": "promote_candidate",
                    "future_rows_weight": 5.0,
                    "active_filtered_tv": 0.30,
                    "candidate_filtered_tv": 0.20,
                    "static_filtered_tv": 0.25,
                    "active_fast_weight_l2_after_action": 0.5,
                }
            ],
            "update_fraction_of_decode": 0.01,
            "feedback_materialization_seconds": 0.02,
            "head_forward_seconds": 0.01,
            "candidate_head_forward_seconds": 0.005,
            "update_seconds": 0.05,
            "update_attempts": 2,
            "updates_rolled_back": 0,
            "static_shadow_resets": 0,
        }
    return {
        "label": label,
        "repetition": repetition,
        "prompt_index": prompt_index,
        "seed": seed,
        "result": {
            "metrics": {
                "output_tokens": 512,
                "decode_tokens_per_second": 11.0 if online else 10.0,
                "decoder_tokens_per_forward": 1.1 if online else 1.0,
                "spec_acceptance_rate": 0.55 if online else 0.50,
                "peak_memory_allocated_bytes": 1_100 if online else 1_000,
                "decode_seconds": 9.0 if online else 10.0,
                "decoder_forwards": 90 if online else 100,
                "cycles": 10,
            },
            "diagnostics": diagnostics,
        },
    }


def test_stage5_analysis_enforces_frozen_design_and_deferred_audits() -> None:
    base_seed = 20260905
    runs = []
    for repetition in range(5):
        for prompt_index in range(3):
            order = ["static", "deferred_s40"]
            shift = (repetition + prompt_index) % 2
            order = order[shift:] + order[:shift]
            run_seed = base_seed + 1_000 * repetition + prompt_index
            runs.extend(
                _run(label, repetition, prompt_index, run_seed) for label in order
            )
    benchmark = {
        "execution_backend": "huggingface_pytorch_kv_cache_online_fast_residual",
        "checkpoint": {
            "base_revision": BASE_REVISION,
            "adapter_revision": ADAPTER_REVISION,
            "base_weight_sha256": BASE_WEIGHT_SHA256,
            "adapter_weight_sha256": ADAPTER_WEIGHT_SHA256,
        },
        "sampling": {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "ignore_stop": True,
        },
        "design": {
            "prompts": list(EXPECTED_PROMPTS),
            "block_size": 8,
            "update_strides": [40],
            "max_new_tokens": 512,
            "repetitions": 5,
            "seed": base_seed,
            "feedback_top_k": 50,
            "supervision": "on_policy",
            "position_discount": 0.97,
            "activation_mode": "deferred",
            "feedback_interval": 4,
            "candidate_evaluation_interval": 4,
            "promotion_margin": 0.0005,
            "future_reset_margin": 0.005,
            "fast": {"rank": 8, "alpha": 8.0, "learning_rate": 0.005},
        },
        "routing_probe": {"clean_rows_match": True, "noise_rows_changed": True},
        "runs": runs,
    }
    result = analyze(
        benchmark,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )
    assert result["integrity"]["safety_gate_pass"]
    assert result["integrity"]["frozen_design_pass"]
    assert result["integrity"]["deferred_action_rule_pass"]
    assert result["decision"]["real_model_learning_gate_pass"]
    assert result["decision"]["hf_system_gate_pass"]
    assert result["decision"]["all_stage5b_gates_pass"]
    assert result["sign_tests"]["tpf"]["wins"] == 15
    assert result["controller"]["actions"] == {"promote_candidate": 15}

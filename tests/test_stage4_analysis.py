from __future__ import annotations

from online_speculation.stage4_analysis import analyze


def _run(label: str, repetition: int) -> dict:
    speed = {"static": 10.0, "online_s10": 9.0, "online_s20": 10.0}[label]
    tpf = {"static": 1.0, "online_s10": 0.9, "online_s20": 1.0}[label]
    diagnostics = None
    if label != "static":
        diagnostics = {
            "parameter_isolation": {
                "trainable_base_parameter_tensors": 0,
                "base_optimizer_overlap": 0,
                "fast_trainable_parameters": 526_336,
            },
            "update_fraction_of_decode": 0.02,
            "feedback_materialization_seconds": 0.1,
            "head_forward_seconds": 0.01,
            "update_seconds": 0.2,
            "update_attempts": 2,
            "updates_rolled_back": 0,
            "static_shadow_resets": 0,
        }
    return {
        "label": label,
        "repetition": repetition,
        "prompt_index": 0,
        "result": {
            "metrics": {
                "output_tokens": 10,
                "decode_tokens_per_second": speed,
                "decoder_tokens_per_forward": tpf,
                "spec_acceptance_rate": tpf / 2,
                "peak_memory_allocated_bytes": 1_000,
                "decode_seconds": 10.0,
            },
            "diagnostics": diagnostics,
        },
    }


def test_stage4_analysis_enforces_safety_and_preregistered_decisions() -> None:
    runs = []
    for repetition in range(2):
        for label in ("static", "online_s10", "online_s20"):
            runs.append(_run(label, repetition))
    benchmark = {
        "execution_backend": "huggingface_pytorch_kv_cache_online_fast_residual",
        "checkpoint": {"base_id": "test", "adapter_id": "test"},
        "design": {"max_new_tokens": 10, "prompts": ["test"]},
        "routing_probe": {"clean_rows_match": True, "noise_rows_changed": True},
        "runs": runs,
    }
    result = analyze(
        benchmark,
        bootstrap_samples=1_000,
        bootstrap_seed=5,
    )
    assert result["integrity"]["safety_gate_pass"]
    assert not result["decision"]["primary_real_model_learning_gate_pass"]
    assert not result["decision"]["primary_hf_system_gate_pass"]
    assert result["decision"]["primary_hf_system_significant_slowdown"]
    assert not result["decision"]["exploratory_s20_tpf_interval_excludes_one"]

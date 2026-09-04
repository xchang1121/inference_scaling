from __future__ import annotations

from online_speculation.hf_stream_uno import DEFAULT_PROMPT, choose_snapshot
from online_speculation.hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
)
from online_speculation.stage6_analysis import analyze


def _metrics(*, tpf: float, tps: float, seconds: float) -> dict:
    return {
        "output_tokens": 512,
        "decoder_tokens_per_forward": tpf,
        "decode_tokens_per_second": tps,
        "spec_acceptance_rate": tpf / 2,
        "peak_memory_allocated_bytes": 1_000,
        "decode_seconds": seconds,
    }


def _diagnostics(*, initial_l2: float, final_l2: float, frozen: bool) -> dict:
    return {
        "reused_persistent_learner": True,
        "initial_fast_weight_l2": initial_l2,
        "final_fast_weight_l2": final_l2,
        "parameter_isolation": {
            "trainable_base_parameter_tensors": 0,
            "base_optimizer_overlap": 0,
            "fast_trainable_parameters": 526_336,
        },
        "feedback_cycles": 0 if frozen else 2,
        "feedback_items_created": 0 if frozen else 10,
        "feedback_items_discarded_at_end": 0,
        "update_attempts": 0 if frozen else 1,
        "updates_rolled_back": 0,
        "static_shadow_resets": 0,
        "update_seconds": 0.0 if frozen else 0.05,
        "feedback_materialization_seconds": 0.0 if frozen else 0.04,
        "head_forward_seconds": 0.01,
        "candidate_head_forward_seconds": 0.0,
    }


def _result(*, tpf: float, tps: float, seconds: float, l2: float) -> dict:
    return {
        "metrics": _metrics(tpf=tpf, tps=tps, seconds=seconds),
        "diagnostics": _diagnostics(
            initial_l2=l2,
            final_l2=l2,
            frozen=True,
        ),
    }


def test_stage6_analysis_reproduces_stream_selection_and_gates() -> None:
    base_seed = 20261005
    l2_pairs = ((0.0, 1.0), (1.0, 2.0), (2.0, 0.0), (0.0, 1.0))
    training_runs = []
    for request_index, (initial_l2, final_l2) in enumerate(l2_pairs):
        order = (
            ["static", "persistent_train"]
            if request_index % 2 == 0
            else ["persistent_train", "static"]
        )
        training_runs.append(
            {
                "request_index": request_index,
                "prompt_index": 0,
                "seed": base_seed + request_index,
                "order": order,
                "static": _metrics(tpf=1.0, tps=10.0, seconds=10.0),
                "persistent_train": {
                    "metrics": _metrics(tpf=1.0, tps=9.0, seconds=11.0),
                    "diagnostics": _diagnostics(
                        initial_l2=initial_l2,
                        final_l2=final_l2,
                        frozen=False,
                    ),
                },
            }
        )

    validation_static = []
    validation_snapshots = []
    snapshot_tpfs = (1.0, 1.1, 0.9, 0.9, 0.9)
    for repetition in range(5):
        validation_static.append(
            {
                "repetition": repetition,
                "prompt_index": 0,
                "metrics": _metrics(tpf=1.0, tps=10.0, seconds=10.0),
            }
        )
        for snapshot_index, tpf in enumerate(snapshot_tpfs):
            validation_snapshots.append(
                {
                    "snapshot_index": snapshot_index,
                    "repetition": repetition,
                    "prompt_index": 0,
                    "seed": base_seed + 100_000 + 1_000 * repetition,
                    "tpf_ratio_over_static": tpf,
                    "result": _result(
                        tpf=tpf,
                        tps=10.0,
                        seconds=10.0,
                        l2=0.0 if snapshot_index == 0 else 1.0,
                    ),
                }
            )
    scores = {index: tpf for index, tpf in enumerate(snapshot_tpfs)}
    selection = choose_snapshot(scores, minimum_gain=0.002)
    selection["all_snapshot_mean_tpf_ratios"] = {
        str(index): score for index, score in scores.items()
    }

    test_runs = []
    for repetition in range(10):
        order = (
            ["static", "persistent_frozen"]
            if repetition % 2 == 0
            else ["persistent_frozen", "static"]
        )
        test_runs.append(
            {
                "repetition": repetition,
                "prompt_index": 0,
                "seed": base_seed + 200_000 + 1_000 * repetition,
                "order": order,
                "static": _metrics(tpf=1.0, tps=10.0, seconds=10.0),
                "persistent_frozen": _result(
                    tpf=1.1,
                    tps=11.0,
                    seconds=9.0,
                    l2=1.0,
                ),
            }
        )

    benchmark = {
        "execution_backend": "huggingface_pytorch_kv_cache_stream_fast_residual",
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
            "train_prompts": [DEFAULT_PROMPT],
            "validation_prompts": [DEFAULT_PROMPT],
            "test_prompts": [DEFAULT_PROMPT],
            "training_requests": 4,
            "validation_repetitions": 5,
            "test_repetitions": 10,
            "max_new_tokens": 512,
            "block_size": 8,
            "update_stride": 40,
            "feedback_interval": 4,
            "feedback_top_k": 50,
            "selection_minimum_gain": 0.002,
            "fast": {"rank": 8, "alpha": 8.0, "learning_rate": 0.005},
            "seed": base_seed,
            "seed_partitions": {
                "train_offset": 0,
                "validation_offset": 100_000,
                "test_offset": 200_000,
            },
        },
        "routing_probe": {"clean_rows_match": True, "noise_rows_changed": True},
        "training_runs": training_runs,
        "validation": {
            "static_runs": validation_static,
            "snapshot_runs": validation_snapshots,
            "selection": selection,
        },
        "test_runs": test_runs,
        "analysis": {
            "selected_snapshot_frozen": {
                "head_unchanged_during_test": True,
                "head_sha256_before_test": "same",
                "head_sha256_after_test": "same",
            },
            "amortization": {
                "observed_training_increment_seconds": 4.0,
                "instrumented_training_cost_seconds": 0.4,
                "mean_future_request_saving_seconds": 1.0,
                "observed_break_even_future_requests": 4,
                "instrumented_break_even_future_requests": 1,
            },
        },
    }
    result = analyze(
        benchmark,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )
    assert result["integrity"]["safety_gate_pass"]
    assert result["integrity"]["selection_reproduction_pass"]
    assert result["decision"]["nonzero_selection_gate_pass"]
    assert result["decision"]["future_request_learning_gate_pass"]
    assert result["decision"]["frozen_serving_system_gate_pass"]
    assert result["decision"]["all_stage6c_gates_pass"]

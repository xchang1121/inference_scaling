"""Integrity checks and preregistered analysis for Stage-6C Stream-Uno."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .hf_stream_uno import DEFAULT_PROMPT, break_even_requests, choose_snapshot
from .hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
)
from .stage2_analysis import bootstrap_interval, exact_two_sided_sign_p


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _close(left: object, right: float, *, tolerance: float = 1e-12) -> bool:
    return isinstance(left, (int, float)) and math.isclose(
        float(left),
        right,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _intervals(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    return {
        "mean": bootstrap_interval(
            values,
            statistic=np.mean,
            samples=samples,
            seed=seed,
        ),
        "median": bootstrap_interval(
            values,
            statistic=np.median,
            samples=samples,
            seed=seed + 1,
        ),
    }


def _sign_test(values: np.ndarray, *, neutral: float) -> dict[str, int | float]:
    ties = np.isclose(values, neutral, rtol=0.0, atol=1e-12)
    wins = int(np.sum((values > neutral) & ~ties))
    losses = int(np.sum((values < neutral) & ~ties))
    non_ties = wins + losses
    return {
        "pairs": int(values.size),
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(ties)),
        "non_ties": non_ties,
        "two_sided_p_excluding_ties": (
            exact_two_sided_sign_p(wins, non_ties) if non_ties else 1.0
        ),
    }


def _expected_design_pass(benchmark: dict[str, Any]) -> bool:
    design = benchmark.get("design", {})
    sampling = benchmark.get("sampling", {})
    fast = design.get("fast", {}) if isinstance(design, dict) else {}
    return bool(
        design.get("train_prompts") == [DEFAULT_PROMPT]
        and design.get("validation_prompts") == [DEFAULT_PROMPT]
        and design.get("test_prompts") == [DEFAULT_PROMPT]
        and design.get("training_requests") == 4
        and design.get("validation_repetitions") == 5
        and design.get("test_repetitions") == 10
        and design.get("max_new_tokens") == 512
        and design.get("block_size") == 8
        and design.get("update_stride") == 40
        and design.get("feedback_interval") == 4
        and design.get("feedback_top_k") == 50
        and _close(design.get("selection_minimum_gain"), 0.002)
        and design.get("seed") == 20261005
        and design.get("seed_partitions")
        == {
            "train_offset": 0,
            "validation_offset": 100_000,
            "test_offset": 200_000,
        }
        and fast.get("rank") == 8
        and _close(fast.get("alpha"), 8.0)
        and _close(fast.get("learning_rate"), 0.005)
        and _close(sampling.get("temperature"), 1.0)
        and sampling.get("top_k") == 50
        and _close(sampling.get("top_p"), 0.95)
        and sampling.get("ignore_stop") is True
    )


def _expected_checkpoint_pass(benchmark: dict[str, Any]) -> bool:
    checkpoint = benchmark.get("checkpoint", {})
    return bool(
        checkpoint.get("base_revision") == BASE_REVISION
        and checkpoint.get("adapter_revision") == ADAPTER_REVISION
        and checkpoint.get("base_weight_sha256") == BASE_WEIGHT_SHA256
        and checkpoint.get("adapter_weight_sha256") == ADAPTER_WEIGHT_SHA256
    )


def _isolation_pass(result: dict[str, Any]) -> bool:
    diagnostics = result["diagnostics"]
    isolation = diagnostics["parameter_isolation"]
    return bool(
        diagnostics["reused_persistent_learner"]
        and int(isolation["trainable_base_parameter_tensors"]) == 0
        and int(isolation["base_optimizer_overlap"]) == 0
        and int(isolation["fast_trainable_parameters"]) == 526_336
    )


def _frozen_pass(result: dict[str, Any]) -> bool:
    diagnostics = result["diagnostics"]
    return bool(
        _isolation_pass(result)
        and int(diagnostics["feedback_cycles"]) == 0
        and int(diagnostics["feedback_items_created"]) == 0
        and int(diagnostics["feedback_items_discarded_at_end"]) == 0
        and int(diagnostics["update_attempts"]) == 0
        and _close(
            diagnostics["initial_fast_weight_l2"],
            float(diagnostics["final_fast_weight_l2"]),
            tolerance=1e-7,
        )
    )


def analyze(
    benchmark: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if benchmark.get("execution_backend") != (
        "huggingface_pytorch_kv_cache_stream_fast_residual"
    ):
        raise ValueError("input is not a Stage-6 Stream-Uno benchmark.")
    training_runs = benchmark.get("training_runs")
    validation = benchmark.get("validation")
    test_runs = benchmark.get("test_runs")
    if not isinstance(training_runs, list) or not isinstance(validation, dict):
        raise ValueError("benchmark is missing training or validation records.")
    if not isinstance(test_runs, list) or not test_runs:
        raise ValueError("benchmark contains no test runs.")

    expected_tokens = int(benchmark["design"]["max_new_tokens"])
    output_lengths = []
    isolation_results = []
    frozen_results = []
    for run in training_runs:
        output_lengths.extend(
            (
                int(run["static"]["output_tokens"]),
                int(run["persistent_train"]["metrics"]["output_tokens"]),
            )
        )
        isolation_results.append(run["persistent_train"])
    for run in validation["static_runs"]:
        output_lengths.append(int(run["metrics"]["output_tokens"]))
    for run in validation["snapshot_runs"]:
        result = run["result"]
        output_lengths.append(int(result["metrics"]["output_tokens"]))
        isolation_results.append(result)
        frozen_results.append(result)
    for run in test_runs:
        output_lengths.extend(
            (
                int(run["static"]["output_tokens"]),
                int(run["persistent_frozen"]["metrics"]["output_tokens"]),
            )
        )
        isolation_results.append(run["persistent_frozen"])
        frozen_results.append(run["persistent_frozen"])

    run_counts_pass = bool(
        len(training_runs) == 4
        and len(validation["static_runs"]) == 5
        and len(validation["snapshot_runs"]) == 25
        and len(test_runs) == 10
    )
    output_lengths_pass = all(length == expected_tokens for length in output_lengths)
    finite_pass = _all_finite(benchmark)
    design_pass = _expected_design_pass(benchmark)
    checkpoint_pass = _expected_checkpoint_pass(benchmark)
    routing = benchmark.get("routing_probe", {})
    routing_pass = bool(
        routing.get("clean_rows_match") and routing.get("noise_rows_changed")
    )
    isolation_pass = all(_isolation_pass(result) for result in isolation_results)
    frozen_runs_pass = all(_frozen_pass(result) for result in frozen_results)

    continuity_pass = True
    previous_final = 0.0
    for request_index, run in enumerate(training_runs):
        diagnostics = run["persistent_train"]["diagnostics"]
        continuity_pass &= int(run["request_index"]) == request_index
        continuity_pass &= _close(
            diagnostics["initial_fast_weight_l2"],
            previous_final,
            tolerance=1e-7,
        )
        previous_final = float(diagnostics["final_fast_weight_l2"])

    static_validation = {
        (int(run["repetition"]), int(run["prompt_index"])): run["metrics"]
        for run in validation["static_runs"]
    }
    snapshot_ratios: dict[int, list[float]] = {index: [] for index in range(5)}
    validation_keys: dict[int, set[tuple[int, int]]] = {
        index: set() for index in range(5)
    }
    validation_ratio_fields_pass = True
    for run in validation["snapshot_runs"]:
        index = int(run["snapshot_index"])
        key = (int(run["repetition"]), int(run["prompt_index"]))
        if index not in snapshot_ratios or key not in static_validation:
            validation_ratio_fields_pass = False
            continue
        ratio = float(run["result"]["metrics"]["decoder_tokens_per_forward"]) / float(
            static_validation[key]["decoder_tokens_per_forward"]
        )
        snapshot_ratios[index].append(ratio)
        validation_keys[index].add(key)
        validation_ratio_fields_pass &= _close(
            run["tpf_ratio_over_static"], ratio, tolerance=1e-12
        )
    complete_validation_grid_pass = all(
        len(snapshot_ratios[index]) == 5 and len(validation_keys[index]) == 5
        for index in range(5)
    )
    zero_snapshot_pass = all(
        math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for ratio in snapshot_ratios[0]
    )
    validation_scores = {
        index: float(np.mean(ratios)) for index, ratios in snapshot_ratios.items()
    }
    recomputed_selection = choose_snapshot(
        validation_scores,
        minimum_gain=float(benchmark["design"]["selection_minimum_gain"]),
    )
    recorded_selection = validation["selection"]
    selection_reproduction_pass = all(
        recorded_selection.get(name) == recomputed_selection[name]
        for name in recomputed_selection
    ) and all(
        _close(
            recorded_selection["all_snapshot_mean_tpf_ratios"][str(index)],
            score,
            tolerance=1e-12,
        )
        for index, score in validation_scores.items()
    )

    base_seed = int(benchmark["design"]["seed"])
    order_and_seed_pass = True
    for request_index, run in enumerate(training_runs):
        expected_order = (
            ["static", "persistent_train"]
            if request_index % 2 == 0
            else ["persistent_train", "static"]
        )
        order_and_seed_pass &= run["order"] == expected_order
        order_and_seed_pass &= int(run["seed"]) == base_seed + request_index
    for repetition, run in enumerate(test_runs):
        expected_order = (
            ["static", "persistent_frozen"]
            if repetition % 2 == 0
            else ["persistent_frozen", "static"]
        )
        order_and_seed_pass &= int(run["repetition"]) == repetition
        order_and_seed_pass &= int(run["prompt_index"]) == 0
        order_and_seed_pass &= run["order"] == expected_order
        order_and_seed_pass &= (
            int(run["seed"]) == base_seed + 200_000 + 1_000 * repetition
        )

    frozen_audit = benchmark["analysis"]["selected_snapshot_frozen"]
    head_frozen_pass = bool(
        frozen_audit["head_unchanged_during_test"]
        and frozen_audit["head_sha256_before_test"]
        == frozen_audit["head_sha256_after_test"]
    )
    safety_pass = all(
        (
            run_counts_pass,
            output_lengths_pass,
            finite_pass,
            design_pass,
            checkpoint_pass,
            routing_pass,
            isolation_pass,
            frozen_runs_pass,
            continuity_pass,
            validation_ratio_fields_pass,
            complete_validation_grid_pass,
            zero_snapshot_pass,
            selection_reproduction_pass,
            order_and_seed_pass,
            head_frozen_pass,
        )
    )

    paired_rows = []
    tpf_ratios = []
    speed_ratios = []
    acceptance_deltas = []
    memory_deltas = []
    savings_seconds = []
    for run in test_runs:
        static = run["static"]
        frozen = run["persistent_frozen"]["metrics"]
        tpf_ratio = float(frozen["decoder_tokens_per_forward"]) / float(
            static["decoder_tokens_per_forward"]
        )
        speed_ratio = float(frozen["decode_tokens_per_second"]) / float(
            static["decode_tokens_per_second"]
        )
        acceptance_delta = float(frozen["spec_acceptance_rate"]) - float(
            static["spec_acceptance_rate"]
        )
        memory_delta = float(frozen["peak_memory_allocated_bytes"]) - float(
            static["peak_memory_allocated_bytes"]
        )
        saving = float(static["decode_seconds"]) - float(frozen["decode_seconds"])
        paired_rows.append(
            {
                "repetition": int(run["repetition"]),
                "seed": int(run["seed"]),
                "tpf_ratio": tpf_ratio,
                "decode_tps_ratio": speed_ratio,
                "acceptance_rate_delta": acceptance_delta,
                "peak_memory_delta_bytes": memory_delta,
                "serving_time_saving_seconds": saving,
            }
        )
        tpf_ratios.append(tpf_ratio)
        speed_ratios.append(speed_ratio)
        acceptance_deltas.append(acceptance_delta)
        memory_deltas.append(memory_delta)
        savings_seconds.append(saving)

    arrays = {
        "paired_tpf_ratio": np.asarray(tpf_ratios, dtype=np.float64),
        "paired_decode_tps_ratio": np.asarray(speed_ratios, dtype=np.float64),
        "paired_acceptance_rate_delta": np.asarray(acceptance_deltas, dtype=np.float64),
        "paired_peak_memory_delta_bytes": np.asarray(memory_deltas, dtype=np.float64),
        "serving_time_saving_seconds": np.asarray(savings_seconds, dtype=np.float64),
    }
    statistics = {
        name: _intervals(
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 10 * index,
        )
        for index, (name, values) in enumerate(arrays.items(), start=1)
    }

    observed_training_increment = sum(
        float(run["persistent_train"]["metrics"]["decode_seconds"])
        - float(run["static"]["decode_seconds"])
        for run in training_runs
    )
    instrumented_training_cost = sum(
        sum(
            float(run["persistent_train"]["diagnostics"][name])
            for name in (
                "update_seconds",
                "feedback_materialization_seconds",
                "head_forward_seconds",
                "candidate_head_forward_seconds",
            )
        )
        for run in training_runs
    )
    mean_saving = float(np.mean(arrays["serving_time_saving_seconds"]))
    recomputed_amortization = {
        "observed_training_increment_seconds": observed_training_increment,
        "instrumented_training_cost_seconds": instrumented_training_cost,
        "mean_future_request_saving_seconds": mean_saving,
        "observed_break_even_future_requests": break_even_requests(
            observed_training_increment, mean_saving
        ),
        "instrumented_break_even_future_requests": break_even_requests(
            instrumented_training_cost, mean_saving
        ),
    }
    recorded_amortization = benchmark["analysis"]["amortization"]
    amortization_reproduction_pass = all(
        (
            recorded_amortization[name] == expected
            if expected is None or isinstance(expected, int)
            else _close(recorded_amortization[name], expected, tolerance=1e-9)
        )
        for name, expected in recomputed_amortization.items()
    )

    training_updates = sum(
        int(run["persistent_train"]["diagnostics"]["update_attempts"])
        for run in training_runs
    )
    training_rollbacks = sum(
        int(run["persistent_train"]["diagnostics"]["updates_rolled_back"])
        for run in training_runs
    )
    training_resets = sum(
        int(run["persistent_train"]["diagnostics"]["static_shadow_resets"])
        for run in training_runs
    )
    selected_nonzero = bool(recorded_selection["nonzero_snapshot_selected"])
    tpf_primary = statistics["paired_tpf_ratio"]["mean"]
    speed_primary = statistics["paired_decode_tps_ratio"]["mean"]
    learning_gate = tpf_primary["ci_95_low"] > 1.0
    system_gate = speed_primary["ci_95_low"] > 1.0
    return {
        "schema_version": 1,
        "input_backend": benchmark["execution_backend"],
        "input_checkpoint": benchmark["checkpoint"],
        "integrity": {
            "run_counts_pass": run_counts_pass,
            "output_lengths_checked": len(output_lengths),
            "output_lengths_pass": output_lengths_pass,
            "all_numeric_values_finite": finite_pass,
            "frozen_design_pass": design_pass,
            "checkpoint_hash_and_revision_pass": checkpoint_pass,
            "routing_pass": routing_pass,
            "persistent_isolation_records": len(isolation_results),
            "parameter_isolation_pass": isolation_pass,
            "training_state_continuity_pass": continuity_pass,
            "frozen_no_learning_pass": frozen_runs_pass,
            "complete_validation_grid_pass": complete_validation_grid_pass,
            "validation_ratio_fields_pass": validation_ratio_fields_pass,
            "zero_snapshot_exact_tpf_pass": zero_snapshot_pass,
            "selection_reproduction_pass": selection_reproduction_pass,
            "method_order_and_seed_partition_pass": order_and_seed_pass,
            "selected_head_sha256_frozen_pass": head_frozen_pass,
            "amortization_reproduction_pass": amortization_reproduction_pass,
            "safety_gate_pass": safety_pass and amortization_reproduction_pass,
        },
        "bootstrap": {
            "preregistered_primary_statistic": "arithmetic mean of paired ratios",
            "robustness_statistic": "median of paired ratios",
            "method": "paired percentile bootstrap",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "validation": {
            "snapshot_mean_tpf_ratios": {
                str(index): score for index, score in validation_scores.items()
            },
            "recomputed_selection": recomputed_selection,
            "validation_to_test_tpf_optimism_gap": (
                float(recomputed_selection["best_validation_mean_tpf_ratio"])
                - float(np.mean(arrays["paired_tpf_ratio"]))
            ),
        },
        "test_statistics": statistics,
        "sign_tests": {
            "tpf": _sign_test(arrays["paired_tpf_ratio"], neutral=1.0),
            "decode_tps": _sign_test(arrays["paired_decode_tps_ratio"], neutral=1.0),
        },
        "training": {
            "requests": len(training_runs),
            "update_attempts": training_updates,
            "update_rollbacks": training_rollbacks,
            "same_buffer_static_resets": training_resets,
            **recomputed_amortization,
        },
        "paired_observations": paired_rows,
        "decision": {
            "safety_gate_pass": safety_pass and amortization_reproduction_pass,
            "nonzero_selection_gate_pass": selected_nonzero,
            "future_request_learning_gate_pass": learning_gate,
            "frozen_serving_system_gate_pass": system_gate,
            "all_stage6c_gates_pass": (
                safety_pass
                and amortization_reproduction_pass
                and selected_nonzero
                and learning_gate
                and system_gate
            ),
            "tpf_mean_interval_excludes_one_as_harm": (tpf_primary["ci_95_high"] < 1.0),
            "speed_mean_interval_excludes_one_as_harm": (
                speed_primary["ci_95_high"] < 1.0
            ),
            "official_runtime_tested": False,
            "full_uno_adapter_updated_online": False,
        },
        "scope_warning": (
            "This is one trained rank-8 residual on one repeated prompt, evaluated over "
            "ten new sampling seeds on the Windows HF fallback. The prompt and pipeline "
            "were tuned in an earlier pilot; only the seed stream is held out."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(
        benchmark,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

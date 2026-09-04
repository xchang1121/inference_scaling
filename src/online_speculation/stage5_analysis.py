"""Integrity checks and preregistered analysis for Stage-5B Deferred Online Uno."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .hf_online_uno import choose_deferred_action, summarize_online_runs
from .hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
)
from .stage2_analysis import bootstrap_interval, exact_two_sided_sign_p


PRIMARY_LABEL = "deferred_s40"
EXPECTED_PROMPTS = (
    "Explain in three concise paragraphs why speculative decoding can be lossless.",
    "Implement a production-quality Python LRU cache from scratch. Explain the "
    "invariants, analyze complexity, and include tests for edge cases.",
    "请严格推导为什么 Metropolis-Hastings 算法以目标分布为平稳分布，说明 detailed "
    "balance、不可约性与非周期性的作用，并给出一个离散状态空间例子。",
)


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _close(value: object, expected: float) -> bool:
    return isinstance(value, (int, float)) and math.isclose(
        float(value),
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError("distribution summary requires at least one scalar.")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


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
        design.get("prompts") == list(EXPECTED_PROMPTS)
        and design.get("block_size") == 8
        and design.get("update_strides") == [40]
        and design.get("max_new_tokens") == 512
        and design.get("repetitions") == 5
        and design.get("feedback_top_k") == 50
        and design.get("supervision") == "on_policy"
        and _close(design.get("position_discount"), 0.97)
        and design.get("activation_mode") == "deferred"
        and design.get("feedback_interval") == 4
        and design.get("candidate_evaluation_interval") == 4
        and _close(design.get("promotion_margin"), 0.0005)
        and _close(design.get("future_reset_margin"), 0.005)
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


def analyze(
    benchmark: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if benchmark.get("execution_backend") != (
        "huggingface_pytorch_kv_cache_online_fast_residual"
    ):
        raise ValueError("input is not a real-checkpoint Online Uno benchmark.")
    runs = benchmark.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("benchmark contains no runs.")

    pairs: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (int(run["repetition"]), int(run["prompt_index"]))
        label = str(run["label"])
        if label in pairs[key]:
            raise ValueError(f"duplicate {label} run for paired key {key}.")
        pairs[key][label] = run
    required_labels = {"static", PRIMARY_LABEL}
    if any(set(methods) != required_labels for methods in pairs.values()):
        raise ValueError("every paired key must contain static and deferred_s40.")

    design = benchmark["design"]
    expected_pairs = int(design["repetitions"]) * len(design["prompts"])
    pair_count_pass = len(pairs) == expected_pairs == 15 and len(runs) == 30
    expected_tokens = int(design["max_new_tokens"])
    output_lengths_pass = all(
        int(run["result"]["metrics"]["output_tokens"]) == expected_tokens
        for run in runs
    )
    finite_pass = _all_finite(runs)
    design_pass = _expected_design_pass(benchmark)
    checkpoint_pass = _expected_checkpoint_pass(benchmark)
    routing = benchmark.get("routing_probe", {})
    routing_pass = bool(
        routing.get("clean_rows_match") and routing.get("noise_rows_changed")
    )

    online_runs = [run for run in runs if run["label"] == PRIMARY_LABEL]
    isolation_records = [run["result"]["diagnostics"] for run in online_runs]
    isolation_pass = all(
        int(diagnostics["parameter_isolation"]["trainable_base_parameter_tensors"]) == 0
        and int(diagnostics["parameter_isolation"]["base_optimizer_overlap"]) == 0
        and int(diagnostics["parameter_isolation"]["fast_trainable_parameters"])
        == 526_336
        for diagnostics in isolation_records
    )
    cycle_accounting_pass = all(
        int(diagnostics["active_head_evaluation_cycles"])
        + int(diagnostics["static_head_skip_cycles"])
        == int(run["result"]["metrics"]["cycles"])
        for run, diagnostics in zip(online_runs, isolation_records)
    )
    decision_accounting_pass = all(
        int(diagnostics["candidate_promotions"])
        + int(diagnostics["candidate_rejections"])
        + int(diagnostics["future_static_resets"])
        == int(diagnostics["candidate_promotion_attempts"])
        == len(diagnostics["promotion_events"])
        for diagnostics in isolation_records
    )
    candidate_evidence_pass = all(
        int(diagnostics["candidate_evaluation_cycles"]) > 0
        and all(
            float(event["future_rows_weight"]) > 0
            for event in diagnostics["promotion_events"]
        )
        for diagnostics in isolation_records
    )
    action_rule_pass = all(
        event["action"]
        == choose_deferred_action(
            active_tv=float(event["active_filtered_tv"]),
            candidate_tv=float(event["candidate_filtered_tv"]),
            static_tv=float(event["static_filtered_tv"]),
            promotion_margin=float(design["promotion_margin"]),
            reset_margin=float(design["future_reset_margin"]),
        )
        for diagnostics in isolation_records
        for event in diagnostics["promotion_events"]
    )
    method_order_pass = True
    run_offset = 0
    base_seed = int(design["seed"])
    for repetition in range(5):
        for prompt_index in range(3):
            expected_order = ["static", PRIMARY_LABEL]
            shift = (repetition + prompt_index) % 2
            expected_order = expected_order[shift:] + expected_order[:shift]
            pair_runs = runs[run_offset : run_offset + 2]
            method_order_pass &= [run["label"] for run in pair_runs] == expected_order
            method_order_pass &= all(
                int(run["seed"]) == base_seed + 1_000 * repetition + prompt_index
                for run in pair_runs
            )
            run_offset += 2

    safety_pass = all(
        (
            pair_count_pass,
            output_lengths_pass,
            finite_pass,
            design_pass,
            checkpoint_pass,
            routing_pass,
            isolation_pass,
            cycle_accounting_pass,
            decision_accounting_pass,
            candidate_evidence_pass,
            action_rule_pass,
            method_order_pass,
        )
    )

    pair_rows = []
    tpf_ratios = []
    speed_ratios = []
    acceptance_deltas = []
    memory_deltas = []
    unaccounted_seconds = []
    unaccounted_fractions = []
    explicit_fractions = []
    for (repetition, prompt_index), methods in sorted(pairs.items()):
        static_metrics = methods["static"]["result"]["metrics"]
        online_result = methods[PRIMARY_LABEL]["result"]
        online_metrics = online_result["metrics"]
        diagnostics = online_result["diagnostics"]
        tpf_ratio = float(online_metrics["decoder_tokens_per_forward"]) / float(
            static_metrics["decoder_tokens_per_forward"]
        )
        speed_ratio = float(online_metrics["decode_tokens_per_second"]) / float(
            static_metrics["decode_tokens_per_second"]
        )
        acceptance_delta = float(online_metrics["spec_acceptance_rate"]) - float(
            static_metrics["spec_acceptance_rate"]
        )
        memory_delta = float(online_metrics["peak_memory_allocated_bytes"]) - float(
            static_metrics["peak_memory_allocated_bytes"]
        )
        explicit_seconds = sum(
            float(diagnostics[name])
            for name in (
                "update_seconds",
                "feedback_materialization_seconds",
                "head_forward_seconds",
                "candidate_head_forward_seconds",
            )
        )
        online_seconds = float(online_metrics["decode_seconds"])
        expected_base_seconds = float(static_metrics["decode_seconds"]) * (
            float(online_metrics["decoder_forwards"])
            / float(static_metrics["decoder_forwards"])
        )
        residual_seconds = online_seconds - expected_base_seconds - explicit_seconds
        pair_rows.append(
            {
                "repetition": repetition,
                "prompt_index": prompt_index,
                "tpf_ratio": tpf_ratio,
                "decode_tps_ratio": speed_ratio,
                "acceptance_rate_delta": acceptance_delta,
                "peak_memory_delta_bytes": memory_delta,
                "static_decoder_forwards": int(static_metrics["decoder_forwards"]),
                "deferred_decoder_forwards": int(online_metrics["decoder_forwards"]),
                "candidate_promotions": int(diagnostics["candidate_promotions"]),
                "candidate_rejections": int(diagnostics["candidate_rejections"]),
                "future_static_resets": int(diagnostics["future_static_resets"]),
                "explicit_online_seconds": explicit_seconds,
                "coarse_unaccounted_seconds": residual_seconds,
            }
        )
        tpf_ratios.append(tpf_ratio)
        speed_ratios.append(speed_ratio)
        acceptance_deltas.append(acceptance_delta)
        memory_deltas.append(memory_delta)
        explicit_fractions.append(explicit_seconds / online_seconds)
        unaccounted_seconds.append(residual_seconds)
        unaccounted_fractions.append(residual_seconds / online_seconds)

    tpf_array = np.asarray(tpf_ratios, dtype=np.float64)
    speed_array = np.asarray(speed_ratios, dtype=np.float64)
    acceptance_array = np.asarray(acceptance_deltas, dtype=np.float64)
    memory_array = np.asarray(memory_deltas, dtype=np.float64)
    primary_statistics = {
        "paired_tpf_ratio": _intervals(
            tpf_array,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        ),
        "paired_decode_speed_ratio": _intervals(
            speed_array,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 3,
        ),
        "paired_acceptance_rate_delta": _intervals(
            acceptance_array,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 5,
        ),
        "paired_peak_memory_delta_bytes": _intervals(
            memory_array,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 7,
        ),
    }

    prompt_breakdown: dict[str, Any] = {}
    for prompt_index, prompt in enumerate(design["prompts"]):
        selected = [row for row in pair_rows if row["prompt_index"] == prompt_index]
        prompt_tpf = np.asarray([row["tpf_ratio"] for row in selected])
        prompt_speed = np.asarray([row["decode_tps_ratio"] for row in selected])
        prompt_acceptance = np.asarray(
            [row["acceptance_rate_delta"] for row in selected]
        )
        local_seed = bootstrap_seed + 1_000 * prompt_index
        prompt_breakdown[str(prompt_index)] = {
            "prompt": prompt,
            "partially_tuned_workload": prompt_index == 0,
            "pairs": len(selected),
            "paired_tpf_ratio": _intervals(
                prompt_tpf,
                samples=bootstrap_samples,
                seed=local_seed + 101,
            ),
            "paired_decode_speed_ratio": _intervals(
                prompt_speed,
                samples=bootstrap_samples,
                seed=local_seed + 103,
            ),
            "paired_acceptance_rate_delta": _intervals(
                prompt_acceptance,
                samples=bootstrap_samples,
                seed=local_seed + 105,
            ),
            "tpf_sign_test": _sign_test(prompt_tpf, neutral=1.0),
            "speed_sign_test": _sign_test(prompt_speed, neutral=1.0),
        }

    actions = Counter(
        event["action"]
        for diagnostics in isolation_records
        for event in diagnostics["promotion_events"]
    )
    promoted_advantages = [
        float(event["active_filtered_tv"]) - float(event["candidate_filtered_tv"])
        for diagnostics in isolation_records
        for event in diagnostics["promotion_events"]
        if event["action"] == "promote_candidate"
    ]
    actions_by_prompt = {}
    for prompt_index in range(len(design["prompts"])):
        prompt_actions = Counter(
            event["action"]
            for run in online_runs
            if int(run["prompt_index"]) == prompt_index
            for event in run["result"]["diagnostics"]["promotion_events"]
        )
        actions_by_prompt[str(prompt_index)] = dict(prompt_actions)

    component_fractions = {}
    for output_name, diagnostic_name in (
        ("update", "update_seconds"),
        ("feedback_materialization", "feedback_materialization_seconds"),
        ("active_head", "head_forward_seconds"),
        ("candidate_head", "candidate_head_forward_seconds"),
    ):
        component_fractions[output_name] = _distribution(
            [
                float(run["result"]["diagnostics"][diagnostic_name])
                / float(run["result"]["metrics"]["decode_seconds"])
                for run in online_runs
            ]
        )
    total_cycles = sum(int(run["result"]["metrics"]["cycles"]) for run in online_runs)
    total_active_cycles = sum(
        int(run["result"]["diagnostics"]["active_head_evaluation_cycles"])
        for run in online_runs
    )
    total_candidate_eval_cycles = sum(
        int(run["result"]["diagnostics"]["candidate_evaluation_cycles"])
        for run in online_runs
    )
    update_rollbacks = sum(
        int(run["result"]["diagnostics"]["updates_rolled_back"]) for run in online_runs
    )
    update_attempts = sum(
        int(run["result"]["diagnostics"]["update_attempts"]) for run in online_runs
    )

    tpf_mean = primary_statistics["paired_tpf_ratio"]["mean"]
    speed_mean = primary_statistics["paired_decode_speed_ratio"]["mean"]
    tpf_median = primary_statistics["paired_tpf_ratio"]["median"]
    speed_median = primary_statistics["paired_decode_speed_ratio"]["median"]
    learning_gate = tpf_mean["ci_95_low"] > 1.0
    system_gate = speed_mean["ci_95_low"] > 1.0
    return {
        "schema_version": 1,
        "input_backend": benchmark["execution_backend"],
        "input_checkpoint": benchmark["checkpoint"],
        "integrity": {
            "runs": len(runs),
            "paired_workloads": len(pairs),
            "pair_count_pass": pair_count_pass,
            "expected_output_tokens_per_run": expected_tokens,
            "output_lengths_pass": output_lengths_pass,
            "all_numeric_values_finite": finite_pass,
            "frozen_design_pass": design_pass,
            "checkpoint_hash_and_revision_pass": checkpoint_pass,
            "routing_pass": routing_pass,
            "parameter_isolation_records": len(isolation_records),
            "parameter_isolation_pass": isolation_pass,
            "cycle_accounting_pass": cycle_accounting_pass,
            "decision_accounting_pass": decision_accounting_pass,
            "candidate_future_evidence_pass": candidate_evidence_pass,
            "deferred_action_rule_pass": action_rule_pass,
            "method_order_and_seed_pass": method_order_pass,
            "safety_gate_pass": safety_pass,
        },
        "bootstrap": {
            "preregistered_primary_statistic": "arithmetic mean of paired ratios",
            "robustness_statistic": "median of paired ratios",
            "method": "paired percentile bootstrap",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "descriptive_runner_summary_median": summarize_online_runs(
            runs,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "primary_statistics": primary_statistics,
        "prompt_breakdown": prompt_breakdown,
        "sign_tests": {
            "tpf": _sign_test(tpf_array, neutral=1.0),
            "decode_tps": _sign_test(speed_array, neutral=1.0),
        },
        "controller": {
            "candidate_decisions": int(sum(actions.values())),
            "actions": dict(actions),
            "actions_by_prompt": actions_by_prompt,
            "runs_with_at_least_one_promotion": sum(
                int(run["result"]["diagnostics"]["candidate_promotions"] > 0)
                for run in online_runs
            ),
            "promoted_candidate_filtered_tv_advantage": (
                _distribution(promoted_advantages) if promoted_advantages else None
            ),
            "update_attempts": update_attempts,
            "update_rollbacks": update_rollbacks,
            "update_rollback_fraction": (
                update_rollbacks / update_attempts if update_attempts else 0.0
            ),
            "total_cycles": total_cycles,
            "active_head_evaluation_cycles": total_active_cycles,
            "static_head_skip_cycles": total_cycles - total_active_cycles,
            "active_head_evaluation_fraction": total_active_cycles / total_cycles,
            "candidate_evaluation_cycles": total_candidate_eval_cycles,
            "candidate_evaluation_fraction": total_candidate_eval_cycles / total_cycles,
        },
        "cost": {
            "component_fraction_of_online_decode": component_fractions,
            "explicit_online_fraction_of_online_decode": _distribution(
                explicit_fractions
            ),
            "coarse_unaccounted_seconds": _distribution(unaccounted_seconds),
            "coarse_unaccounted_fraction_of_online_decode": _distribution(
                unaccounted_fractions
            ),
            "coarse_unaccounted_definition": (
                "online decode seconds minus paired static seconds scaled by the "
                "decoder-forward-count ratio, minus instrumented update/feedback/head "
                "seconds; this is noisy and is not a causal kernel attribution"
            ),
        },
        "paired_observations": pair_rows,
        "decision": {
            "preregistered_primary": PRIMARY_LABEL,
            "safety_gate_pass": safety_pass,
            "real_model_learning_gate_pass": learning_gate,
            "hf_system_gate_pass": system_gate,
            "all_stage5b_gates_pass": safety_pass and learning_gate and system_gate,
            "tpf_mean_interval_excludes_one_as_harm": tpf_mean["ci_95_high"] < 1.0,
            "speed_mean_interval_excludes_one_as_harm": (
                speed_mean["ci_95_high"] < 1.0
            ),
            "speed_mean_interval_excludes_one_as_gain": system_gate,
            "median_tpf_learning_gate_pass": tpf_median["ci_95_low"] > 1.0,
            "median_hf_system_gate_pass": speed_median["ci_95_low"] > 1.0,
            "official_runtime_tested": False,
            "full_uno_adapter_updated_online": False,
        },
        "scope_warning": (
            "Fifteen pairs cover five seeds on three fixed prompts and one Windows HF "
            "fallback backend. Prompt 0 was used in engineering pilots. The controller "
            "updates a rank-8 post-logit residual, not the full diffusion LoRA."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260905)
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

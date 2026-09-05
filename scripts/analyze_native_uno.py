"""Audit complete native Uno studies without suppressing output divergences."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


def validate(payload):
    if (not payload["completed"] or payload["stage"] != "complete" or payload["error"]
            or not payload["parameters_frozen"] or not payload["parameters_frozen_after"]
            or payload["engine_stats"]["preemptions"] != 0
            or not payload["environment"]["tracked_source_clean"]):
        raise RuntimeError("study must complete with frozen parameters and clean source, no preemptions")
    design = payload["design"]
    if payload.get("frozen_weights_before") != payload.get("frozen_weights_after"):
        raise RuntimeError("frozen teacher/offline Uno hash mismatch")
    methods = {str(m) for m in design["methods"]}
    expected = {(name, design["seed"] + p * 100 + r, m)
                for p, (name, _) in enumerate(design["workloads"])
                for r in range(design["repetitions"]) for m in methods}
    seen = Counter((r["workload"], r["seed"], str(r["block_size"])) for r in payload["records"])
    if set(seen) != expected or any(n != 1 for n in seen.values()):
        raise RuntimeError("incomplete, duplicated or altered study matrix")
    for row in payload["records"]:
        ids = row["output"]["token_ids"]
        seconds = row["end_to_end_seconds"]
        stats = row["output"]["stats"]
        if (not math.isfinite(seconds) or seconds <= 0 or row["output_tokens"] != len(ids)
                or not 0 <= len(ids) <= design["max_new_tokens"]
                or any(type(n) is not int or n < 0 for n in ids)
                or not math.isclose(row["e2e_tps"], len(ids) / seconds, rel_tol=1e-12)):
            raise RuntimeError("invalid token count or timing")
        # Official _run_prefill emits one token but does NOT update seq.stats.
        if stats["forwards"] < 1 or stats["accepts"] != design["max_new_tokens"] - 1:
            raise RuntimeError("incomplete committed token budget or invalid official stats")
        if row.get("online") is not None:
            diagnostic = row["online"]
            if diagnostic.get("algorithm") == "last_mlp_online_lora":
                events = diagnostic["events"]
                if (diagnostic["teacher_weight_updates"] != 0
                        or diagnostic["offline_uno_weight_updates"] != 0
                        or diagnostic["cycles"] * 2 != stats["forwards"]
                        or diagnostic["optimizer_steps"] != len(events)
                        or diagnostic["model_weight_updates"] != len(events)
                        or [e["version"] for e in events] != list(range(1, len(events) + 1))
                        or any(not math.isfinite(e["kl_before"]) or e["grad_norm"] <= 0
                               or e["right_norm"] <= 0 or not 1 <= e["rows"] <= 7
                               or e["cycle"] % diagnostic["stride"] != 0 for e in events)):
                    raise RuntimeError("invalid real online LoRA update audit")
                continue
            policy, cycles = diagnostic["policy"], diagnostic["cycles"]
            if (policy["pending"] is not None or policy["optimizer_steps"] != 0
                    or policy["model_weight_updates"] != 0
                    or diagnostic["additional_cuda_synchronizations"] != 0
                    or sum(c["tokens"] for c in cycles) != stats["accepts"]
                    or len(cycles) * 2 != stats["forwards"]):
                raise RuntimeError("online feedback does not reconcile with official output stats")
            expected_counts = Counter(c["width"] for c in cycles)
            for width in policy["widths"]:
                if int(policy["completed_epochs_by_width"][str(width)]) != expected_counts[width] // policy["epoch_cycles"]:
                    raise RuntimeError("epoch count mismatch")
    return len(expected)


def summarize(payload):
    validate(payload)
    rows = payload["records"]
    methods = [str(m) for m in payload["design"]["methods"]]
    pairs = {(r["workload"], r["seed"], str(r["block_size"])): r for r in rows}
    summaries = {}
    for method in methods:
        selected = [r for r in rows if str(r["block_size"]) == method]
        total_time = sum(r["end_to_end_seconds"] for r in selected)
        tokens = sum(r["output_tokens"] for r in selected)
        accepted = sum(r["output"]["stats"]["accepts"] for r in selected)
        forwards = sum(r["output"]["stats"]["forwards"] for r in selected)
        mismatches = []
        for row in selected:
            actual = row["output"]["token_ids"]
            reference = pairs[(row["workload"], row["seed"], "1")]["output"]["token_ids"]
            if actual != reference:
                first = next((j for j, (a, b) in enumerate(zip(actual, reference)) if a != b), min(len(actual), len(reference)))
                mismatches.append({"workload": row["workload"], "seed": row["seed"], "first_difference": first,
                                   "actual_length": len(actual), "ar_length": len(reference)})
        summary = {"runs": len(selected), "returned_tokens": tokens, "total_e2e_seconds": total_time,
                   "aggregate_e2e_tps": tokens / total_time, "official_tpf_decode_only": accepted / forwards,
                   "ar_exact_matches": len(selected) - len(mismatches), "ar_mismatches": mismatches,
                   "cuda_graph_hits": sum(r["cuda_graph_hits"] for r in selected),
                   "cuda_graph_misses": sum(r["cuda_graph_misses"] for r in selected)}
        if method == "shadow8":
            summary["same_width_B8_token_matches"] = sum(
                r["output"]["token_ids"] == pairs[(r["workload"], r["seed"], "8")]["output"]["token_ids"]
                for r in selected)
        if method == "fast8":
            summary["same_width_B8_token_matches"] = sum(
                r["output"]["token_ids"] == pairs[(r["workload"], r["seed"], "8")]["output"]["token_ids"]
                for r in selected)
            summary["optimizer_steps"] = sum(r["online"]["optimizer_steps"] for r in selected)
            summary["update_seconds"] = sum(r["online"]["update_seconds"] for r in selected)
        if method == "online":
            summary["cycle_width_counts"] = dict(Counter(c["width"] for r in selected for c in r["online"]["cycles"]))
            summary["cycle_reasons"] = dict(Counter(c["reason"] for r in selected for c in r["online"]["cycles"]))
            summary["completed_policy_epochs"] = sum(r["online"]["policy"]["completed_epochs"] for r in selected)
            summary["instrumented_choice_update_seconds"] = sum(r["online"]["instrumented_choice_update_seconds"] for r in selected)
        summaries[method] = summary
    comparisons = {}
    adaptive = "fast8" if "fast8" in methods else "online"
    if adaptive in methods:
        workloads = [n for n, _ in payload["design"]["workloads"]]
        for comparator in [m for m in methods if m != adaptive]:
            def ratio(selected_workloads, comparator=comparator):
                online_tokens = online_seconds = fixed_tokens = fixed_seconds = 0
                for name in selected_workloads:
                    for row in rows:
                        if row["workload"] != name:
                            continue
                        if row["block_size"] == adaptive:
                            online_tokens += row["output_tokens"]
                            online_seconds += row["end_to_end_seconds"]
                        elif str(row["block_size"]) == comparator:
                            fixed_tokens += row["output_tokens"]
                            fixed_seconds += row["end_to_end_seconds"]
                return (online_tokens / online_seconds) / (fixed_tokens / fixed_seconds)

            rng = random.Random(20270905)
            samples = sorted(ratio(rng.choices(workloads, k=len(workloads))) for _ in range(3000))
            paired = [pairs[(r["workload"], r["seed"], comparator)]["end_to_end_seconds"] / r["end_to_end_seconds"]
                      for r in rows if r["block_size"] == adaptive]
            comparisons[comparator] = {"aggregate_tps_ratio": ratio(workloads),
                                       "prompt_cluster_bootstrap_95": [samples[74], samples[2924]],
                                       "paired_fixed_time_over_online_time": paired,
                                       "paired_geomean_time_ratio": math.exp(sum(map(math.log, paired)) / len(paired))}
    return {"valid_runs": len(rows), "methods": summaries, "online_over_fixed": comparisons,
            "gpu_after_counts": dict(Counter(r["gpu_after"] for r in rows)),
            "scope": "engineering pilot; four reused prompt clusters; not confirmatory; no retroactive equivalence margin",
            "bitwise_exactness_is_not_inferred_from_theory": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an earlier audit")
    raw = args.input.read_bytes()
    result = summarize(json.loads(raw))
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    result["source"] = str(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

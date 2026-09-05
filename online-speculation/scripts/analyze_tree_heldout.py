"""Audit a completed frozen tree study without editing or dropping raw runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def memory_clock(snapshot):
    return int(snapshot.split(",")[2].strip().split()[0])


def validate_study(payload):
    """Require every frozen method/prompt/seed exactly once, including whole missing arms."""
    if not payload["completed"]:
        raise RuntimeError("Study did not finish; do not analyze it as completed")
    design = payload["design"]
    methods = design["methods"].split(",")
    workloads = [name for name, _ in design["workloads"]]
    if len(set(methods)) != len(methods) or len(set(workloads)) != len(workloads):
        raise RuntimeError("Duplicate methods or workload names in frozen design")
    expected = {
        (method, name, design["seed"] + prompt_index * 100 + rep)
        for method in methods
        for prompt_index, name in enumerate(workloads)
        for rep in range(design["repetitions"])
    }
    actual = Counter((r["method"], r["workload"], r["seed"]) for r in payload["records"])
    if not expected or set(actual) != expected or any(n != 1 for n in actual.values()):
        raise RuntimeError("Incomplete or duplicated study matrix")
    for row in payload["records"]:
        metric = row["metrics"]
        seconds = metric["end_to_end_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("Non-positive or non-finite complete-call timing")
        if metric["output_tokens"] != len(metric["output_token_ids"]):
            raise RuntimeError("Token count disagrees with recorded output IDs")
        if design["fixed_output_tokens"] and metric["output_tokens"] != design["max_new_tokens"]:
            raise RuntimeError("Run did not produce the frozen output budget")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="treebudget:8:16")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    expected = validate_study(payload)
    records = payload["records"]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)
    if args.candidate not in grouped:
        raise ValueError("requested candidate is absent from the frozen study")
    ar = {(r["workload"], r["seed"]): r for r in grouped["ar"]}
    table, differences, post_low = {}, [], []
    for method, rows in grouped.items():
        seconds = sum(r["metrics"]["end_to_end_seconds"] for r in rows)
        tokens = sum(r["metrics"]["output_tokens"] for r in rows)
        decode_tokens = sum(r["metrics"]["decoder_tokens"] for r in rows)
        forwards = sum(r["metrics"]["decoder_forwards"] for r in rows)
        table[method] = {"runs": len(rows), "total_tokens": tokens, "total_seconds": seconds,
                         "absolute_e2e_tps": tokens/seconds, "aggregate_decode_tpf": decode_tokens/forwards}
        for row in rows:
            reference = ar[(row["workload"], row["seed"])]["metrics"]["output_token_ids"]
            output = row["metrics"]["output_token_ids"]
            if output != reference:
                first = next((i for i, (a, b) in enumerate(zip(output, reference)) if a != b), min(len(output), len(reference)))
                differences.append({"method": method, "workload": row["workload"], "seed": row["seed"], "first_difference": first})
            if memory_clock(row["gpu_after"]) < 9000:
                post_low.append({"method": method, "workload": row["workload"], "seed": row["seed"], "snapshot": row["gpu_after"]})
    result = {
        "schema_version": 2, "raw_file": str(args.input), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "study_scope": payload["scope"], "candidate_method": args.candidate,
        "runs": len(records), "expected_runs": expected, "removed_runs": 0,
        "table": table, "token_differences_from_ar": differences,
        "all_token_ids_equal_to_ar": not differences,
        "post_run_memory_clocks_by_method": {m: dict(Counter(memory_clock(r["gpu_after"]) for r in rows)) for m, rows in grouped.items()},
        "low_post_clock_records": post_low,
        "low_post_clock_exclusively_ar": bool(post_low) and all(r["method"] == "ar" for r in post_low),
        "confirmatory_clock_gate_passed": not post_low,
        "timing_scope": "Pilot/design-selection data; never a confirmatory claim. " if payload["scope"] == "pilot" else (
            "Conservative downgrade under preregistered clock rule; descriptive held-out measurements only. "
            "AR-associated state differences are recorded, not excluded or proven causal."
            if post_low else "Passed the recorded post-run memory-clock stability check."
        ),
        "candidate_diagnostics": {
            "all_parameters_frozen": all(r["diagnostics"]["model_parameters_frozen"] for r in grouped[args.candidate]),
            "all_optimizer_steps_zero": all(r["diagnostics"]["optimizer_steps"] == 0 for r in grouped[args.candidate]),
            "all_feedback_completed": all(not r["diagnostics"]["budget_controller"].get("pending_feedback", False) for r in grouped[args.candidate]),
            "tree_shapes": dict(sum((Counter(r["diagnostics"]["tree_shapes"]) for r in grouped[args.candidate]), Counter())),
            "selection_reasons": dict(sum((Counter(r["diagnostics"]["budget_reasons"]) for r in grouped[args.candidate]), Counter())),
        },
        "online_comparisons": {
            name: comparisons[args.candidate]
            for name, comparisons in payload["secondary_summaries"].items() if args.candidate in comparisons
        },
        "online_vs_linear8": payload["summary"][args.candidate],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("runs", "table", "all_token_ids_equal_to_ar", "confirmatory_clock_gate_passed")}, indent=2))


if __name__ == "__main__":
    main()

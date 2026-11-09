"""Recompute generation, selection and matched-pair audits without a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from summarize_flare_gap import bootstrap, grade, read_jsonl
from inference_scaling.retrieval.scaled import SearchLedger
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, select_metropolized, weigh

EOS, CLOSE = 248046, 248069


def average(values):
    return statistics.fmean(values) if values else 0.


def audit_search(unit, seed):
    detail = unit["detail"]
    if not isinstance(detail, dict) or "root_ids" not in detail:
        return {}
    calls = [c for c in unit["calls"] if c["purpose"] == "search"]
    backend = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=EOS), model_id="offline")
    ledger = SearchLedger(backend, detail["root_ids"], (CLOSE,), 8192, seed, 8192)
    ledger.calls = calls
    checks = {"root_hash": token_hash(ledger.root) == detail["root_sha256"],
        "search_work": sum(len(c["token_ids"]) for c in calls) == detail["search_work"],
        "unchanged_context": all(tuple(c["prefix_ids"][:len(ledger.root)]) == ledger.root for c in calls)}
    selected = detail["selected_suffix_ids"]
    body_end = next((i for i, token in enumerate(selected) if token in {EOS, CLOSE}), len(selected))
    body = selected[:body_end]
    answer_calls = [c for c in unit["calls"] if c["purpose"] == "answer"]
    if EOS in selected:
        checks["thinking_eos_not_reopened"] = not answer_calls and unit["result"]["eos_inside_thinking"]
    else:
        expected = list(ledger.root)+body+[CLOSE]
        for ai, call in enumerate(answer_calls):
            checks[f"answer_{ai}_selected_prefix"] = call["prefix_ids"] == expected
            expected += call["token_ids"]
    checks["answer_ids"] = [t for c in answer_calls for t in c["token_ids"]] == unit["result"]["answer_ids"]
    if body:
        checks["selected_reason_suffix"] = unit["result"]["reason_ids"][-len(body):] == body
    if detail["method"] == "sc_bon":
        scores = ledger.reward(ledger.root, [c["token_ids"] for c in calls])
        checks["bon_scores"] = all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
            for x, y in zip(scores, detail["scores"], strict=True))
        chosen = max(range(len(scores)), key=lambda i: (scores[i], -i))
        checks["bon_selection"] = chosen == detail["selected_index"] and selected == calls[chosen]["token_ids"]
    elif detail["method"] == "block_met_sc":
        kept = ()
        for si, record in enumerate(detail["steps"]):
            fixed = kept[:record["generated_length_before"]]
            remaining = detail["config"]["total_length"]-len(fixed)
            width = min(detail["config"]["block_size"], remaining)
            m, k = detail["config"]["candidate_count"], detail["config"]["rollout_count"]
            carried = bool(kept)
            bound = (m-int(carried))*(width+k*(remaining-width))+int(carried)*(k-1)*(remaining-width)
            checks[f"pool_{si}_reservation"] = bound == record["reserved_work"]
            blocks, leaves = [], []
            valid_scores = True
            for candidate in record["candidates"]:
                block, group = tuple(candidate["token_ids"]), []
                for leaf in candidate["rollouts"]:
                    suffix = tuple(leaf["token_ids"])
                    score, = ledger.reward(ledger.root, [fixed+block+suffix])
                    valid_scores &= math.isclose(score, leaf["reward"], rel_tol=1e-12, abs_tol=1e-12)
                    valid_scores &= math.isclose(score/detail["config"]["reward_temperature"], leaf["log_weight"], rel_tol=1e-12, abs_tol=1e-12)
                    group.append(RolloutEvaluation(suffix, leaf["reward"], leaf["log_weight"]))
                blocks.append(block); leaves.append(group)
            candidates = weigh(blocks, leaves)
            mi, ki = select_metropolized(candidates, SeedStream(seed), "conditional_is", si, group_blocks=True)
            checks[f"pool_{si}_scores"] = valid_scores
            checks[f"pool_{si}_group_weights"] = all(math.isclose(c.log_weight, r["log_weight"], rel_tol=1e-12, abs_tol=1e-12)
                for c, r in zip(candidates, record["candidates"], strict=True))
            checks[f"pool_{si}_selection"] = mi == record["selected_index"] and ki == record["completion_index"]
            checks[f"pool_{si}_work"] = sum(len(c["token_ids"]) for c in calls
                if c["request_id"].startswith(f"conditional-is:step:{si}:")) == record["actual_work"] <= record["reserved_work"]
            kept = fixed+candidates[mi].token_ids+candidates[mi].rollouts[ki].token_ids
        checks["is_selected_sequence"] = tuple(selected) == kept
        if detail["stop"] == "whole_pool_budget":
            remaining = detail["config"]["total_length"]-detail["fixed_tokens"]
            width = min(detail["config"]["block_size"], remaining)
            next_bound = (m-1)*(width+k*(remaining-width))+(k-1)*(remaining-width)
            checks["is_budget_stop_justified"] = next_bound > 8192-detail["common_work"]-detail["search_work"]
        else:
            checks["is_terminal_stop_justified"] = detail["fixed_tokens"] >= len(selected)
    else:
        checks["single_selected_sequence"] = len(calls) == 1 and selected == calls[0]["token_ids"]
    return checks


def summarize(args):
    rows = read_jsonl(args.input)
    metadata = next(r for r in rows if r["kind"] == "metadata")
    complete = [r for r in rows if r["kind"] == "completed"]
    unit_rows = [r for r in rows if r["kind"] == "execution_unit"]
    units = {r["unit_id"]: r for r in unit_rows}
    measurements = [r for r in rows if r["kind"] == "measurement"]
    gold = {r["id"]: r for r in read_jsonl(args.gold)} if args.gold else {}
    cfg = metadata["config"]
    audits = {}
    for uid, unit in units.items():
        checks = {}
        for ci, call in enumerate(unit["calls"]):
            ids = call["token_ids"]
            checks[f"{ci}_hash"] = token_hash(call["prefix_ids"]) == call["prefix_sha256"]
            checks[f"{ci}_length"] = 0 < len(ids) <= call["max_new_tokens"]
            checks[f"{ci}_context"] = len(call["prefix_ids"])+len(ids) <= cfg["context_tokens"]
            checks[f"{ci}_logprobs"] = len(call["token_logprobs"]) == len(ids)
            checks[f"{ci}_terminal"] = (EOS not in ids or ids[-1] == EOS) and (CLOSE not in ids or ids[-1] == CLOSE)
            if call["purpose"] not in {"gap_plan", "support_check"}:
                checks[f"{ci}_statistics"] = len(call["token_statistics"]) == len(ids) and all(v is not None and math.isfinite(v) for v in call["token_statistics"])
        seed = SeedStream(metadata["seed"]).derive("question", uid.split(":")[0])
        checks.update(audit_search(unit, seed))
        checks["no_scoring_forwards"] = unit["cost"].get("score_forward_token_slots", 0) == 0
        checks["no_cancellations"] = unit["cost"].get("cancelled_requests", 0) == 0
        checks["native_graphs"] = not unit["calls"] or unit.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) > 0
        checks["native_covers_logical"] = unit.get("graph_execution", {}).get("forward_token_slots", 0) >= unit["cost"].get("generation_forward_token_slots", 0)
        checks["no_preemptions"] = unit["cost"].get("preemptions", 0) == 0
        audits[uid] = checks
    condition_checks, graded = {}, []
    for m in measurements:
        result = m["result"]
        used = [units[u] for u in m["execution_units"]]
        calls = [c for u in used for c in u["calls"]]
        work = sum(len(c["token_ids"]) for c in calls if c["purpose"] != "answer")
        answers = sum(len(c["token_ids"]) for c in calls if c["purpose"] == "answer")
        key = m["question_id"]+":"+m["arm"]
        condition_checks[key] = {"work_accounting": work == result["work_tokens"] <= cfg["work_tokens"],
            "answer_accounting": answers == result["answer_tokens"] <= cfg["answer_tokens"],
            "unique_units": len(set(m["execution_units"])) == len(m["execution_units"])}
        target = gold.get(m["question_id"])
        scored = grade(result["answer_text"], target["answer"]) if target else {"f1": None, "exact_match": None}
        graded.append({"question_id": m["question_id"], "cohort": m["cohort"], "arm": m["arm"],
            **scored, "work_tokens": work, "answer_tokens": answers,
            "seconds": sum(u["seconds"] for u in used),
            "native_pflops": sum(u.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) for u in used)/1e15,
            **{k: result[k] for k in ("empty_answer", "eos_inside_thinking", "immediate_thinking_end", "immediate_eos", "forced_thinking_end", "evidence_tokens", "provisional_eos")},
            "search_bypassed": m["search_bypassed_natural_end"],
            "search_steps": len(m["search"].get("steps", [])),
            "raw_search_eos": m["search"].get("raw_search_eos_count", 0),
            "raw_probe_eos": sum(EOS in c["token_ids"] for c in calls if c["purpose"] == "thinking"),
            "raw_immediate_search_eos": sum(bool(c["token_ids"]) and c["token_ids"][0] == EOS for c in calls if c["purpose"] == "search"),
            "raw_immediate_search_close": m["search"].get("raw_immediate_close_count", 0),
            "unused_work_budget": cfg["work_tokens"]-work})
    pairs = {}
    grouped = defaultdict(dict)
    for m in measurements:
        if m["context"] != "closed":
            grouped[(m["question_id"], m["context"], m["method"])][m["content"]] = m
    for key, pair in grouped.items():
        if set(pair) != {"relevant", "unrelated"}:
            continue
        real, noise = pair["relevant"], pair["unrelated"]
        checks = {"evidence_same_length": real["result"]["evidence_tokens"] == noise["result"]["evidence_tokens"]}
        if not real["result"]["evidence_tokens"]:
            checks["no_gate_exact_same_result"] = real["result"] == noise["result"] and real["execution_units"] == noise["execution_units"]
        else:
            rroot, nroot = real["search"]["root_ids"], noise["search"]["root_ids"]
            checks["root_same_length"] = len(rroot) == len(nroot)
            checks["different_content"] = rroot != nroot
            # Injection payload differences must exactly match body replacement.
            ri = next(e for e in real["result"]["events"] if e["kind"] == "injection")
            ni = next(e for e in noise["result"]["events"] if e["kind"] == "injection")
            body_positions = {p for b in ri["packet"]["bodies"] for p in range(b["start"], b["stop"])}
            checks["wrapper_exact"] = all(a == b for p, (a, b) in enumerate(zip(ri["payload_ids"], ni["payload_ids"], strict=True)) if p not in body_positions)
            offset = ri["confirmed_offset"]
            checks["same_confirmed_thinking"] = (not offset or rroot[-offset:] == nroot[-offset:]) and offset == ni["confirmed_offset"]
        pairs[":".join(key)] = checks
    roots = defaultdict(list)
    for m in measurements:
        if "root_ids" in m["search"]:
            roots[(m["question_id"], m["context"], m["content"])].append(m["search"]["root_ids"])
    for key, group in roots.items():
        pairs[":".join(key)+":methods"] = {"same_frozen_search_root": all(r == group[0] for r in group)}
    summaries = []
    for cohort in sorted({g["cohort"] for g in graded}):
        for arm in sorted({g["arm"] for g in graded}):
            group = [g for g in graded if g["cohort"] == cohort and g["arm"] == arm]
            if not group:
                continue
            summaries.append({"cohort": cohort, "arm": arm, "n": len(group),
                "f1_percent": average([g["f1"]*100 for g in group if g["f1"] is not None]) if any(g["f1"] is not None for g in group) else None,
                "exact_match_percent": 100*average([g["exact_match"] for g in group if g["exact_match"] is not None]) if any(g["exact_match"] is not None for g in group) else None,
                **{k: average([g[k] for g in group]) for k in ("work_tokens", "seconds", "native_pflops", "unused_work_budget", "search_steps")},
                **{k: sum(g[k] for g in group) for k in ("empty_answer", "eos_inside_thinking", "immediate_thinking_end", "immediate_eos", "forced_thinking_end", "search_bypassed", "raw_search_eos", "raw_probe_eos", "raw_immediate_search_eos", "raw_immediate_search_close", "provisional_eos")}})
    lookup = {(g["question_id"], g["arm"]): g for g in graded}
    contrasts = []
    for cohort in sorted({g["cohort"] for g in graded}):
        ids = sorted({g["question_id"] for g in graded if g["cohort"] == cohort})
        for context in ("closed.none", "static.relevant", "dynamic.relevant"):
            for method in ("sc_bon", "block_met_sc"):
                arm, base = context+"/"+method, context+"/single"
                diffs = [100*(lookup[(i, arm)]["f1"]-lookup[(i, base)]["f1"]) for i in ids
                    if (i, arm) in lookup and (i, base) in lookup and lookup[(i, arm)]["f1"] is not None]
                if diffs:
                    contrasts.append({"cohort": cohort, "contrast": arm+" minus "+base, **bootstrap(diffs)})
        for method in ("single", "sc_bon", "block_met_sc"):
            arm, base = "dynamic.relevant/"+method, "static.relevant/"+method
            diffs = [100*(lookup[(i, arm)]["f1"]-lookup[(i, base)]["f1"]) for i in ids
                if (i, arm) in lookup and (i, base) in lookup and lookup[(i, arm)]["f1"] is not None]
            if diffs:
                contrasts.append({"cohort": cohort, "contrast": arm+" minus "+base, **bootstrap(diffs)})
    failed_units = [r for r in rows if r["kind"] == "failed_unit"]
    ids = set(gold) if gold else {m["question_id"] for m in measurements}
    expected_grid = {(i, context+"."+content+"/"+method) for i in ids for context in metadata["contexts"]
        for method in metadata["methods"] for content in (("none",) if context == "closed" else ("relevant", "unrelated"))}
    actual_grid = {(m["question_id"], m["arm"]) for m in measurements}
    global_audit = {"run": {"unique_execution_units": len(unit_rows) == len(units),
        "unique_conditions": len(actual_grid) == len(measurements), "no_failed_units": not failed_units,
        "complete_grid": not complete or actual_grid == expected_grid}}
    audit_groups = [("unit", audits), ("condition", condition_checks), ("pair", pairs), ("global", global_audit)]
    failed = [(group, uid, k) for group, mapping in audit_groups
        for uid, checks in mapping.items() for k, ok in checks.items() if not ok]
    report = {"complete": bool(complete), "conditions": len(measurements), "expected_conditions": metadata["expected_conditions"],
        "execution_units": len(units), "audit_checks": sum(len(c) for _, mapping in audit_groups for c in mapping.values()),
        "audit_failures": failed, "failed_units": failed_units, "summary": summaries, "contrasts": contrasts,
        "actual_study": {"seconds": sum(u["seconds"] for u in units.values()),
            "work_tokens": sum(len(c["token_ids"]) for u in units.values() for c in u["calls"] if c["purpose"] != "answer"),
            "native_pflops": sum(u.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) for u in units.values())/1e15}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [("summary.json", report), ("audit.json", {"units": audits, "conditions": condition_checks, "pairs": pairs, "global": global_audit})]:
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for name, records in [("graded.jsonl", graded), ("gate-events.jsonl", [{"unit_id": uid, "event": e} for uid, u in units.items() if uid.endswith(":probe") for e in u["result"]["events"] if e["kind"] in {"gap_check", "support_check", "search_checkpoint", "rollback"}])]:
        (args.output/name).write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in records), encoding="utf-8")
    if summaries:
        with (args.output/"summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    cohort_note = ", ".join(cohort+"="+str(len({g["question_id"] for g in graded if g["cohort"] == cohort})) for cohort in sorted({g["cohort"] for g in graded}))
    text = ["# Retrieval + inference scaling pilot", "", f"Completed: {bool(complete)}; {len(measurements)}/{metadata['expected_conditions']} conditions; {len(units)} unique execution units; {report['audit_checks']} checks, {len(failed)} failures.", "", f"F1 is normalized answer token F1 (%), not exact-answer accuracy. Question counts: {cohort_note}; one seed. Equal generated-token caps do not imply equal realized work or FLOPs. Normalized EM is in summary.csv.", "", "Times cover measured execution units, including online dynamic lookup. Model startup, corpus-index construction and offline static-packet preparation are excluded. PFLOPs are native dense-padded graph estimates, not hardware-counter measurements of every operation.", "", "| Cohort | Evidence / method | F1 % | Mean work | Mean s | Mean PFLOPs | Empty | Thinking EOS | Forced close |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        score = f"{s['f1_percent']:.3f}" if s['f1_percent'] is not None else "not scored"
        text.append(f"| {s['cohort']} | {s['arm']} | {score} | {s['work_tokens']:.1f} | {s['seconds']:.2f} | {s['native_pflops']:.3f} | {s['empty_answer']} | {s['eos_inside_thinking']} | {s['forced_thinking_end']} |")
    text += ["", "Paired question-bootstrap contrasts (percentage points):", ""]
    for c in contrasts:
        text.append(f"- {c['cohort']}: {c['contrast']}: {c['difference']:+.3f}, 95% [{c['bootstrap_95'][0]:+.3f}, {c['bootstrap_95'][1]:+.3f}], n={c['n_questions']}.")
    (args.output/"REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("complete", "conditions", "expected_conditions", "execution_units", "audit_checks", "audit_failures")}, ensure_ascii=False))
    if failed or (complete and len(measurements) != metadata["expected_conditions"]):
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--gold", type=Path)
    p.add_argument("--output", type=Path, required=True)
    summarize(p.parse_args())

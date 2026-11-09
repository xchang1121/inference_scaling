"""Offline structural/cost audit and paired generative TechQA diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import statistics

from inference_scaling.retrieval.ablation import static_prompt
from inference_scaling.retrieval.stepwise import parse_plan, parse_support, step_char_boundaries, token_hash

EOS, CLOSE, OPEN, END = 248046, 248069, 248068, 248046
ARMS = ("closed.none", "static.relevant", "static.unrelated", "legacy_sc.relevant", "legacy_sc.unrelated",
        "step_sc.relevant", "step_sc.unrelated", "flare_gap.relevant", "flare_gap.unrelated",
        "flare_gap_rebuild.relevant", "flare_gap_rebuild.unrelated")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def grade(answer, target):
    def normalize(text):
        return re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", text.lower())).split()
    a, b = normalize(answer), normalize(target)
    shared = sum((Counter(a) & Counter(b)).values())
    return {"f1": 2*shared/(len(a)+len(b)) if a or b else 1., "exact_match": a == b}


def bootstrap(values):
    values = list(values)
    if not values: return None
    rng = random.Random(20261109)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return {"difference": mean(values), "bootstrap_95": [samples[124], samples[4874]], "n_questions": len(values)}


def audit_unit(unit, expected_prefix, cfg):
    checks = {}
    main_prefix = list(expected_prefix)
    phase = "thinking"
    for ci, call in enumerate(unit["calls"]):
        key = f"call_{ci}"
        ids = call["token_ids"]
        checks[key+"_hash"] = token_hash(call["prefix_ids"]) == call["prefix_sha256"]
        checks[key+"_length"] = 0 < len(ids) <= call["max_new_tokens"]
        checks[key+"_context"] = len(call["prefix_ids"])+len(ids) <= cfg["context_tokens"]
        checks[key+"_logprobs"] = len(ids) == len(call["token_logprobs"])
        if call["purpose"] in {"gap_plan", "support_check"}:
            checks[key+"_auxiliary_cap"] = call["max_new_tokens"] <= cfg["auxiliary_tokens"]
            checks[key+"_auxiliary_no_thinking"] = OPEN not in ids and CLOSE not in ids
            checks[key+"_auxiliary_no_statistics"] = not call["token_statistics"]
            continue
        if call["purpose"] == "answer" and phase == "thinking":
            if not main_prefix or main_prefix[-1] != CLOSE:
                main_prefix.append(CLOSE)
            phase = "answer"
        checks[key+"_exact_prefix"] = call["prefix_ids"] == main_prefix
        checks[key+"_statistics"] = len(call["token_statistics"]) == len(ids) and all(v is not None for v in call["token_statistics"])
        checks[key+"_eos_last"] = EOS not in ids or ids[-1] == EOS
        if call["purpose"] == "thinking":
            checks[key+"_close_last"] = CLOSE not in ids or ids[-1] == CLOSE
        main_prefix += ids
    native = unit.get("graph_execution", {})
    checks["native_graph_recorded"] = not unit["calls"] or native.get("estimated_dense_padded_flops", 0) > 0
    checks["native_covers_logical"] = native.get("forward_token_slots", 0) >= unit["cost"].get("generation_forward_token_slots", 0)
    checks["no_score_or_cancel"] = all(unit["cost"].get(k, 0) == 0 for k in ("score_forward_token_slots", "cancelled_requests", "preemptions"))
    return checks


def summarize(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.generation)
    metadata = next(r for r in rows if r["kind"] == "metadata")
    cfg = metadata["config"]
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in prepared["cases"]}
    gold = {g["id"]: g for g in read_jsonl(args.gold)} if args.gold else {}
    units = {r["unit_id"]: r for r in rows if r["kind"] == "execution_unit"}
    measurements = [r for r in rows if r["kind"] == "measurement"]
    assert len(units) == sum(r["kind"] == "execution_unit" for r in rows), "duplicate unit"
    assert len({(r["question_id"], r["arm"]) for r in measurements}) == len(measurements), "duplicate condition"
    audits, graded, by_key = [], [], {}
    unit_audits = {}
    for uid, unit in units.items():
        qid, policy, part = uid.split(":")
        case = cases[qid]
        if part == "common":
            prefix = case["prompt_ids"]
        else:
            injection = next(e for e in unit["result"]["events"] if e["kind"] == "injection")
            stem_policy = "flare_gap" if policy == "flare_gap_rebuild" else policy
            common = units[qid+":"+stem_policy+":common"]["result"]
            if injection["position"] == "assistant_inline":
                prefix = case["prompt_ids"]+common["reason_ids"]+injection["payload_ids"]
            else:
                # The pinned tokenizer encodes two newlines as token 271. The
                # actual separator is also checked from saved continuation IDs.
                prefix = list(static_prompt(case["prompt_ids"], injection["payload_ids"], END, (271,)))+common["reason_ids"]
            assert injection["continuation_prefix_ids"] == prefix, uid
        unit_audits[uid] = audit_unit(unit, prefix, cfg)
        gate_packet = None
        for ei, event in enumerate(unit["result"]["events"]):
            if event["kind"] == "gap_check":
                plan, error = parse_plan(event["raw"])
                unit_audits[uid][f"gate_{ei}_plan_recomputed"] = plan == event["plan"] and error == event["error"]
                gate_packet = event.get("packet")
            elif event["kind"] == "support_check":
                support, error = parse_support(event["raw"], gate_packet)
                unit_audits[uid][f"gate_{ei}_support_recomputed"] = support == event["support"] and error == event["error"]
    for row in measurements:
        qid, arm, result = row["question_id"], row["arm"], row["result"]
        case, common = cases[qid], units[row["execution_units"][0]]
        selected = [units[u] for u in row["execution_units"]]
        calls = [c for u in selected for c in u["calls"]]
        work = sum(len(c["token_ids"]) for c in calls if c["purpose"] != "answer")
        aux = sum(len(c["token_ids"]) for c in calls if c["purpose"] in {"gap_plan", "support_check"})
        answer = [t for c in calls if c["purpose"] == "answer" for t in c["token_ids"]]
        injection = next((e for e in result["events"] if e["kind"] == "injection"), None)
        last_thinking = next((c for c in reversed(calls) if c["purpose"] == "thinking"), None)
        terminal = sum(t in {CLOSE, EOS} for t in last_thinking["token_ids"]) if last_thinking else 0
        checks = {"work_count_recomputed": work == result["work_tokens"] <= cfg["work_tokens"],
            "auxiliary_fully_charged": aux == result["auxiliary_tokens"],
            "answer_count_recomputed": answer == result["answer_ids"] and len(answer) == result["answer_tokens"] <= cfg["answer_tokens"],
            "retained_count": len(result["reason_ids"]) == result["retained_thinking_tokens"],
            "discarded_count_recomputed": work-aux-len(result["reason_ids"])-terminal == result["discarded_tokens"],
            "empty_kept": result["empty_answer"] == (not bool(result["answer_text"].strip())),
            "evidence_cap": result["evidence_tokens"] == (len(injection["payload_ids"]) if injection else 0) <= cfg["evidence_tokens"],
            "all_unit_contracts": all(all(unit_audits[u].values()) for u in row["execution_units"]),
            "main_call_count": sum(c["purpose"] in {"thinking", "answer"} for c in calls) == result["main_calls"],
            "auxiliary_call_count": sum(c["purpose"] in {"gap_plan", "support_check"} for c in calls) == result["aux_calls"]}
        if injection:
            payload, packet = injection["payload_ids"], injection["packet"]
            checks["payload_hash"] = token_hash(payload) == injection["payload_sha256"]
            checks["packet_hash"] = token_hash(packet["ids"]) == packet["sha256"]
            checks["marker_free_packet"] = not {OPEN, CLOSE, EOS}.intersection(payload)
            checks["checkpoint_offset"] = injection["confirmed_offset"] == len(common["result"]["reason_ids"])
            checks["mid_step_boundary"] = row["policy"] not in {"step_sc", "flare_gap", "flare_gap_rebuild"} or injection["confirmed_offset"] >= cfg["minimum_tokens"]
            if row["policy"] in {"step_sc", "flare_gap", "flare_gap_rebuild"}:
                text = common["result"]["thinking_text"]
                checks["complete_confirmed_step"] = any(not text[at:].strip() for at in step_char_boundaries(text))
            if row["content"] == "relevant":
                checks["exact_relevant_packet"] = payload == packet["ids"]
            else:
                body_positions = {i for b in packet["bodies"] for i in range(b["start"], b["stop"])}
                checks["same_length_noise"] = len(payload) == len(packet["ids"])
                checks["labels_wrappers_identical"] = all(a == b for i, (a, b) in enumerate(zip(payload, packet["ids"])) if i not in body_positions)
            if row["policy"] in {"static", "step_sc", "legacy_sc"}:
                checks["same_frozen_question_retrieval"] = packet == case["static_packet"]
            if row["policy"] in {"flare_gap", "flare_gap_rebuild"}:
                support = next(e for e in result["events"] if e["kind"] == "support_check")
                checks["support_gate_validated"] = parse_support(support["raw"], packet)[0] == support["support"] and support["support"]["use"]
        audits.append({"question_id": qid, "arm": arm, **checks})
        score = grade(result["answer_text"], gold[qid]["answer"]) if qid in gold else {}
        cost_keys = set().union(*(u["cost"] for u in selected))
        standalone_cost = {k: sum(u["cost"].get(k, 0) for u in selected) for k in cost_keys}
        offset = injection["confirmed_offset"] if injection else None
        value = {"question_id": qid, "cohort": row["cohort"], "arm": arm, **score,
            **{k: v for k, v in result.items() if k not in {"events", "reason_ids", "answer_ids", "thinking_text"}},
            "injected": bool(injection), "injection_offset": offset,
            "post_injection_thinking": len(result["reason_ids"])-offset if offset is not None else None,
            "answer_thinking_reopens": result["answer_ids"].count(OPEN),
            "standalone_seconds": sum(u["seconds"] for u in selected), "standalone_cost": standalone_cost,
            "native_pflops": sum(u.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) for u in selected)/1e15,
            "shared_no_intervention": row["shared_no_intervention"], "execution_units": row["execution_units"],
            "gold_parent_hit": bool(injection) and qid in gold and bool(set(gold[qid]["document_ids"]) & {b["id"].split("@")[0] for b in injection["packet"]["bodies"]})}
        graded.append(value); by_key[qid, arm] = value
    pair_audits = []
    for qid in cases:
        for policy in ("static", "legacy_sc", "step_sc", "flare_gap", "flare_gap_rebuild"):
            pair = [next((r for r in measurements if r["question_id"] == qid and r["arm"] == policy+"."+v), None) for v in ("relevant", "unrelated")]
            if not all(pair): continue
            a, b = pair
            ea = next((e for e in a["result"]["events"] if e["kind"] == "injection"), None)
            eb = next((e for e in b["result"]["events"] if e["kind"] == "injection"), None)
            check = {"question_id": qid, "policy": policy, "same_common_unit": a["execution_units"][0] == b["execution_units"][0],
                "same_injection_decision": bool(ea) == bool(eb),
                "same_checkpoint": not ea or (ea["confirmed_offset"] == eb["confirmed_offset"] and ea["work_spent"] == eb["work_spent"]),
                "same_length_packet": not ea or len(ea["payload_ids"]) == len(eb["payload_ids"]),
                "same_prefix_length": not ea or len(ea["continuation_prefix_ids"]) == len(eb["continuation_prefix_ids"])}
            pair_audits.append(check)
    cohorts = {}
    for cohort in sorted({r["cohort"] for r in graded}):
        qids = [q for q, c in cases.items() if c.get("cohort", "smoke") == cohort and all((q, arm) in by_key for arm in ARMS)]
        groups = []
        for arm in ARMS:
            chosen = [by_key[q, arm] for q in qids]
            groups.append({"arm": arm, "n": len(chosen), "f1": mean(r["f1"] for r in chosen) if gold else None,
                "exact_match": mean(r["exact_match"] for r in chosen) if gold else None,
                "injected": sum(r["injected"] for r in chosen),
                **{k: sum(bool(r[k]) for r in chosen) for k in ("eos_inside_thinking", "empty_answer", "immediate_thinking_end", "immediate_eos", "forced_thinking_end", "sampled_thinking_end", "shared_no_intervention")},
                **{"mean_"+k: mean(r[k] for r in chosen) for k in ("work_tokens", "auxiliary_tokens", "discarded_tokens", "retained_thinking_tokens", "answer_tokens", "standalone_seconds", "native_pflops", "post_injection_thinking")},
                "answer_cap": sum(r["finish_reason"] == "answer_budget" for r in chosen),
                "retrieval_attempts": sum(r["retrieval_attempts"] for r in chosen)})
        contrasts = []
        if gold:
            for a, b in (("static.relevant", "flare_gap.relevant"), ("legacy_sc.relevant", "flare_gap.relevant"),
                         ("step_sc.relevant", "flare_gap_rebuild.relevant"), ("legacy_sc.relevant", "step_sc.relevant"),
                         ("flare_gap.relevant", "flare_gap_rebuild.relevant"), ("flare_gap.unrelated", "flare_gap_rebuild.unrelated"),
                         *((p+".unrelated", p+".relevant") for p in ("static", "legacy_sc", "step_sc", "flare_gap", "flare_gap_rebuild"))):
                contrasts.append({"a": a, "b": b, "b_minus_a": bootstrap(by_key[q, b]["f1"]-by_key[q, a]["f1"] for q in qids)})
        cohorts[cohort] = {"complete_questions": qids, "groups": groups, "contrasts": contrasts}
    gate_events = []
    for uid, unit in units.items():
        if not uid.endswith(":flare_gap:common"): continue
        for event in unit["result"]["events"]:
            if event["kind"] in {"gap_check", "support_check"}:
                gate_events.append({"unit_id": uid, **event})
    unique_checks = sum(len(x) for x in unit_audits.values())
    failures = [{"scope": "unit", "key": uid, "failed": [k for k, v in checks.items() if not v]} for uid, checks in unit_audits.items() if not all(checks.values())]
    failures += [{"scope": "condition", "key": [r["question_id"], r["arm"]], "failed": [k for k, v in r.items() if type(v) is bool and not v]} for r in audits if any(type(v) is bool and not v for v in r.values())]
    failures += [{"scope": "pair", "key": [r["question_id"], r["policy"]], "failed": [k for k, v in r.items() if type(v) is bool and not v]} for r in pair_audits if any(type(v) is bool and not v for v in r.values())]
    summary = {"conditions": len(measurements), "expected_conditions": metadata["expected_conditions"],
        "completed_marker": any(r["kind"] == "completed" for r in rows), "config": cfg,
        "execution_units": len(units), "actual_unique_seconds": sum(u["seconds"] for u in units.values()),
        "actual_unique_native_pflops": sum(u.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) for u in units.values())/1e15,
        "warmup_native_pflops": sum(r.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) for r in rows if r["kind"] == "warmup")/1e15,
        "failed_executions": [r for r in rows if r["kind"] == "execution_failed"],
        "unit_checks": unique_checks, "condition_audits": len(audits), "pair_audits": len(pair_audits),
        "audit_failures": failures, "cohorts": cohorts,
        "gate_counts": dict(Counter(e["kind"]+":"+(e.get("error") or ("need" if e.get("plan", {}).get("need") else "no_need") if e["kind"] == "gap_check" else e.get("error") or ("use" if e.get("support", {}).get("use") else "reject")) for e in gate_events))}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    for name, data in (("graded.jsonl", graded), ("audit.jsonl", audits), ("pair-audit.jsonl", pair_audits), ("gate-events.jsonl", gate_events)):
        (args.output/name).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in data), encoding="utf-8")
    (args.output/"unit-audit.json").write_text(json.dumps(unit_audits, indent=2)+"\n", encoding="utf-8")
    with (args.output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
        group_rows = [{"cohort": c, **r} for c, val in cohorts.items() for r in val["groups"]]
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0]) if group_rows else ["cohort"])
        writer.writeheader(); writer.writerows(group_rows)
    lines = ["# FLARE-gap step-boundary experiment", "", f"Completed conditions: {len(measurements)}/{metadata['expected_conditions']}. Structural audit failures: {len(failures)}.",
        "", "Work cap 1024 (all previews/checks included), answer cap 512, evidence cap 256. Token F1 is a generative overlap diagnostic, not official TechQA accuracy. One seed per question; cohort sizes are small.", ""]
    for cohort, val in cohorts.items():
        lines += ["## "+cohort, "", "| Arm | n | F1 % | Injected | EOS in thinking | Empty | Immediate close | Work tokens | Aux tokens | Seconds |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for g in val["groups"]:
            f1 = f"{100*g['f1']:.2f}" if g["f1"] is not None else "—"
            fmt = lambda v: f"{v:.1f}" if v is not None else "—"
            lines.append(f"| {g['arm']} | {g['n']} | {f1} | {g['injected']} | {g['eos_inside_thinking']} | {g['empty_answer']} | {g['immediate_thinking_end']} | {fmt(g['mean_work_tokens'])} | {fmt(g['mean_auxiliary_tokens'])} | {fmt(g['mean_standalone_seconds'])} |")
        lines.append("")
        for c in val["contrasts"]:
            b = c["b_minus_a"]
            if b: lines.append(f"- {c['b']} − {c['a']}: {100*b['difference']:+.2f} pp F1; paired question bootstrap 95% [{100*b['bootstrap_95'][0]:+.2f}, {100*b['bootstrap_95'][1]:+.2f}].")
        lines.append("")
    lines += ["## Accounting and limits", "", f"Unique executed units: {len(units)}; measured unit time {summary['actual_unique_seconds']:.1f}s, native padded dense FLOP estimate {summary['actual_unique_native_pflops']:.3f} PFLOP. Warmup is separate. Model load, smoke, failed prep and container lifetime appear in provenance receipts.",
        "", "Each condition's cost includes its full common stem plus cold branch prefill. Study totals deduplicate shared execution. A no-intervention real/noise pair reuses one result and is not an independent replicate. Unrelated bodies are substituted after the real-evidence support gate. Primary FLARE-gap appends after a complete assistant step. Its rebuild companion shares the same common stem/payload and moves references to the user region. Step SC also reconstructs the user region; legacy SC retains assistant-inline placement. Query, timing and gating still differ between trigger policies.",
        "", "Source labels and wrappers stay identical in paired content controls. EOS, empty answers, capped outputs, parse failures and unsupported retrievals remain in the denominator. Bootstrap intervals do not cover sampling-seed variability or generalization beyond this small sample.", ""]
    (args.output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("conditions", "expected_conditions", "execution_units", "gate_counts", "audit_failures")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation", "prepared", "gold", "output"):
        parser.add_argument("--"+name, type=Path, required=name != "gold")
    summarize(parser.parse_args())

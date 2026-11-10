"""Offline audits and question-clustered evaluation of grounded retrieval/IS."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from summarize_flare_gap import bootstrap, grade
from inference_scaling.retrieval.grounded import parse_evidence, parse_verdict, verdict_reward
from inference_scaling.retrieval.scaled import SearchLedger
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, select_metropolized, weigh

EOS, CLOSE, OPEN = 248046, 248069, 248068


def average(values):
    return statistics.fmean(values) if values else 0.


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Decoder:
    def __init__(self, vocabulary):
        assert vocabulary["decoder"]["type"] == "ByteLevel"
        alphabet = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
        codes, extra = alphabet.copy(), 0
        for byte in range(256):
            if byte not in alphabet:
                alphabet.append(byte); codes.append(256+extra); extra += 1
        self.mapping = dict(zip(map(chr, codes), alphabet, strict=True))
        self.special = set(vocabulary["special_ids"])
        self.vocab = vocabulary["vocab"]

    def __call__(self, ids, skip=False):
        encoded = "".join(self.vocab[str(t)] for t in ids if not skip or t not in self.special)
        return bytes(self.mapping[c] for c in encoded).decode("utf-8", errors="replace")


def audit_search(unit, seed, decode):
    d = unit["detail"]
    if not isinstance(d, dict) or "root_ids" not in d:
        return {}
    checks = {}
    calls = unit["calls"]
    search = [c for c in calls if c["purpose"] == "search"]
    backend = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=EOS), model_id="offline")
    ledger = SearchLedger(backend, d["root_ids"], (CLOSE,), 8192, seed, 8192)
    ledger.calls = search
    verified = d.get("answer_prepaid_in_work", False)
    paid = [c for c in calls if c["purpose"] != "answer"]
    checks["root_hash"] = token_hash(ledger.root) == d["root_sha256"]
    checks["search_work"] = sum(len(c["token_ids"]) for c in paid) == d["search_work"]
    checks["immutable_root"] = all(tuple(c["prefix_ids"][:len(ledger.root)]) == ledger.root for c in search)
    selected = tuple(d["selected_suffix_ids"])
    injections = [e for e in unit["result"]["events"] if e["kind"] == "injection"]
    bodies = []
    if injections:
        inj = injections[-1]
        bodies = [decode(inj["payload_ids"][b["start"]:b["stop"]]) for b in inj["packet"]["bodies"]]
    verdicts = {v["suffix_sha256"]: v for v in d.get("verdicts", [])}
    answer_calls = {c["request_id"].split(":", 1)[1]: c for c in calls if c["purpose"] == "leaf_answer"}
    verifier_calls = {c["request_id"].split(":", 1)[1]: c for c in calls if c["purpose"] == "leaf_verify"}
    for key, verdict in verdicts.items():
        parsed, error = parse_verdict(verdict["raw"], bodies)
        checks["verdict_parse_"+key] = parsed == verdict["parsed"] and error == verdict["error"]
        checks["verdict_raw_"+key] = decode(verifier_calls[key]["token_ids"], True) == verdict["raw"]
        score = verdict_reward(parsed, verdict["sc"])
        if not decode(answer_calls[key]["token_ids"], True).strip():
            score = -3.
        checks["verdict_score_"+key] = math.isclose(score, verdict["reward"], abs_tol=1e-12)
    kept, paid_keys = (), set()
    for si, record in enumerate(d["steps"]):
        fixed = kept[:record["generated_length_before"]]
        config = d["config"]
        remaining = config["total_length"]-len(fixed)
        width = min(config["block_size"], remaining)
        m, k = config["candidate_count"], config["rollout_count"]
        carried = int(bool(kept))
        bound = (m-carried)*(width+k*(remaining-width))+carried*(k-1)*(remaining-width)
        if verified:
            bound += ((m-carried)*k+carried*(k-1))*(512+96)
        checks[f"pool{si}_reservation"] = bound == record["reserved_work"]
        blocks, leaves, keys = [], [], set()
        score_checks = True
        for candidate in record["candidates"]:
            block, group = tuple(candidate["token_ids"]), []
            for leaf in candidate["rollouts"]:
                tail = tuple(leaf["token_ids"])
                suffix = fixed+block+tail
                sc, = ledger.reward(ledger.root, [suffix])
                key = token_hash(suffix)
                keys.add(key)
                value = sc
                if verified:
                    if EOS in suffix:
                        value = -3.
                    else:
                        value = verdicts[key]["reward"]
                        score_checks &= math.isclose(verdicts[key]["sc"], sc, abs_tol=1e-12)
                        expected_prefix = list(ledger.root+suffix+(() if suffix[-1:] == (CLOSE,) else (CLOSE,)))
                        checks["candidate_answer_prefix_"+key] = answer_calls[key]["prefix_ids"] == expected_prefix
                score_checks &= math.isclose(value, leaf["reward"], abs_tol=1e-12)
                score_checks &= math.isclose(value/config["reward_temperature"], leaf["log_weight"], abs_tol=1e-12)
                group.append(RolloutEvaluation(tail, leaf["reward"], leaf["log_weight"]))
            blocks.append(block); leaves.append(group)
        candidates = weigh(blocks, leaves)
        mi, ki = select_metropolized(candidates, SeedStream(seed), "conditional_is", si, group_blocks=True)
        checks[f"pool{si}_scores"] = score_checks
        checks[f"pool{si}_weights"] = all(math.isclose(c.log_weight, r["log_weight"], abs_tol=1e-12)
            for c, r in zip(candidates, record["candidates"], strict=True))
        checks[f"pool{si}_selection"] = mi == record["selected_index"] and ki == record["completion_index"]
        work = sum(len(c["token_ids"]) for c in search if c["request_id"].startswith(f"conditional-is:step:{si}:"))
        for key in keys-paid_keys:
            work += sum(len(c[key]["token_ids"]) for c in [answer_calls, verifier_calls] if key in c)
        paid_keys.update(keys)
        checks[f"pool{si}_actual_work"] = work == record["actual_work"] <= bound
        kept = fixed+candidates[mi].token_ids+candidates[mi].rollouts[ki].token_ids
    checks["selected_sequence"] = selected == kept
    body_end = next((i for i,t in enumerate(selected) if t in {EOS,CLOSE}),len(selected))
    if body_end:
        checks["retained_suffix"] = unit["result"]["reason_ids"][-body_end:] == list(selected[:body_end])
    if verified:
        key = token_hash(selected)
        checks["no_answer_regeneration"] = not any(c["purpose"] == "answer" for c in calls)
        expected = answer_calls[key]["token_ids"] if key in answer_calls else []
        checks["returned_prepaid_answer"] = unit["result"]["answer_ids"] == expected
        if EOS in selected:
            checks["eos_not_reopened"] = key not in answer_calls and unit["result"]["eos_inside_thinking"]
    else:
        answers = [c for c in calls if c["purpose"] == "answer"]
        prefix = list(ledger.root)+list(selected[:body_end])+[CLOSE]
        for i,c in enumerate(answers):
            checks[f"answer{i}_prefix"] = c["prefix_ids"] == prefix
            prefix += c["token_ids"]
        checks["returned_answer"] = unit["result"]["answer_ids"] == [t for c in answers for t in c["token_ids"]]
        if EOS in selected:
            checks["eos_not_reopened"] = not answers and unit["result"]["eos_inside_thinking"]
    if d["stop"] == "whole_pool_budget":
        remaining = d["config"]["total_length"]-d["fixed_tokens"]
        width = min(d["config"]["block_size"], remaining)
        m,k = d["config"]["candidate_count"],d["config"]["rollout_count"]
        bound = (m-1)*(width+k*(remaining-width))+(k-1)*(remaining-width)
        if verified:
            bound += ((m-1)*k+k-1)*608
        checks["budget_stop"] = bound > 8192-d["common_work"]-d["search_work"]
    else:
        checks["terminal_stop"] = d["fixed_tokens"] >= len(selected)
    return checks


def summarize(snapshot, output, stage, gold_path=None):
    gold = {x["id"]: x for x in read_jsonl(gold_path)} if gold_path else {}
    audits, all_rows, all_units, metadata = {}, [], {}, []
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir() or not (folder/(stage+".jsonl")).exists():
            continue
        rows = read_jsonl(folder/(stage+".jsonl"))
        meta = next(x for x in rows if x["kind"] == "metadata")
        metadata.append(meta)
        ms = [x for x in rows if x["kind"] == "measurement"]
        done = [x for x in rows if x["kind"] == "completed"]
        audits[folder.name+":completion"] = {"exit_zero": (folder/(stage+".exit")).read_text().strip() == "0",
            "queue_complete": len(done) == 1 and done[0]["conditions"] == len(ms) == meta["expected_conditions"],
            "no_failed_unit": not any(x["kind"] == "failed_unit" for x in rows),
            "exact_grid": {(x["question_id"],x["seed"],x["arm"]) for x in ms} ==
                {(q,s,arm) for q in meta["cases"] for s in meta["seeds"] for arm in meta["arms"]}}
        vocabulary = json.loads((folder/"trace-tokenizer-subset.json").read_text(encoding="utf-8"))
        prepared = json.loads((folder/("prepared-"+stage+".json")).read_text(encoding="utf-8"))
        frozen = json.loads((folder/"protocol/SOURCE-FROZEN.json").read_text(encoding="utf-8"))
        audits[folder.name+":frozen_inputs"] = {
            "prepared": hashlib.sha256((folder/("prepared-"+stage+".json")).read_bytes()).hexdigest() == meta["prepared_sha256"],
            "settings": frozen["hashes"]["settings.json"] == meta["settings_sha256"],
            "source": all(frozen["source_files"].get("source/"+n) == h for n,h in meta["source_hashes"].items()),
            "worker": folder.name == meta["worker"],
        }
        if stage == "formal":
            plan = json.loads((folder/"protocol/FORMAL-WORKER-PLAN.json").read_text(encoding="utf-8"))
            audits[folder.name+":frozen_inputs"]["formal_assignment"] = meta["cases"] == plan["assignments"][folder.name] and meta["seeds"] == plan["seeds"]
        decode = Decoder(vocabulary)
        audits[folder.name+":tokenizer"] = {"same": vocabulary["tokenizer_sha256"] == prepared["tokenizer_hashes"]["tokenizer.json"]}
        for unit in [x for x in rows if x["kind"] == "execution_unit"]:
            uid = unit["unit_id"]
            full_id = folder.name+":"+uid
            all_units[full_id] = unit
            result, checks = unit["result"], {}
            for ci, call in enumerate(unit["calls"]):
                ids = call["token_ids"]
                checks[f"call{ci}_hash"] = token_hash(call["prefix_ids"]) == call["prefix_sha256"]
                checks[f"call{ci}_length"] = 0 <= len(ids) <= call["max_new_tokens"]
                checks[f"call{ci}_context"] = len(call["prefix_ids"])+call["max_new_tokens"] <= 8192
                checks[f"call{ci}_logprobs"] = len(ids) == len(call["token_logprobs"])
                checks[f"call{ci}_eos"] = EOS not in ids or ids[-1] == EOS
                if call["purpose"] in {"search", "thinking"}:
                    checks[f"call{ci}_stop"] = CLOSE not in ids or ids[-1] == CLOSE
                if call["purpose"] in {"search", "thinking", "answer"}:
                    checks[f"call{ci}_statistics"] = len(call["token_statistics"]) == len(ids) and all(x is not None and math.isfinite(x) for x in call["token_statistics"])
                if call["purpose"] in {"gap_plan", "support_check", "leaf_verify"}:
                    checks[f"call{ci}_no_hidden_thinking"] = OPEN not in ids and CLOSE not in ids
            checks["thinking_decode"] = decode(result["reason_ids"]) == result["thinking_text"]
            checks["answer_decode"] = decode(result["answer_ids"], True).strip() == result["answer_text"]
            checks["native_graphs"] = not unit["calls"] or unit.get("graph_execution", {}).get("estimated_dense_padded_flops", 0) > 0
            checks["no_scoring_forwards"] = unit["cost"].get("score_forward_token_slots", 0) == 0
            checks["no_preemptions"] = unit["cost"].get("preemptions", 0) == 0
            checks["no_cancellations"] = unit["cost"].get("cancelled_requests", 0) == 0
            qid,run_seed = uid.split(":")[:2]
            seed = SeedStream(int(run_seed)).derive("question",qid)
            checks.update(audit_search(unit,seed,decode))
            if uid.endswith(":probe"):
                for ei,event in enumerate(result["events"]):
                    if event["kind"] == "grounded_evidence" and "raw" in event:
                        parsed,error = parse_evidence(event["raw"], event["candidates"]["sources"])
                        checks[f"gate{ei}_parse"] = parsed == event["support"]
                        checks[f"gate{ei}_error"] = error == event["error"] or event["error"] == "excerpt_packet_cap_or_marker"
                        if event.get("packet"):
                            packet=event["packet"]
                            checks[f"gate{ei}_packet_cap"] = len(packet["ids"]) <= 256
                            checks[f"gate{ei}_body"] = all(decode(packet["ids"][b["start"]:b["stop"]]) == b["text"] for b in packet["bodies"])
            audits[full_id]=checks
        for row in ms:
            row = {**row,"full_execution_units":[folder.name+":"+u for u in row["execution_units"]]}
            all_rows.append(row)
    graded=[]
    for row in all_rows:
        used=[all_units[u] for u in row["full_execution_units"]]
        calls=[c for u in used for c in u["calls"]]
        r=row["result"]
        work=sum(len(c["token_ids"]) for c in calls if c["purpose"] != "answer")
        extra_answer=sum(len(c["token_ids"]) for c in calls if c["purpose"] == "answer")
        aux=sum(len(c["token_ids"]) for c in calls if c["purpose"] in {"gap_plan","support_check","leaf_verify"})
        key=f"condition:{row['worker']}:{row['question_id']}:{row['seed']}:{row['arm']}"
        audits[key]={"work":work == r["work_tokens"] <= 8192,"auxiliary":aux == r["auxiliary_tokens"],
            "returned_answer_cap":len(r["answer_ids"]) == r["answer_tokens"] <= 512,
            "answer_accounting":row["search"].get("answer_prepaid_in_work",False) or extra_answer == r["answer_tokens"],
            "unique_execution_units":len(row["execution_units"]) == len(set(row["execution_units"]))}
        score=grade(r["answer_text"],gold[row["question_id"]]["answer"]) if row["question_id"] in gold else {"f1":None,"exact_match":None}
        gates=[e for u in used if u["unit_id"].endswith(":probe") for e in u["result"]["events"]]
        verdicts=row["search"].get("verdicts",[])
        generated_by_purpose = Counter()
        for call in calls:
            generated_by_purpose[call["purpose"]] += len(call["token_ids"])
        graded.append({k:row[k] for k in ["question_id","cohort","seed","worker","arm","context","method","content"]}|score|
            {"work_tokens":work,"total_generated_tokens":work+extra_answer,"returned_answer_tokens":r["answer_tokens"],
             "auxiliary_tokens":aux,"seconds":sum(u["seconds"] for u in used),
             "probe_generated_tokens":sum(len(c["token_ids"]) for u in used if u["unit_id"].endswith(":probe") for c in u["calls"]),
             "search_thinking_tokens":generated_by_purpose["search"],
             "candidate_answer_tokens":generated_by_purpose["leaf_answer"],
             "verifier_tokens":generated_by_purpose["leaf_verify"],
             "retrieval_auxiliary_tokens":generated_by_purpose["gap_plan"]+generated_by_purpose["support_check"],
             "retained_thinking_tokens":r["retained_thinking_tokens"],
             "discarded_confirmed_tokens":sum(len(e.get("discarded_confirmed_ids", [])) for e in gates),
             "native_pflops":sum(u.get("graph_execution",{}).get("estimated_dense_padded_flops",0) for u in used)/1e15,
             "gate_accepted":r["evidence_tokens"]>0,"prefix_reset":any("discarded_confirmed_ids" in e for e in gates),
             "invalid_extractor":sum(e.get("error") is not None for e in gates if e["kind"] == "grounded_evidence"),
             "invalid_verifiers":sum(v["error"] is not None for v in verdicts),"verifiers":len(verdicts),
             "search_steps":len(row["search"].get("steps",[])),"unused_budget":8192-work,
             "answer_truncated":r["finish_reason"] == "answer_budget",
             "raw_thinking_eos":sum(EOS in c["token_ids"] for c in calls if c["purpose"] in {"search","thinking"}),
             "raw_first_token_close":sum(bool(c["token_ids"]) and c["token_ids"][0] == CLOSE for c in calls if c["purpose"] == "search"),
             **{k:r[k] for k in ["empty_answer","eos_inside_thinking","immediate_thinking_end","immediate_eos","forced_thinking_end","evidence_tokens"]}})
    pairs=defaultdict(dict)
    for row in all_rows:
        if row["method"] == "verified_is":
            pairs[(row["worker"],row["question_id"],row["seed"],row["context"])][row["content"]]=row
    for key,pair in pairs.items():
        if set(pair) != {"relevant","unrelated"}:continue
        real,noise=pair["relevant"],pair["unrelated"]
        checks={"length":real["result"]["evidence_tokens"] == noise["result"]["evidence_tokens"]}
        if not real["result"]["evidence_tokens"]:
            checks["exact_no_gate_reuse"] = real["result"] == noise["result"] and real["execution_units"] == noise["execution_units"]
        else:
            checks["root_length"] = len(real["search"]["root_ids"]) == len(noise["search"]["root_ids"])
            ri=next(e for e in real["result"]["events"] if e["kind"] == "injection")
            ni=next(e for e in noise["result"]["events"] if e["kind"] == "injection")
            body={i for b in ri["packet"]["bodies"] for i in range(b["start"],b["stop"])}
            checks["wrapper"] = all(x == y for i,(x,y) in enumerate(zip(ri["payload_ids"],ni["payload_ids"],strict=True)) if i not in body)
            checks["changed_body"] = ri["payload_ids"] != ni["payload_ids"]
            checks["same_boundary"] = ri["confirmed_offset"] == ni["confirmed_offset"]
            checks["same_probe"] = real["execution_units"][:-1] == noise["execution_units"][:-1]
        audits["pair:"+str(key)] = checks
    groups=defaultdict(list)
    for row in graded:groups[(row["cohort"],row["arm"])].append(row)
    summary=[]
    for (cohort,arm),values in sorted(groups.items()):
        summary.append({"cohort":cohort,"arm":arm,"conditions":len(values),"questions":len({r['question_id'] for r in values}),
            **{k:average([r[k] for r in values if r[k] is not None]) for k in ["f1","exact_match","work_tokens","total_generated_tokens","returned_answer_tokens","auxiliary_tokens","seconds","native_pflops","gate_accepted","prefix_reset","search_steps","unused_budget","probe_generated_tokens","search_thinking_tokens","candidate_answer_tokens","verifier_tokens","retrieval_auxiliary_tokens","retained_thinking_tokens","discarded_confirmed_tokens"]},
            **{k:sum(r[k] for r in values) for k in ["empty_answer","eos_inside_thinking","immediate_thinking_end","immediate_eos","forced_thinking_end","answer_truncated","raw_thinking_eos","raw_first_token_close","invalid_extractor","invalid_verifiers","verifiers"]}})
    contrasts=[]
    comparisons=[("dynamic_verified_is","dynamic_legacy_sc"),("dynamic_verified_is","dynamic_anchor_sc"),
        ("dynamic_verified_is","static_verified_is"),("dynamic_verified_is","static_legacy_sc"),
        ("dynamic_anchor_sc","dynamic_legacy_sc"),("static_verified_is","static_anchor_sc"),
        ("dynamic_verified_is","dynamic_verified_noise"),("static_verified_is","static_verified_noise")]
    for cohort in sorted({r["cohort"] for r in graded}):
        for left,right in comparisons:
            values=defaultdict(lambda:defaultdict(list))
            for row in graded:
                if row["cohort"] == cohort and row["arm"] in {left,right} and row["f1"] is not None:
                    values[row["question_id"]][row["arm"]].append(row["f1"])
            diffs=[average(x[left])-average(x[right]) for x in values.values() if set(x) == {left,right}]
            if diffs:contrasts.append({"cohort":cohort,"left":left,"right":right,**bootstrap(diffs)})
    failed={key:[k for k,v in checks.items() if not v] for key,checks in audits.items() if not all(checks.values())}
    audit={"passed":not failed,"checks":sum(len(x) for x in audits.values()),"failed":failed,"details":audits}
    unique_generated = Counter()
    for unit in all_units.values():
        for call in unit["calls"]:
            unique_generated[call["purpose"]] += len(call["token_ids"])
    unique_cost={"units":len(all_units),"conditions":len(graded),
        "generated_tokens":sum(len(c["token_ids"]) for u in all_units.values() for c in u["calls"]),
        "generated_tokens_by_purpose":dict(sorted(unique_generated.items())),
        "unit_seconds":sum(u["seconds"] for u in all_units.values()),
        "native_pflops":sum(u.get("graph_execution",{}).get("estimated_dense_padded_flops",0) for u in all_units.values())/1e15}
    output.mkdir(parents=True,exist_ok=True)
    (output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    result={"stage":stage,"audit_passed":not failed,"audit_checks":audit["checks"],"summary":summary,"contrasts":contrasts,"unique_execution":unique_cost,"metadata":metadata}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (output/"graded.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in graded),encoding="utf-8")
    if summary:
        with (output/"summary.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    text=["# Grounded retrieval / IS: complete results", "", "F1 is lexical overlap, not factual accuracy. Question means average seeds; regression and fresh cohorts remain separate.", "",
        "| Cohort | Arm | Questions | F1 % | Work | Total generated | Seconds | PFLOPs | Empty | Forced close |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        text.append(f"| {r['cohort']} | {r['arm']} | {r['questions']} | {100*r['f1']:.3f} | {r['work_tokens']:.1f} | {r['total_generated_tokens']:.1f} | {r['seconds']:.2f} | {r['native_pflops']:.3f} | {r['empty_answer']} | {r['forced_thinking_end']} |")
    text += ["", "## Realized token allocation", "", "Per-condition means. Probe tokens include main drafts and retrieval auxiliaries, so this column overlaps retrieval auxiliaries; do not sum those two columns. Candidate answers and verifiers are paid inside verified-IS work.", "",
        "| Cohort | Arm | Probe total | Search thinking | Candidate answers | Verifiers | Retrieval auxiliaries | Discarded confirmed prefix |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        text.append(f"| {r['cohort']} | {r['arm']} | {r['probe_generated_tokens']:.1f} | {r['search_thinking_tokens']:.1f} | {r['candidate_answer_tokens']:.1f} | {r['verifier_tokens']:.1f} | {r['retrieval_auxiliary_tokens']:.1f} | {r['discarded_confirmed_tokens']:.1f} |")
    text += ["", "## Paired question bootstrap", "", "Differences and 95% intervals are F1 percentage points, descriptive and unadjusted.", ""]
    for r in contrasts:
        low,high=r["bootstrap_95"]
        text.append(f"- {r['cohort']}: {r['left']} − {r['right']}: {100*r['difference']:+.3f} [{100*low:+.3f}, {100*high:+.3f}], n={r['n_questions']}.")
    text += ["", f"Audits passed: {not failed}; checks: {audit['checks']}; unique execution units: {len(all_units)}."]
    (output/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps({"audit_passed":not failed,"checks":audit["checks"],"failures":failed,**unique_cost}))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=["smoke","formal"],required=True)
    p.add_argument("--gold",type=Path)
    a=p.parse_args()
    result=summarize(a.snapshot,a.output,a.stage,a.gold)
    raise SystemExit(0 if result["audit_passed"] else 1)

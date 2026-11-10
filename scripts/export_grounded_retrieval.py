"""Export all external-model candidates and references, with token decode checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_grounded_retrieval import Decoder, read_jsonl


def export(snapshot, output, stage):
    output.mkdir(parents=True, exist_ok=True)
    reports, count, unit_count = [], 0, 0
    reviews = []
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir() or not (folder/(stage+".jsonl")).exists():
            continue
        rows = read_jsonl(folder/(stage+".jsonl"))
        if not any(x["kind"] == "completed" for x in rows):
            raise ValueError("do not export an incomplete queue as a completed result")
        prepared=json.loads((folder/("prepared-"+stage+".json")).read_text(encoding="utf-8"))
        decode=Decoder(json.loads((folder/"trace-tokenizer-subset.json").read_text(encoding="utf-8")))
        cases={x["id"]:x for x in prepared["cases"]}
        units={x["unit_id"]:x for x in rows if x["kind"] == "execution_unit"}
        for uid,u in units.items():
            assert decode(u["result"]["reason_ids"]) == u["result"]["thinking_text"],uid
            assert decode(u["result"]["answer_ids"],True).strip() == u["result"]["answer_text"],uid
            unit_count += 1
        grouped={}
        for row in rows:
            if row["kind"] == "measurement":
                grouped.setdefault((row["question_id"],row["seed"]),[]).append(row)
        for (qid,seed),measurements in grouped.items():
            ids=dict.fromkeys(uid for row in measurements for uid in row["execution_units"])
            text=[f"{qid}, seed {seed}, worker {folder.name}", "",
                "Saved Qwen3.8-27B experimental model outputs. These can contain factual errors.",
                ">>> marks exporter annotations. Every candidate is included, not just selected examples.",
                "These are external model outputs, not the assistant's private reasoning.", "",
                ">>> Original prompt", decode(cases[qid]["prompt_ids"])]
            for uid in ids:
                u=units[uid];d=u["detail"] or {}
                text += ["", ">>> Execution unit "+uid]
                if "root_ids" in d:
                    text += [">>> Immutable search context",decode(d["root_ids"])]
                    for i,pool in enumerate(d.get("steps",[])):
                        text += [f">>> Pool {i}: selected block={pool['selected_index']}, rollout={pool['completion_index']}, actual/reserved work={pool['actual_work']}/{pool['reserved_work']}"]
                    for v in d.get("verdicts",[]):
                        text += [">>> Candidate verdict "+json.dumps(v,ensure_ascii=False)]
                for call in u["calls"]:
                    if call["purpose"] in {"gap_plan","support_check","leaf_verify"}:
                        text += [">>> Auxiliary prompt",decode(call["prefix_ids"])]
                    text += [f">>> {call['purpose']} request={call.get('request_id','')} seed={call['seed']} finish={call['finish_reason']}",decode(call["token_ids"])]
                for event in u["result"]["events"]:
                    if event["kind"] == "injection":
                        text += [f">>> Input evidence at reasoning boundary {event['confirmed_offset']}",decode(event["payload_ids"])]
                    elif uid.endswith(":probe"):
                        if event["kind"] == "grounded_evidence":
                            reviews.append({"worker":folder.name,"question_id":qid,"seed":seed,"unit_id":uid,
                                "question":cases[qid]["question"],"event":event})
                            text += [">>> Evidence selection / prefix repair",json.dumps(event,ensure_ascii=False)]
                        elif event["kind"] in {"rollback","search_checkpoint"}:
                            text += [">>> Discarded provisional draft",decode(event["discarded_ids"])]
            for row in measurements:
                r=row["result"]
                text += ["",">>> Final condition "+row["arm"],
                    f">>> work={r['work_tokens']}; returned answer={r['answer_tokens']}; evidence={r['evidence_tokens']}; empty={r['empty_answer']}; EOS-in-thinking={r['eos_inside_thinking']}; forced-close={r['forced_thinking_end']}",
                    ">>> Retained thinking",r["thinking_text"],">>> Returned answer",r["answer_text"]]
            data=("\n".join(text)+"\n").encode()
            name=f"{qid}-seed-{seed}-{folder.name}-all-conditions.txt"
            (output/name).write_bytes(data)
            reports.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
            count += len(measurements)
    (output/"manifest.json").write_text(json.dumps({"selection":"all questions/seeds/arms and unselected candidates",
        "condition_examples":count,"decoded_units":unit_count,"files":reports},indent=2)+"\n",encoding="utf-8")
    (output/"gate-events.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in reviews),encoding="utf-8")
    print(json.dumps({"conditions":count,"units":unit_count,"files":len(reports),"gate_events":len(reviews)}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=["smoke","formal"],required=True)
    a=p.parse_args();export(a.snapshot,a.output,a.stage)

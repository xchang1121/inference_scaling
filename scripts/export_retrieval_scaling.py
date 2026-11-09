"""Verify token decoding and export every pilot case, including unselected samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepared = json.loads((a.snapshot/"prepared-formal-v2.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (a.snapshot/"formal.jsonl").read_text(encoding="utf-8").splitlines()]
    if not any(r["kind"] == "completed" for r in rows):
        raise ValueError("the formal queue is incomplete")
    vocabulary = json.loads((a.snapshot/"trace-tokenizer-subset.json").read_text(encoding="utf-8"))
    assert vocabulary["tokenizer_sha256"] == prepared["tokenizer_hashes"]["tokenizer.json"]
    assert vocabulary["decoder"]["type"] == "ByteLevel"
    alphabet = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
    codes, extra = alphabet.copy(), 0
    for byte in range(256):
        if byte not in alphabet:
            alphabet.append(byte); codes.append(256+extra); extra += 1
    mapping = dict(zip(map(chr, codes), alphabet, strict=True))
    special = set(vocabulary["special_ids"])

    def decode(ids, skip=False):
        encoded = "".join(vocabulary["vocab"][str(t)] for t in ids if not skip or t not in special)
        return bytes(mapping[c] for c in encoded).decode("utf-8", errors="replace")

    units = {r["unit_id"]: r for r in rows if r["kind"] == "execution_unit"}
    for uid, unit in units.items():
        assert decode(unit["result"]["reason_ids"]) == unit["result"]["thinking_text"], uid
        assert decode(unit["result"]["answer_ids"], True).strip() == unit["result"]["answer_text"], uid
    a.output.mkdir(parents=True, exist_ok=True)
    files = []
    count = 0
    for case in prepared["cases"]:
        measurements = [r for r in rows if r["kind"] == "measurement" and r["question_id"] == case["id"]]
        used = dict.fromkeys(uid for m in measurements for uid in m["execution_units"])
        text = [case["id"]+" / all 15 conditions", "",
            "Saved Qwen3.8-27B model outputs; they may contain factual errors.",
            "All cases are exported. >>> denotes exporter annotations, not model text.",
            "Shared execution units are shown once, with all generated candidates, including those not selected.",
            "Original token IDs/statistics and every exact prefix are in snapshot/formal.jsonl.",
            "", ">>> Original complete prompt", decode(case["prompt_ids"])]
        for uid in used:
            unit = units[uid]
            detail = unit["detail"] or {}
            text += ["", ">>> Execution unit "+uid]
            if "root_ids" in detail:
                text += [">>> Frozen search context", decode(detail["root_ids"]),
                    f">>> Method={detail['method']}; search work={detail['search_work']}; common work={detail['common_work']}"]
                if detail["method"] == "sc_bon":
                    text += [f">>> Scores={detail['scores']}; selected index={detail['selected_index']}"]
                for si, step in enumerate(detail.get("steps", [])):
                    text += [f">>> Pool {si}: selected candidate={step['selected_index']}, rollout={step['completion_index']}; reserved={step['reserved_work']}, actual={step['actual_work']}"]
            for i, call in enumerate(unit["calls"]):
                if call["purpose"] in {"gap_plan", "support_check"}:
                    text += [">>> Complete independent auxiliary prompt", decode(call["prefix_ids"])]
                text += [f">>> Call {i}: {call['purpose']}; request={call.get('request_id', '')}; seed={call['seed']}; prefix_sha256={call['prefix_sha256']}; finish={call['finish_reason']}",
                    decode(call["token_ids"])]
            if "selected_suffix_ids" in detail:
                text += [">>> Selected complete suffix", decode(detail["selected_suffix_ids"])]
            for event in unit["result"]["events"]:
                if event["kind"] == "injection":
                    text += [f">>> Injected INPUT at {event['position']}, confirmed offset={event['confirmed_offset']}; not generated work", decode(event["payload_ids"])]
                elif uid.endswith(":probe") and event["kind"] in {"rollback", "search_checkpoint"}:
                    text += [">>> Discarded provisional draft", decode(event["discarded_ids"])]
        for measurement in measurements:
            result = measurement["result"]
            text += ["", ">>> Condition "+measurement["arm"],
                ">>> Execution units: "+", ".join(measurement["execution_units"]),
                f">>> work={result['work_tokens']}, answer={result['answer_tokens']}, evidence={result['evidence_tokens']}, empty={result['empty_answer']}, thinking_EOS={result['eos_inside_thinking']}, forced_close={result['forced_thinking_end']}",
                ">>> Retained thinking", result["thinking_text"], ">>> Final answer", result["answer_text"]]
        data = ("\n".join(text)+"\n").encode()
        name = case["id"]+"-all-15-conditions.txt"
        (a.output/name).write_bytes(data)
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        count += len(measurements)
    manifest = {"selection": "all eight frozen cases and every condition; no quality-based selection",
        "condition_examples": count, "all_execution_unit_decodes_verified": len(units),
        "tokenizer_sha256": vocabulary["tokenizer_sha256"], "files": files}
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"condition_examples": count, "units_decoded": len(units), "files": len(files)}))


if __name__ == "__main__":
    main()

"""Verify saved token decoding and export preselected complete model traces."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = json.loads((args.snapshot/"prepared.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.snapshot/"formal.jsonl").read_text(encoding="utf-8").splitlines()]
    if not any(r["kind"] == "completed" for r in rows): raise ValueError("formal queue is incomplete")
    vocab = json.loads((args.snapshot/"trace-tokenizer-subset.json").read_text(encoding="utf-8"))
    assert vocab["tokenizer_sha256"] == prepared["tokenizer_hashes"]["tokenizer.json"]
    assert vocab["decoder"]["type"] == "ByteLevel"
    alphabet = list(range(33, 127))+list(range(161, 173))+list(range(174, 256)); codes = alphabet.copy(); extra = 0
    for b in range(256):
        if b not in alphabet: alphabet.append(b); codes.append(256+extra); extra += 1
    mapping = dict(zip(map(chr, codes), alphabet, strict=True)); special = set(vocab["special_ids"])
    def decode(ids, skip=False):
        text = "".join(vocab["vocab"][str(t)] for t in ids if not skip or t not in special)
        return bytes(mapping[c] for c in text).decode("utf-8", errors="replace")
    units = {r["unit_id"]: r for r in rows if r["kind"] == "execution_unit"}
    for uid, unit in units.items():
        assert decode(unit["result"]["reason_ids"]) == unit["result"]["thinking_text"], uid
        assert decode(unit["result"]["answer_ids"], True).strip() == unit["result"]["answer_text"], uid
    selected = [c for c in prepared["cases"] if c["cohort"] == "heldout"][:2]
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for case in selected:
        measurements = [r for r in rows if r["kind"] == "measurement" and r["question_id"] == case["id"]]
        blocks = []
        for row in measurements:
            result = row["result"]
            text = [f"{case['id']} / {row['arm']}",
                "Saved Qwen3.8-27B output, including every discarded draft and auxiliary output. Original model content may be wrong.",
                "Lines beginning >>> are exporter annotations. Shared execution units are identified; these are not independent stem reruns.",
                f"work={result['work_tokens']}; aux={result['auxiliary_tokens']}; discarded={result['discarded_tokens']}; answer={result['answer_tokens']}; evidence={result['evidence_tokens']}",
                f"finish={result['finish_reason']}; forced_close={result['forced_thinking_end']}; empty={result['empty_answer']}",
                ">>> Original complete base prompt", decode(case["prompt_ids"]), ""]
            for uid in row["execution_units"]:
                unit = units[uid]
                text += [f">>> Execution unit {uid}"]
                injection = next((e for e in unit["result"]["events"] if e["kind"] == "injection"), None)
                if injection:
                    text += [f">>> Injected input, not generated / {injection['position']} / confirmed offset {injection['confirmed_offset']}",
                        decode(injection["payload_ids"]), ">>> Exact reconstructed continuation prefix", decode(injection["continuation_prefix_ids"]), ""]
                for i, call in enumerate(unit["calls"]):
                    text += [f">>> Model call {i} / {call['purpose']} / seed {call['seed']} / prefix_sha256 {call['prefix_sha256']}",
                        decode(call["token_ids"]), ""]
                if uid.endswith(":common"):
                    for event in unit["result"]["events"]:
                        if event["kind"] == "rollback":
                            text += [f">>> Discarded main draft/tail / terminal={event['terminal']}", decode(event["discarded_ids"]), ""]
                if unit["result"]["forced_thinking_end"]:
                    text += [">>> Controller inserted closing marker at work cap; marker was not sampled", "</think>", ""]
            text += [">>> Retained thinking", result["thinking_text"], ">>> Final answer", result["answer_text"], ">>> End of saved condition", ""]
            blocks.append("\n".join(text))
        data = ("\n"+"="*88+"\n\n").join(blocks).encode()
        name = case["id"]+"-all-11-conditions.txt"; (args.output/name).write_bytes(data)
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"selection": "first two heldout IDs in frozen input, all 11 conditions; chosen before outcomes",
        "questions": [c["id"] for c in selected], "condition_examples": 11*len(selected),
        "all_execution_unit_thinking_and_answer_decodes_verified": len(units),
        "tokenizer_sha256": vocab["tokenizer_sha256"], "files": files}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__": main()

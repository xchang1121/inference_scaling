"""Export full first-question trajectories from saved IDs and a decoder subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inference_scaling.retrieval.ablation import arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--token-map",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    prepared = json.loads((args.snapshot/"prepared.json").read_text(encoding="utf-8"))
    raw = (args.snapshot/"generation.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if not any(r.get("kind") == "complete" for r in rows):
        raise ValueError("wait for the completed queue before exporting final examples")
    question = prepared["cases"][0]
    selected = {(r["budget"],r["arm"]):r for r in rows if r.get("kind") == "measurement" and r["question_id"] == question["id"]}
    vocab = json.loads(args.token_map.read_text(encoding="utf-8"))
    assert vocab["tokenizer_sha256"] == prepared["tokenizer_sha256"]
    assert vocab["decoder"]["type"] == "ByteLevel"
    alphabet = list(range(33,127))+list(range(161,173))+list(range(174,256));codes=alphabet.copy();extra=0
    for b in range(256):
        if b not in alphabet:alphabet.append(b);codes.append(256+extra);extra+=1
    mapping = dict(zip(map(chr,codes),alphabet,strict=True))
    special = set(vocab["special_ids"])

    def decode(ids,skip_special=False):
        text = ''.join(vocab["vocab"][str(t)] for t in ids if not skip_special or t not in special)
        return bytes(mapping[c] for c in text).decode("utf-8",errors="replace")

    args.output.mkdir(parents=True,exist_ok=True)
    outputs={}
    for budget in (256,1024,4096):
        blocks=[]
        for arm in arms(budget):
            row=selected[budget,arm]
            assert hashlib.sha256(json.dumps(row["context_ids"]).encode()).hexdigest() == row["trajectory_sha256"]
            thinking=[t for s in row["segments"] if s["kind"] == "generated" and s["phase"] == "thinking" for t in s["token_ids"]]
            if 248069 in thinking:thinking=thinking[:thinking.index(248069)]
            assert decode(thinking) == row["thinking_text"]
            answer=[t for s in row["segments"] if s["kind"] == "generated" and s["phase"] == "answer" for t in s["token_ids"]]
            assert decode(answer,True) == row["answer_text"]
            variant=question["variants"][row["content_condition"]+"."+row["wrapper_condition"]]
            evidence=[t for s in row["segments"] if s["kind"] in {"evidence","prompt_evidence"} for t in s["token_ids"]]
            if evidence:assert decode(evidence) == variant["text"]
            text=[f"{question['id']} / thinking budget {budget} / {arm}",
                "完整原始 Qwen3.8-27B 轨迹；仅 >>> 行是导出器的说明。未省略或补写模型内容。",
                "原始回答可能有错误。预算强制结束与模型自然结束明确区分。",
                f"内容条件={row['content_condition']}；包装条件={row['wrapper_condition']}；thinking={row['thinking_tokens']}；answer={row['answer_generated_tokens']}；evidence={row['injected_tokens']}",
                f"finish_reason={row['finish_reason']}；trajectory_sha256={row['trajectory_sha256']}",
                f"控制器强制结束={any(e['kind']=='forced_thinking_end' for e in row['events'])}","",
                ">>> 完整初始输入；静态资料已经位于 user 消息中",decode(row["prompt_ids"]),""]
            groups=[];context=row["prompt_ids"].copy()
            for s in row["segments"]:
                if s["kind"] == "prompt_evidence":continue
                assert s["context_offset"] == len(context);context+=s["token_ids"]
                key=s["kind"],s.get("phase")
                if groups and groups[-1]["key"] == key and s["kind"] == "generated":groups[-1]["ids"]+=s["token_ids"]
                else:groups.append({"key":key,"offset":s["context_offset"],"ids":s["token_ids"].copy()})
            assert context == row["context_ids"]
            for g in groups:
                kind,phase=g["key"]
                label=("模型生成 / "+phase if kind == "generated" else "控制器注入资料 / 非模型生成" if kind == "evidence" else "控制器结构标记 / 非模型生成")
                text += [f">>> {label} / context_offset={g['offset']} / {len(g['ids'])} token",decode(g["ids"]),""]
            text += [">>> 本条轨迹结束；无省略。",""]
            blocks.append('\n'.join(text))
        name=f"{question['id']}-budget-{budget}-all-14-methods.txt"
        data=('\n'+'='*88+'\n\n').join(blocks).encode()
        (args.output/name).write_bytes(data)
        outputs[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    manifest={"question_id":question["id"],"selection":"first question in frozen input, every method and budget; not selected by outcome",
        "trajectories":42,"records_sha256":hashlib.sha256(raw).hexdigest(),"tokenizer_sha256":prepared["tokenizer_sha256"],
        "thinking_answer_and_evidence_decode_verified":True,"all_prefixes_and_trajectory_hashes_verified":True,"files":outputs}
    (args.output/"trace-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False))


if __name__ == "__main__":
    main()

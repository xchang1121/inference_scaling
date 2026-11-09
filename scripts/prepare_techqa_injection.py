"""Adapt NVIDIA's published TechQA-RAG-Eval to separate source/query/gold files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REVISION = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="train.json and corpus.zip from the pinned dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.raw/"train.json").read_text(encoding="utf-8"))
    # Public train.json contains both the original TRAIN and DEV partitions.
    candidates = [r for r in rows if r["id"].startswith("DEV_") and not r["is_impossible"] and r["answer"].strip()]
    candidates.sort(key=lambda r:hashlib.sha256(f"{args.seed}:{r['id']}".encode()).hexdigest())
    selected = candidates[:args.count]
    if len(selected) != args.count:
        raise ValueError("not enough answerable DEV examples")
    docs = []
    with zipfile.ZipFile(args.raw/"corpus.zip") as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".txt"):
                text = archive.read(name).decode("utf-8", errors="replace")
                title = text.splitlines()[0].removeprefix("Title: ").strip() if text.strip() else name
                docs.append({"id":Path(name).name, "title":title, "text":text,
                    "source":f"nvidia/TechQA-RAG-Eval@{REVISION}/{name}"})
    ids = {d["id"] for d in docs}
    questions, gold = [], []
    for row in selected:
        expected = [c["filename"] for c in row["contexts"]]
        if not set(expected) <= ids:
            raise ValueError(f"missing source documents for {row['id']}")
        questions.append({"id":row["id"], "dataset":"techqa_rag_eval_dev", "category":"technical_support",
            "question":row["question"]+"\n\nGive a concise technical answer. In the final answer, state the concrete fix or rule; omit the reasoning and introductory phrases."})
        gold.append({"id":row["id"], "answer":row["answer"], "document_ids":expected})
    for name, data in (("corpus",docs),("questions",questions),("gold",gold)):
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data),encoding="utf-8")
    manifest = {"source":"nvidia/TechQA-RAG-Eval", "revision":REVISION, "license":"Apache-2.0",
        "source_files":{n:hashlib.sha256((args.raw/n).read_bytes()).hexdigest() for n in ("train.json","corpus.zip")},
        "original_rows":len(rows), "answerable_dev_population":len(candidates), "selected_ids":[r["id"] for r in selected],
        "count":len(selected), "corpus_documents":len(docs), "seed":args.seed,
        "protocol":"Generative answerable-DEV pilot with full published reduced corpus; normalized answer token F1/EM, not the IBM extractive leaderboard.",
        "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jsonl")}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

"""Freeze public prompts separately from private CPU judge inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from coding_common import MODES, digest, read_jsonl

SEED = 20261104
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
HE_SHA = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
LCB_SHA = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"


def rank(dataset, task_id):
    return hashlib.sha256(f"{SEED}:{dataset}:{task_id}".encode()).hexdigest()


def lcb_prompt(row):
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        prompt += "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{row['starter_code']}\n```\n\n"
    else:
        prompt += "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in (("HumanEvalPlus-v0.1.10.jsonl", HE_SHA), ("livecodebench-v6.jsonl", LCB_SHA)):
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"data hash mismatch: {name}")
    he = sorted(read_jsonl(args.data / "HumanEvalPlus-v0.1.10.jsonl"), key=lambda r: rank("he", r["task_id"]))
    raw_lcb = read_jsonl(args.data / "livecodebench-v6.jsonl")
    buckets = {difficulty: sorted((r for r in raw_lcb if r["difficulty"] == difficulty),
                                key=lambda r: rank("lcb", r["question_id"]))[:count]
               for difficulty, count in {"easy": 25, "medium": 30, "hard": 45}.items()}
    # Pilot coverage is fixed before observing outputs; full cohort is proportional.
    pilot = [buckets[d][i] for i in range(2) for d in ("easy", "medium", "hard")]
    remaining = sorted([r for d, rows in buckets.items() for r in rows[2:]],
                       key=lambda r: rank("lcb-order", r["question_id"]))
    lcb = pilot + remaining
    he_public = [{"id": r["task_id"], "task_id": r["task_id"], "dataset": "humanevalplus",
                  "entry_point": r["entry_point"], "review_anchor": i < 6,
                  "prompt": "Implement the Python function described below. Return the complete function and necessary imports in a single fenced Python code block.\n\n```python\n" + r["prompt"] + "\n```"}
                 for i, r in enumerate(he)]
    lcb_public = [{"id": "lcb/" + r["question_id"], "task_id": r["question_id"], "dataset": "livecodebench",
                   "prompt": lcb_prompt(r), "review_anchor": i < 12,
                   **{k: r[k] for k in ("difficulty", "platform", "contest_date", "question_title")}}
                  for i, r in enumerate(lcb)]
    phases = [("humaneval-pilot", he_public[:6]), ("livecodebench-pilot", lcb_public[:6]),
              ("livecodebench-main", lcb_public[6:]), ("humaneval-full", he_public[6:])]
    public = []
    for phase, rows in phases:
        for row in rows:
            public.append({**row, "phase": phase, "global_index": len(public)})
    sources = {"humanevalplus": {"version": "v0.1.10", "sha256": HE_SHA, "count": len(he)},
               "livecodebench": {"revision": LCB_REVISION, "file": "test6.jsonl", "sha256": LCB_SHA,
                                 "available": len(raw_lcb), "selected": len(lcb),
                                 "difficulty": dict(Counter(r['difficulty'] for r in lcb)),
                                 "window": [min(r['contest_date'] for r in raw_lcb), max(r['contest_date'] for r in raw_lcb)]}}
    manifest = {"schema": 1, "seed": SEED, "modes": MODES, "sources": sources,
                "selection": "SHA256(seed:dataset:id) ranking within fixed difficulty quotas; predeclared pilot first",
                "problems": public}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("public-manifest.json", manifest),
                       ("judge-data.json", {"humanevalplus": {r['task_id']: r for r in he},
                                            "livecodebench": {r['question_id']: r for r in lcb}})):
        target = args.out / name
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to replace changed frozen input {target}")
        target.write_text(text, encoding="utf-8")
    print(json.dumps({"problems": len(public), "measurements": len(public) * len(MODES),
                      "manifest_digest": digest(manifest), "sources": sources}, indent=2))


if __name__ == "__main__":
    main()

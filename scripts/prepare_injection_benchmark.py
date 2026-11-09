"""Prepare a reproducible API-contract diagnostic; optionally import TechQA later.

The diagnostic is synthetic and is never presented as a published benchmark.
Only source-document fields enter corpus.jsonl. Questions and gold stay separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def controlled(seed=20261109, count=16):
    rng = random.Random(seed)
    docs, questions, gold = [], [], []
    actions = "retry_same_key, query_status, continue_cursor, restart_pagination, retry_after, acknowledge_only, apply_event"
    for number in range(16):
        service = "Aster-"+hashlib.sha256(f"{seed}:{number}".encode()).hexdigest()[:6]
        for version in (1, 2):
            revision = f"v{version}.{number}"
            retention = rng.choice([17, 29, 43, 61, 97])*60
            cursor_ttl = rng.choice([13, 31, 47, 83])*60
            floor = rng.choice([7, 19, 37, 59])
            policies = {
                "idempotency": f"For {service} {revision}, a timed-out POST with the ORIGINAL idempotency key can be retried with that key while its age is at most {retention} seconds. At a larger age, query the original request status first; do not immediately replay it. This is the complete replay-window contract.",
                "pagination": f"For {service} {revision}, opaque cursor tokens remain valid for at most {cursor_ttl} seconds since issuance. Continue using the current cursor at or below this age. Above this age, restart pagination from the first page. Do not infer expiration from other API versions.",
                "rate_limit": f"For {service} {revision}, after HTTP 429 the retry delay is max(Retry-After, {floor}) seconds, even when the server's Retry-After header is smaller. Retry after waiting; do not choose a new idempotency key merely because of a 429.",
                "webhook": f"For {service} {revision}, event ids are deduplicated for {retention} seconds. An already-seen event inside this interval must only be acknowledged. After this interval, apply the event again. The event-id retention rule takes precedence over a resource-version comparison for this endpoint.",
            }
            for category, policy in policies.items():
                docid = f"{service}-{revision}-{category}"
                docs.append({"id": docid, "title": f"{service} {revision} {category} API contract",
                             "text": policy, "source": "synthetic-api-contracts-v1"})
                if version != 2:
                    continue
                for side in (-1, 1):
                    value = (cursor_ttl if category == "pagination" else retention) + side*60
                    if category == "idempotency":
                        scene = f"A POST timed out. The original request has an idempotency key, whose age is {value} seconds. What should the client do immediately?"
                        answer = {"action": "retry_same_key" if side < 0 else "query_status", "wait_seconds": 0}
                    elif category == "pagination":
                        scene = f"The next-page cursor was issued {value} seconds ago. Should the client continue it or restart pagination?"
                        answer = {"action": "continue_cursor" if side < 0 else "restart_pagination", "wait_seconds": 0}
                    elif category == "rate_limit":
                        header = floor + side*3
                        scene = f"HTTP 429 was returned with Retry-After: {header}. What action and minimum waiting time are required?"
                        answer = {"action": "retry_after", "wait_seconds": max(header, floor)}
                    else:
                        scene = f"A webhook with an already-seen event id arrives {value} seconds after its first delivery. What should this endpoint do?"
                        answer = {"action": "acknowledge_only" if side < 0 else "apply_event", "wait_seconds": 0}
                    qid = f"api-{number}-{category}-{side}"
                    text = f"Use the {service} {revision} API contract. {scene}\nReason carefully. End with ONLY a JSON object containing action and wait_seconds. Allowed actions: {actions}."
                    questions.append({"id": qid, "dataset": "api_contract_diagnostic", "category": category, "question": text})
                    gold.append({"id": qid, "answer": answer, "document_ids": [docid]})
    # Balanced categories, fixed hash selection; no tuning on model outcomes.
    grouped = defaultdict(list)
    for q in questions:
        grouped[q["category"]].append(q)
    chosen = []
    for category, rows in sorted(grouped.items()):
        rows.sort(key=lambda q: hashlib.sha256(f"{seed}:{q['id']}".encode()).hexdigest())
        chosen.extend(rows[:count//4])
    chosen.sort(key=lambda q: q["id"])
    keep = {q["id"] for q in chosen}
    return docs, chosen, [g for g in gold if g["id"] in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    if args.count <= 0 or args.count % 4:
        parser.error("count must be positive and divisible by four")
    args.output.mkdir(parents=True, exist_ok=True)
    docs, questions, gold = controlled(args.seed, args.count)
    write_jsonl(args.output/"corpus.jsonl", docs)
    write_jsonl(args.output/"questions.jsonl", questions)
    write_jsonl(args.output/"gold.jsonl", gold)
    metadata = {"dataset": "synthetic-api-contracts-v1", "seed": args.seed, "count": len(questions),
                "description": "Controlled diagnostic, not a published or contamination-free leaderboard",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jsonl")}}
    (args.output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()

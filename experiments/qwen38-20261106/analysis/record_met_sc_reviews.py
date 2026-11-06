"""Append explicit content-review judgments, keyed by exact problem and output hashes."""
import json
import sys
from datetime import datetime, timezone
from met_sc_study import REVIEW, labels, read_jsonl

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
entries = json.load(sys.stdin)
cases = {c["case_id"]: c for c in read_jsonl(REVIEW / "blind_cases.jsonl")}
existing = labels()
for entry in entries:
    cid = entry["case_id"]
    assert cid in cases, cid
    assert cid not in existing or entry.pop("revision", False), ("already reviewed", cid)
    assert entry["grade"] in ("valid", "local_error", "critical_error")
    assert isinstance(entry["answer_equivalent"], bool)
    assert len(entry["evidence"]) >= 20
    entry["problem_id"] = cases[cid]["problem_id"]
    entry["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    entry["reviewer"] = "assistant; method/score/timing hidden in review packet"
    existing[cid] = entry
with (REVIEW / "labels.jsonl").open("a", encoding="utf-8", newline="\n") as file:
    for entry in entries:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(json.dumps({"appended": len(entries), "reviewed_unique": len(existing)}))

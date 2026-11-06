"""Local provenance checks and method-hidden review packets for the 100-question study."""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent / "met_sc_20261103"
REVIEW = ROOT / "review"
CONTROL_MODES = ("met-k1", "met-k2", "block-met-k2", "is-self-certainty")
NEW_MODES = ("met-k1-self-certainty", "met-k2-self-certainty", "block-met-k2-self-certainty")
FOCUS_MODES = CONTROL_MODES + NEW_MODES


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def case_id(problem_id, text):
    return hashlib.sha256((problem_id + "\0" + text).encode("utf-8")).hexdigest()[:20]


def load_study():
    rows, metadata, keys = [], [], set()
    for replica in ("a", "b"):
        for path in sorted((ROOT / replica / "results").glob("*/*.jsonl")):
            data = read_jsonl(path)
            meta = [r for r in data if r["kind"] == "metadata"]
            assert len(meta) == 1, path
            meta = meta[0]
            meta["local_file"] = str(path.relative_to(ROOT))
            meta["replica"] = replica
            metadata.append(meta)
            assert meta["dataset"]["source_sha256"] == "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
            for r in data:
                if r["kind"] != "measurement":
                    continue
                key = (r["mode"], r["problem_id"], r["repeat"])
                assert key not in keys, key
                keys.add(key)
                r["replica"] = replica
                r["source_file"] = meta["local_file"]
                r["cohort"] = "old" if ("selector-quality" in str(path) or "-old" in path.name) else "new"
                r["case_id"] = case_id(r["problem_id"], r.get("answer_text", ""))
                r["metadata"] = meta
                rows.append(r)
    return rows, metadata


def labels():
    path = REVIEW / "labels.jsonl"
    result = {}
    for entry in read_jsonl(path) if path.exists() else []:
        assert entry["grade"] in ("valid", "local_error", "critical_error")
        assert isinstance(entry["answer_equivalent"], bool)
        assert entry.get("evidence", "").strip()
        result[entry["case_id"]] = entry
    return result


def prepare():
    rows, metas = load_study()
    dataset_path = ROOT / "a/runtime/datasets/math500/test.jsonl"
    dataset = {r["unique_id"]: r for r in read_jsonl(dataset_path)}
    REVIEW.mkdir(parents=True, exist_ok=True)
    cases, mapping = {}, []
    for r in rows:
        if r["mode"] not in FOCUS_MODES or r["status"] != "complete":
            continue
        pid, cid = r["problem_id"], r["case_id"]
        q = dataset[pid]
        cases[cid] = {"case_id": cid, "problem_id": pid, "question": q["problem"],
                      "reference_answer": q["answer"], "reference_solution": q["solution"],
                      "final_content": r["answer_text"]}
        mapping.append({k: r[k] for k in ("case_id", "problem_id", "mode", "replica", "cohort", "source_file", "seed", "correct")})
    ordered = sorted(cases.values(), key=lambda x: (hashlib.sha256(x["problem_id"].encode()).hexdigest(), x["case_id"]))
    (REVIEW / "blind_cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in ordered), encoding="utf-8")
    (REVIEW / "method_key.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    reviewed = labels()
    focused = [r for r in rows if r["mode"] in FOCUS_MODES]
    controls = {(r["mode"], r["problem_id"], r["repeat"]): r for r in rows}
    checks = []
    for r in focused:
        if r["mode"] not in NEW_MODES:
            continue
        old = controls[(r["mode"].removesuffix("-self-certainty"), r["problem_id"], r["repeat"])]
        fast = controls[("is-self-certainty", r["problem_id"], r["repeat"])]
        assert r["replica"] == old["replica"] == fast["replica"]
        assert r["seed"] == old["seed"] == fast["seed"]
        assert r["algorithm_config"] == old["algorithm_config"], (r["mode"], r["problem_id"])
        assert not r["algorithm_config"]["selection_ready"]
        assert r["reward_config"] == {"temperature": .25, "score_temperature": 1, "scope": "thinking"}
        # Existing fast SC and supplement use the exact same frozen settings per cohort.
        assert r["metadata"]["settings"] == fast["metadata"]["settings"]
        assert r["metadata"]["engine"] == fast["metadata"]["engine"]
        checks.append({"mode": r["mode"], "problem_id": r["problem_id"], "seed_match": True,
                       "same_kernel_config": True, "fast_sc_settings_equal": True,
                       "status": r["status"], "no_early_cancellation": r.get("counters", {}).get("cancelled_requests") == 0,
                       "no_rescoring_forward": r.get("counters", {}).get("score_forward_token_slots") == 0,
                       "no_fallback": not r.get("fallbacks"), "no_preemption": r.get("counters", {}).get("preemptions") == 0})
    (ROOT / "alignment-checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"measurements": len(focused), "unique_cases": len(cases), "reviewed_unique_cases": len(set(cases) & set(reviewed)),
            "remaining_unique_cases": len(set(cases) - set(reviewed)), "new_measurements": len(checks),
            "method_counts": dict(Counter(r["mode"] for r in focused)),
            "alignment_checks_passed": True}


def batch(problem_limit, problem_id=None):
    cases = read_jsonl(REVIEW / "blind_cases.jsonl")
    done = labels()
    groups = defaultdict(list)
    for c in cases:
        if c["case_id"] not in done and (problem_id is None or c["problem_id"] == problem_id):
            groups[c["problem_id"]].append(c)
    for pid, group in list(groups.items())[:problem_limit]:
        print(json.dumps({"problem_id": pid, "question": group[0]["question"],
                          "reference_answer": group[0]["reference_answer"],
                          "reference_solution": group[0]["reference_solution"]}, ensure_ascii=False))
        for c in group:
            print(json.dumps({"case_id": c["case_id"], "final_content": c["final_content"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "batch"))
    parser.add_argument("--problems", type=int, default=3)
    parser.add_argument("--problem")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare()))
    else:
        batch(args.problems, args.problem)

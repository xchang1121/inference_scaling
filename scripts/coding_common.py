"""Public-only coding benchmark data and immutable experiment identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inference_scaling.datasets.base import Dataset, Grade, Problem, file_sha256

MODES = ("base", "is", "sir-k1", "met-k1", "sir-k2", "met-k2", "block-met-k2",
         "bon-consilience", "is-self-certainty", "met-k1-self-certainty",
         "met-k2-self-certainty", "block-met-k2-self-certainty")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_code(text):
    """LiveCodeBench generic extraction: the last pair of fenced delimiters.

    Apply exactly the same extraction to every arm and both datasets. No repair,
    canonical implementation, execution feedback, or extra code completion.
    """
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[fences[-2] + 1:fences[-1]]) if len(fences) >= 2 else ""


class PublicCodingDataset(Dataset):
    name = "coding-public"
    oracle = False

    def __init__(self, path, *, shard=0, shards=1, limit=None):
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not 0 <= shard < shards:
            raise ValueError("invalid shard")
        allowed = {"id", "dataset", "prompt", "task_id", "difficulty", "platform", "contest_date",
                   "entry_point", "global_index", "phase", "review_anchor", "question_title"}
        rows = manifest["problems"]
        if any(set(row) - allowed for row in rows):
            raise ValueError("public manifest contains unknown fields")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate problem id")
        self.manifest = manifest
        self.rows = {row["id"]: row for row in rows}
        selected = [row for row in rows if row["global_index"] % shards == shard]
        if limit is not None:
            selected = selected[:limit]
        super().__init__({"path": str(path), "prompt_template": "{question}"},
                         tuple(Problem(row["id"], row["prompt"], "", row) for row in selected),
                         file_sha256(path))

    def answer(self, text):
        return extract_code(text) or None

    def same(self, left, right):
        return left == right

    def grade(self, text, problem):
        # Best-of-N asks for display-only candidate grades. There are no tests or
        # reference solutions in this process. The runner removes this placeholder.
        code = self.answer(text)
        return Grade(code, code is not None, False)


def completed_keys(rows, identity):
    """Fail closed on changed experiments, duplicates or damaged append-only files."""
    metadata = [r for r in rows if r.get("kind") == "metadata"]
    if rows and (len(metadata) != 1 or metadata[0]["identity"] != identity):
        raise ValueError("resume identity mismatch")
    complete = [r for r in rows if r.get("kind") == "measurement" and r.get("status") == "complete"]
    keys = [(r["problem_id"], r["mode"]) for r in complete]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate completed measurement")
    return set(keys)

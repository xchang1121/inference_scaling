"""Optional CPU-only oracle audit against a pinned, local research checkout.

The serving/training package never imports this source. This audit reads a small
set of CPU definitions from the pinned Git object, not from mutable working files,
and does not load the author's model, engine, package initializers or GPU kernels.
"""

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
from typing import Sequence

import torch

from blockspec.tree import build_tree, traverse_greedy


def load_reference(checkout):
    lock = Path(__file__).resolve().parents[1] / "references" / "upstream.lock.json"
    commit = json.loads(lock.read_text(encoding="utf-8"))["source"]["commit"]
    source_file = "nano_vllm_uno/engine/draft_tree.py"
    source = subprocess.run(["git", "--no-replace-objects", "-C", str(checkout), "show",
                             f"{commit}:{source_file}"], check=True, capture_output=True).stdout
    wanted = {"_TreeNode", "DraftTree", "build_tree_from_candidates", "walk_tree"}
    selected = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise ValueError("pinned reference does not expose the expected CPU contract")
    # walk_tree's CUDA branch is never reached: every audit tensor is on CPU.
    namespace = {"__name__": __name__, "torch": torch, "Tensor": torch.Tensor,
                 "dataclass": dataclass, "field": field, "Sequence": Sequence,
                 "heapq": heapq, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), f"reference:{commit}:{source_file}", "exec"), namespace)
    return namespace, {"commit": commit, "file": source_file, "source_sha256": hashlib.sha256(source).hexdigest()}


def audit(checkout, *, trials=100):
    if trials < 1:
        raise ValueError("positive trial count required")
    oracle, provenance = load_reference(checkout)
    generator = torch.Generator().manual_seed(8721)
    built = walked = 0
    for depth in (1, 2, 3, 4):
        q = torch.randn(depth, 5, generator=generator, dtype=torch.float64).softmax(-1)
        values, ids = q.topk(3, dim=-1)
        for budget in (1, 2, 5, 12):
            ours = build_tree(2, q, top_k=3, prefix_budget=budget)
            theirs = oracle["build_tree_from_candidates"](2, ids.tolist(), values.log().tolist(), budget)
            assert ours.tokens == list(theirs.token_ids)
            assert ours.parents == list(theirs.parent_indices)
            assert ours.depths == list(theirs.depths)
            torch.testing.assert_close(torch.tensor(ours.scores), torch.tensor(theirs.log_masses), atol=1e-6, rtol=1e-6)
            built += 1
            tree_tokens = torch.tensor([ours.tokens])
            parents = torch.tensor([ours.parents])
            for _ in range(trials):
                # Arbitrary pre-sampled target IDs couple deterministic traversal;
                # their probability law is verified independently by unit tests.
                target = torch.randint(5, tree_tokens.shape, generator=generator)
                emitted, cached, lengths = oracle["walk_tree"](tree_tokens, parents, target,
                                                               max_depth=depth, pad_token_id=-1)
                result = traverse_greedy(ours, target[0], budget=depth + 2)
                assert result.tokens == emitted[0, :int(lengths[0])].tolist()
                assert result.path == cached[0][cached[0] >= 0].tolist()
                walked += 1
    return {"reference": provenance, "tree_builds_matched": built, "target_walks_matched": walked,
            "scope": "non-tied prefix scores and CPU pre-sampled target paths; not model/kernel/TPS equivalence"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing read-only Git checkout; no download")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.source, trials=args.trials)), flush=True)

"""Prefix-closed Uno candidate trees and past-only rank calibration.

Tree drafting is a performance heuristic. Exactness comes from sampling once
from each reached target node, never from treating path scores as target laws.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TreeConfig:
    block_size: int = 8
    nodes: int = 16
    top_k: int = 4
    include_spine: bool = True
    online_rank: bool = False
    prior_strength: float = 8.0
    decay: float = 0.98

    def validate(self) -> None:
        if self.block_size < 2 or self.nodes < 1 or self.top_k < 1:
            raise ValueError("positive tree sizes and block_size >= 2 required")
        if self.include_spine and self.nodes < self.block_size:
            raise ValueError("spine requires at least block_size nodes")
        if not math.isfinite(self.prior_strength) or self.prior_strength <= 0:
            raise ValueError("prior_strength must be finite and positive")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must lie in (0, 1]")


@dataclass(frozen=True)
class TreeNode:
    token: int
    parent: int
    depth: int
    rank: int
    weight: float


@dataclass(frozen=True)
class CandidateTree:
    nodes: tuple[TreeNode, ...]

    def ancestor_indices(self, index: int) -> tuple[int, ...]:
        path = []
        while index >= 0:
            path.append(index)
            index = self.nodes[index].parent
        return tuple(reversed(path))

    @property
    def surrogate_committed(self) -> float:
        return 2.0 + sum(node.weight for node in self.nodes[1:])


def build_tree(
    free_token: int, token_ids: Sequence[Sequence[int]],
    rank_weights: Sequence[Sequence[float]], *, nodes: int,
    include_spine: bool = True,
) -> CandidateTree:
    """Best-first extension, optionally constrained to contain the top-1 spine."""
    if nodes < 1 or len(token_ids) != len(rank_weights):
        raise ValueError("invalid tree budget or row alignment")
    for ids, weights in zip(token_ids, rank_weights, strict=True):
        if len(ids) != len(weights) or not ids or len(set(ids)) != len(ids):
            raise ValueError("each depth needs aligned unique candidate labels")
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) > 1 + 1e-6:
            raise ValueError("rank weights must be sub-probabilities")
    if include_spine and nodes < len(token_ids) + 1:
        raise ValueError("budget is too short for the forced spine")
    tree = [TreeNode(int(free_token), -1, 0, -1, 1.0)]
    selected = set()
    if include_spine:
        for depth, (ids, weights) in enumerate(zip(token_ids, rank_weights, strict=True), 1):
            parent = len(tree) - 1
            tree.append(TreeNode(int(ids[0]), parent, depth, 0, tree[parent].weight * weights[0]))
            selected.add((parent, 0))
    frontier = []

    def expand(parent: int) -> None:
        row = tree[parent].depth
        if row >= len(token_ids):
            return
        for rank, weight in enumerate(rank_weights[row]):
            if (parent, rank) not in selected:
                heapq.heappush(frontier, (-tree[parent].weight * weight, parent, rank))

    for parent in range(len(tree)):
        expand(parent)
    while len(tree) < nodes and frontier:
        neg_weight, parent, rank = heapq.heappop(frontier)
        depth = tree[parent].depth + 1
        tree.append(TreeNode(int(token_ids[depth - 1][rank]), parent, depth, rank, -neg_weight))
        expand(len(tree) - 1)
    return CandidateTree(tuple(tree))


@dataclass(frozen=True)
class TreeWalk:
    committed: tuple[int, ...]
    # Includes root and every matched node, i.e. all cached output tokens
    # before the terminal token. Truncation must slice this with actual C-1.
    path_indices: tuple[int, ...]
    observations: tuple[tuple[int, int], ...]
    used_leaf_lookahead: bool


def walk_target_draws(tree: CandidateTree, target_draws: Sequence[int]) -> TreeWalk:
    if len(tree.nodes) != len(target_draws):
        raise ValueError("one independent target draw per tree node is required")
    children: dict[tuple[int, int], int] = {}
    for index, node in enumerate(tree.nodes[1:], 1):
        key = (node.parent, node.token)
        if key in children or not 0 <= node.parent < index:
            raise ValueError("tree must have ordered parents and distinct sibling labels")
        children[key] = index
    committed = [tree.nodes[0].token]
    path, observations = [0], []
    current = 0
    while True:
        draw = int(target_draws[current])
        committed.append(draw)
        observations.append((tree.nodes[current].depth, draw))
        child = children.get((current, draw))
        if child is None:
            has_child = any(parent == current for parent, _ in children)
            return TreeWalk(tuple(committed), tuple(path), tuple(observations), not has_child)
        path.append(child)
        current = child


class RankCalibrator:
    """Request-local rank counts; no target or LoRA parameter mutation."""
    def __init__(self, config: TreeConfig) -> None:
        self.config = config
        self.counts = [[0.0] * config.top_k for _ in range(config.block_size - 1)]
        self.totals = [0.0] * (config.block_size - 1)
        self.observations = 0

    def weights(self, prior: Sequence[Sequence[float]]) -> list[list[float]]:
        strength = self.config.prior_strength
        if not self.config.online_rank:
            return [list(row) for row in prior]
        return [
            [(self.counts[d][k] + strength * q) / (self.totals[d] + strength)
             for k, q in enumerate(row)]
            for d, row in enumerate(prior)
        ]

    def observe(
        self, observations: Sequence[tuple[int, int]], token_ids: Sequence[Sequence[int]],
    ) -> None:
        if not self.config.online_rank:
            return
        for d, token in observations:
            if d >= len(self.counts):
                continue
            self.counts[d] = [x * self.config.decay for x in self.counts[d]]
            self.totals[d] = self.totals[d] * self.config.decay + 1.0
            if token in token_ids[d]:
                self.counts[d][list(token_ids[d]).index(token)] += 1.0
            self.observations += 1

    def snapshot(self) -> dict[str, object]:
        return {"observations": self.observations, "counts": self.counts, "totals": self.totals}

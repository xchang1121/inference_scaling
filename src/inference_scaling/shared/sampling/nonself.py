"""Exclude an unchanged suffix, using the current trajectory's base probabilities.

The cut and first divergence are sampled before generation. Copied tokens are
an exact part of this conditional proposal, not a second model's prediction.
Only the current pool needs per-token probabilities; no history is persisted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NonselfPlan:
    cut: int
    divergence: int
    change_mass: float


def nonself_cut_law(logprobs, probabilities):
    logs = np.asarray(logprobs, dtype=np.float64)
    chances = np.asarray(probabilities, dtype=np.float64)
    if (logs.ndim != 1 or not len(logs) or logs.shape != chances.shape or not np.isfinite(logs).all()
            or (logs > 0).any() or not np.isfinite(chances).all() or (chances < 0).any()
            or not np.isclose(chances.sum(), 1.)):
        raise ValueError("finite base log-probabilities and a normalized cut law are required")
    mass = chances * -np.expm1(np.cumsum(logs[::-1])[::-1])
    total = float(mass.sum())
    return total, mass / total if total > 0 else mass


def plan_nonself(logprobs, probabilities, rng):
    total, chances = nonself_cut_law(logprobs, probabilities)
    if total == 0:
        return None
    cut = int(rng.choice(len(chances), p=chances))
    logs = np.asarray(logprobs, dtype=np.float64)[cut:]
    mass = np.exp(np.r_[0., np.cumsum(logs[:-1])]) * -np.expm1(logs)
    divergence = cut + int(rng.choice(len(mass), p=mass / mass.sum()))
    return NonselfPlan(cut, divergence, total)


class FirstTokenExclusions:
    """One corrected draw per request; continuations alias the original seed.

    The worker serializes these operations with sampling. Disjoint request
    groups can overlap, and removing a failed registration is idempotent.
    """

    def __init__(self):
        self.entries = {}
        self.aliases = {}

    def add(self, entries):
        entries = tuple(entries)
        seeds = [seed for seed, _ in entries]
        if (not entries or len(set(seeds)) != len(entries)
                or any(type(seed) is not int or seed < 0 or type(token) is not int or token < 0
                       for seed, token in entries)):
            raise ValueError("first-token exclusion requires distinct nonnegative seeds and token IDs")
        if any(seed in self.aliases for seed in seeds):
            raise ValueError("a first-token exclusion seed is already registered")
        for seed, token in entries:
            self.entries[seed] = {"excluded": token, "selected": None}
            self.aliases[seed] = seed

    def bind(self, entries):
        entries = tuple(entries)
        if (len({alias for _, alias in entries}) != len(entries)
                or any(original not in self.entries or type(alias) is not int or alias < 0
                       or alias in self.aliases and self.aliases[alias] != original for original, alias in entries)):
            raise ValueError("unknown original seed or colliding first-token exclusion alias")
        self.aliases.update((alias, original) for original, alias in entries)

    def active(self, seed):
        original = self.aliases.get(seed)
        return original is not None and self.entries[original]["selected"] is None

    @property
    def has_active(self):
        return any(state["selected"] is None for state in self.entries.values())

    def sample(self, seed, logits):
        original = self.aliases[seed]
        state = self.entries[original]
        values = np.array(logits, dtype=np.float64, copy=True)
        token = state["excluded"]
        if (state["selected"] is not None or values.ndim != 1 or not 0 <= token < len(values)
                or not np.isfinite(values).all()):
            raise ValueError("an unused exclusion and finite full-support logits are required")
        # Mask before normalization: subtracting a rounded probability near one
        # would erase the remaining mass. The native base payload stays intact.
        values[token] = -np.inf
        values = np.exp(values - values.max())
        if not np.isfinite(values).all() or not values.sum() > 0:
            raise ValueError("no alternative token is available")
        cumulative = np.cumsum(values / values.sum())
        cumulative /= cumulative[-1]
        rng = np.random.default_rng(np.random.SeedSequence([original, 20261028, 42]))
        selected = int(np.searchsorted(cumulative, rng.random(), side="right"))
        if selected == token or selected >= len(values):
            raise ArithmeticError("invalid first-divergence token")
        state["selected"] = selected
        return selected

    def remove(self, seeds):
        seeds = set(seeds)
        removed = {seed: self.entries.pop(seed) for seed in seeds if seed in self.entries}
        self.aliases = {alias: original for alias, original in self.aliases.items() if original not in seeds}
        touched = sum(state["selected"] is not None for state in removed.values())
        return {"registered": len(removed), "touched": touched, "constrained_tokens": touched,
                "unresolved": len(removed) - touched, "maximum_cdf_error": 0.,
                "trajectories": {str(seed): {"tokens": int(state["selected"] is not None),
                    "resolved": state["selected"] is not None, **state} for seed, state in removed.items()}}

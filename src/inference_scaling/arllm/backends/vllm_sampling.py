"""Optional worker-side sampling constraints, independent of model architecture.

The native sampler still computes the original processed probability payload;
only registered tokens are replaced before the enclosing sampler gathers their
log-probabilities. Unregistered requests and resolved prefixes stay native.
"""
from __future__ import annotations

from typing import Any

import torch

from inference_scaling.shared.sampling.stratified import InitialStrata
from inference_scaling.shared.sampling.nonself import FirstTokenExclusions


class SamplingConstraintsWorker:
    model_runner: Any

    def _sampling_states(self):
        if not hasattr(self, "_initial_strata"):
            self._initial_strata = InitialStrata()
            self._first_token_exclusion = FirstTokenExclusions()
            self._constrained_sampling_forward = None
        return (self._initial_strata, self._first_token_exclusion)

    def begin_initial_strata(self, entries):
        self._sampling_states()
        if any(seed in self._first_token_exclusion.aliases for seed, _, _ in entries):
            raise ValueError("sampling constraint seeds must not overlap")
        self._initial_strata.add(entries)
        self._install_sampling_constraints()
        return {"registered": len(entries)}

    def bind_initial_strata(self, entries):
        if any(alias in self._first_token_exclusion.aliases for _, alias in entries):
            raise ValueError("sampling constraint aliases must not overlap")
        self._initial_strata.bind(entries)
        return {"bound": len(entries)}

    def end_initial_strata(self, seeds):
        self._sampling_states()
        result = self._initial_strata.remove(seeds)
        self._restore_sampling_constraints()
        return result

    def begin_first_token_exclusion(self, entries):
        self._sampling_states()
        if any(seed in self._initial_strata.aliases for seed, _ in entries):
            raise ValueError("sampling constraint seeds must not overlap")
        self._first_token_exclusion.add(entries)
        self._install_sampling_constraints()
        return {"registered": len(entries)}

    def bind_first_token_exclusion(self, entries):
        if any(alias in self._initial_strata.aliases for _, alias in entries):
            raise ValueError("sampling constraint aliases must not overlap")
        self._first_token_exclusion.bind(entries)
        return {"bound": len(entries)}

    def end_first_token_exclusion(self, seeds):
        self._sampling_states()
        result = self._first_token_exclusion.remove(seeds)
        self._restore_sampling_constraints()
        return result

    def _restore_sampling_constraints(self):
        if self._constrained_sampling_forward is not None and not any(state.has_active for state in self._sampling_states()):
            self.model_runner.sampler.topk_topp_sampler.forward = self._constrained_sampling_forward
            self._constrained_sampling_forward = None

    def _install_sampling_constraints(self):
        states = self._sampling_states()
        if self._constrained_sampling_forward is not None or not any(state.has_active for state in states):
            return
        sampler = self.model_runner.sampler.topk_topp_sampler
        original = self._constrained_sampling_forward = sampler.forward

        def forward(logits, generators, k, p):
            selected = [(row, generator.initial_seed(), state) for row, generator in generators.items()
                        for state in states if state.active(generator.initial_seed())]
            if not selected:
                return original(logits, generators, k, p)
            rows = torch.tensor([row for row, _, _ in selected], device=logits.device)
            if ((k is not None and any(0 < value < logits.shape[-1] for value in k[rows].tolist()))
                    or (p is not None and any(value < 1 for value in p[rows].tolist()))):
                raise ValueError("sampling constraints require full-support sampling")
            choices = [None] * len(selected)
            for state in states:
                members = [(i, row, seed) for i, (row, seed, owner) in enumerate(selected) if owner is state]
                if not members:
                    continue
                indices = torch.tensor([row for _, row, _ in members], device=logits.device)
                values = logits.index_select(0, indices)
                values = (values.softmax(dim=-1, dtype=torch.float32) if state is self._initial_strata else values.float()).cpu().numpy()
                for index, (position, _, seed) in enumerate(members):
                    choices[position] = state.sample(seed, values[index])
            tokens, payload = original(logits, generators, k, p)
            tokens[rows] = torch.tensor(choices, dtype=tokens.dtype, device=tokens.device)
            self._restore_sampling_constraints()
            return tokens, payload

        sampler.forward = forward

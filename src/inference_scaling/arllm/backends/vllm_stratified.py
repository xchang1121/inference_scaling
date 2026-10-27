"""Optional worker-side initial strata, independent of model architecture.

The native sampler still computes the original processed probability payload;
only registered tokens are replaced before the enclosing sampler gathers their
log-probabilities. Later suffix proposals and unregistered requests stay native.
"""
from __future__ import annotations

import torch

from inference_scaling.shared.sampling.stratified import InitialStrata


class InitialStrataWorker:
    def begin_initial_strata(self, entries):
        if not hasattr(self, "_initial_strata"):
            self._initial_strata = InitialStrata()
            self._initial_strata_forward = None
        self._initial_strata.add(entries)
        self._install_initial_strata()
        return {"registered": len(entries)}

    def bind_initial_strata(self, entries):
        self._initial_strata.bind(entries)
        return {"bound": len(entries)}

    def end_initial_strata(self, seeds):
        state = getattr(self, "_initial_strata", None)
        if state is None:
            return {"registered": 0, "touched": 0, "constrained_tokens": 0,
                    "unresolved": 0, "maximum_cdf_error": 0.}
        result = state.remove(seeds)
        self._restore_initial_strata()
        return result

    def _restore_initial_strata(self):
        if self._initial_strata_forward is not None and not self._initial_strata.has_active:
            self.model_runner.sampler.topk_topp_sampler.forward = self._initial_strata_forward
            self._initial_strata_forward = None

    def _install_initial_strata(self):
        if self._initial_strata_forward is not None or not self._initial_strata.has_active:
            return
        sampler = self.model_runner.sampler.topk_topp_sampler
        original = self._initial_strata_forward = sampler.forward
        state = self._initial_strata

        def forward(logits, generators, k, p):
            selected = [(row, generator.initial_seed()) for row, generator in generators.items()
                        if state.active(generator.initial_seed())]
            if not selected:
                return original(logits, generators, k, p)
            rows = torch.tensor([row for row, _ in selected], device=logits.device)
            if ((k is not None and any(0 < value < logits.shape[-1] for value in k[rows].tolist()))
                    or (p is not None and any(value < 1 for value in p[rows].tolist()))):
                raise ValueError("initial strata require full-support sampling")
            probabilities = logits.index_select(0, rows).softmax(dim=-1, dtype=torch.float32).cpu().numpy()
            choices = [state.sample(seed, probabilities[index]) for index, (_, seed) in enumerate(selected)]
            tokens, payload = original(logits, generators, k, p)
            tokens[rows] = torch.tensor(choices, dtype=tokens.dtype, device=tokens.device)
            self._restore_initial_strata()
            return tokens, payload

        sampler.forward = forward

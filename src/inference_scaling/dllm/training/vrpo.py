"""Variance-reduced ELBO estimators used by dLLM preference optimization.

VRPO calls the shared-randomness construction "antithetic sampling": the
current and reference models evaluate the same timesteps and the same masked
sequences.  It does not pair a timestep ``t`` with ``1 - t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inference_scaling.dllm.config import VRPOSamplingConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class VRPOMaskSample:
    timestep_index: int
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VRPOMaskPlan:
    answer_length: int
    samples: tuple[VRPOMaskSample, ...]


def sample_vrpo_mask_plan(answer_length: int, config: VRPOSamplingConfig, rng: np.random.Generator) -> VRPOMaskPlan:
    """Draw the doubly Monte Carlo mask plan used by one ELBO estimate.

    A discrete timestep is represented by a mask count sampled uniformly from
    ``1, ..., answer_length``.  All masks assigned to that timestep share the
    count but use independent uniform subsets.  The variance-optimal fixed
    budget setting is ``masks_per_timestep=1``.
    """

    samples = []
    for timestep_index in range(config.timestep_samples):
        mask_count = int(rng.integers(1, answer_length + 1))
        for _ in range(config.masks_per_timestep):
            positions = sorted(rng.choice(answer_length, size=mask_count, replace=False).tolist())
            samples.append(VRPOMaskSample(timestep_index, tuple(positions)))
    return VRPOMaskPlan(answer_length, tuple(samples))


def estimate_masked_elbo(
    model: Any,
    *,
    prompt: TokenSequence,
    answer: TokenSequence,
    mask_token_id: int,
    plan: VRPOMaskPlan,
) -> Any:
    """Return a differentiable Monte Carlo estimate of the conditional ELBO."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("VRPO estimation requires PyTorch") from exc
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    batch = torch.tensor(prompt + answer, dtype=torch.long, device=device).repeat(len(plan.samples), 1)
    masked = torch.zeros_like(batch, dtype=torch.bool)
    for row, sample in enumerate(plan.samples):
        masked[row, [len(prompt) + position for position in sample.positions]] = True
    logits = model(torch.where(masked, mask_token_id, batch)).logits
    token_losses = functional.cross_entropy(logits[masked].float(), batch[masked], reduction="none")
    # Each row's masked-token loss, divided by its mask ratio.
    losses = torch.zeros(len(plan.samples), device=device).index_add(0, masked.nonzero()[:, 0], token_losses)
    return -(losses * plan.answer_length / masked.sum(-1)).mean()


@dataclass(frozen=True, slots=True)
class VRPOPreferenceEstimate:
    loss: Any
    preference_score: Any


class AdapterDisabledReference:
    """Evaluate the frozen base policy through a trainable PEFT model.

    Current-policy and reference-policy calls share one resident base model.
    This avoids loading a second multi-billion-parameter checkpoint while the
    adapter context provides the exact frozen reference used by DPO-style VRPO.
    """

    def __init__(self, model: Any) -> None:
        if not callable(getattr(model, "disable_adapter", None)):
            raise TypeError("model must provide the PEFT disable_adapter context")
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        was_training = bool(getattr(self.model, "training", False))
        try:
            self.model.eval()
            with self.model.disable_adapter():
                return self.model(*args, **kwargs)
        finally:
            self.model.train(was_training)


def vrpo_forward_token_slots(
    *, prompt_length: int, chosen_length: int, rejected_length: int, config: VRPOSamplingConfig,
) -> dict[str, int]:
    """Return current/reference forward token slots for one preference pair."""

    one_policy = config.forward_passes * (2 * prompt_length + chosen_length + rejected_length)
    return {"current_policy": one_policy, "reference_policy": one_policy, "total": 2 * one_policy}


def estimate_vrpo_preference_loss(
    current_model: Any,
    reference_model: Any,
    *,
    prompt: TokenSequence,
    chosen: TokenSequence,
    rejected: TokenSequence,
    mask_token_id: int,
    config: VRPOSamplingConfig,
    beta: float,
    seed: int,
) -> VRPOPreferenceEstimate:
    """Estimate the ELBO-based DPO loss with VRPO variance reduction."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("VRPO estimation requires PyTorch") from exc
    if beta <= 0:
        raise ValueError("beta must be positive")
    seeds = SeedStream(seed)

    def elbo(model: Any, policy: str, name: str, answer: TokenSequence) -> Any:
        # Antithetic sampling evaluates the reference on the current policy's masks.
        stream = seeds.generator("vrpo", "current" if config.antithetic else policy, name)
        plan = sample_vrpo_mask_plan(len(answer), config, stream)
        return estimate_masked_elbo(model, prompt=prompt, answer=answer, mask_token_id=mask_token_id, plan=plan)

    current = elbo(current_model, "current", "chosen", chosen) - elbo(current_model, "current", "rejected", rejected)
    with torch.no_grad():
        reference = (elbo(reference_model, "reference", "chosen", chosen)
                     - elbo(reference_model, "reference", "rejected", rejected))
    preference_score = beta * (current - reference)
    return VRPOPreferenceEstimate(-functional.logsigmoid(preference_score), preference_score)


__all__ = [
    "AdapterDisabledReference",
    "VRPOMaskPlan",
    "VRPOMaskSample",
    "VRPOPreferenceEstimate",
    "estimate_masked_elbo",
    "estimate_vrpo_preference_loss",
    "sample_vrpo_mask_plan",
    "vrpo_forward_token_slots",
]

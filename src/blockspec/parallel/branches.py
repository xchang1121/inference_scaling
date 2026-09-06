"""Draft policies supply verifier inputs and clean history to a shared loop."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..diffusion import UniformNoise
from ..state import Cache, trim_cache as crop_cache
from ..sampling import SamplingConfig
from .sampling import ProposalSampler


@dataclass
class DraftBatch:
    verifier_inputs: Tensor
    candidates: Tensor
    proposal: Tensor | None
    cache: Cache | None
    guaranteed: list[int]
    draft_inputs: Tensor | None = None
    draft_cache: Cache | None = None
    boundary: object = None
    calibration_feedback: object = None
    collect_feedback: bool = False


class MaskedAttentionBranch:
    """The committed anchor is reprocessed with a bidirectional masked block."""

    name = "masked_attention"
    initial_ar_token = True
    input_budget_extra = 1

    def __init__(self, model):
        self.model = model
        self.default_block_size = model.config.block_size

    def ar(self, tokens, cache=None, logits_to_keep=0):
        result = self.model(tokens, cache=cache, logits_to_keep=logits_to_keep)
        return result.logits, result.cache

    def draft(self, anchor, cache, block_size, sampling=SamplingConfig(), generator=None, *,
              sampler=None, feedback=None):
        sampler = ProposalSampler(sampling) if sampler is None else sampler
        inputs = anchor.new_full((1, block_size), self.model.config.mask_token_id)
        inputs[:, :1] = anchor
        capture = None if feedback is None else feedback.capture_layer
        output = self.model(inputs, view="draft", cache=cache, capture_layer=capture)
        candidates, q, calibration = sampler.propose(output.logits[0, :-1], generator, protected_rows=0)
        return DraftBatch(torch.cat((anchor, candidates[None]), 1), candidates, q, cache, [], inputs, cache,
                          output.boundary, calibration, feedback is not None and feedback.learner is not None
                          and feedback.learner.needs_decoder_feedback)


class CausalLowRankBranch:
    """A clean anchor row produces a new exact root alongside noisy rows."""

    name = "causal_low_rank"
    initial_ar_token = True
    input_budget_extra = 0

    def __init__(self, model, executor=None, noise=None, *, initial_ar_token=True):
        self.model = model
        self.default_block_size = 8
        self.initial_ar_token = initial_ar_token
        if executor is not None:
            executor.validate(model)
        self.forward = model if executor is None else executor._forward
        self.noise = UniformNoise() if noise is None else noise

    def ar(self, tokens, cache=None, logits_to_keep=0):
        logits, updated = self.forward(tokens, cache=cache, return_cache=True)
        return logits[:, -logits_to_keep:] if logits_to_keep else logits, updated

    def draft(self, anchor, cache, block_size, sampling=SamplingConfig(), generator=None, *,
              sampler=None, feedback=None):
        sampler = ProposalSampler(sampling) if sampler is None else sampler
        noisy = self.noise.sample((1, block_size - 1), self.model.config.vocab_size,
                                  device=anchor.device, generator=generator)
        inputs = torch.cat((anchor, noisy), 1)
        route = torch.ones_like(inputs, dtype=torch.bool)
        route[:, 0] = False
        capture = None if feedback is None else feedback.capture_layer
        result = self.forward(inputs, cache=cache, adapter_mask=route, return_cache=True, capture_layer=capture)
        logits, temporary = result[:2]
        boundary = result[2] if capture is not None else None
        ids, q, calibration = sampler.propose(logits[0], generator)
        history = 0 if cache is None else cache[0][0].shape[-2]
        return DraftBatch(ids[None], ids[1:], None if q is None else q[1:],
                          crop_cache(temporary, history + 1), [int(ids[0])], inputs, cache,
                          boundary, calibration, feedback is not None and feedback.learner is not None
                          and feedback.learner.needs_decoder_feedback)

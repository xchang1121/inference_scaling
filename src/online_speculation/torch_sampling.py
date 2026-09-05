"""Torch reference implementation of Uno's filtered linear verifier.

The implementation intentionally keeps the proposal distribution object alive.
An online learner may update its parameters after verification, but the verifier
must still receive the distribution that actually sampled the proposal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingConfig:
    """One sampling configuration shared by draft and target distributions."""

    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 0.95

    def validate(self, vocab_size: int) -> None:
        if self.temperature <= 0:
            raise ValueError("Filtered stochastic sampling requires temperature > 0.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1].")
        if vocab_size < 2:
            raise ValueError("vocabulary must contain at least two tokens.")


@dataclass(frozen=True)
class FilteredDistribution:
    """Sparse categorical rows after temperature/top-k/top-p filtering."""

    token_ids: Tensor
    probabilities: Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2 or self.probabilities.ndim != 2:
            raise ValueError("token_ids and probabilities must both be rank two.")
        if self.token_ids.shape != self.probabilities.shape:
            raise ValueError("token_ids and probabilities must have identical shapes.")
        if self.token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long.")
        if bool(torch.any(self.probabilities < 0).item()):
            raise ValueError("probabilities must be non-negative.")
        row_sums = self.probabilities.sum(dim=-1)
        if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-6)):
            raise ValueError("each probability row must sum to one.")

    @property
    def rows(self) -> int:
        return int(self.token_ids.size(0))

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        offsets = torch.multinomial(
            self.probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        )
        return self.token_ids.gather(1, offsets).squeeze(1)

    def probability_of(self, tokens: Tensor) -> Tensor:
        tokens = tokens.reshape(-1).to(device=self.token_ids.device, dtype=torch.long)
        if tokens.numel() != self.rows:
            raise ValueError(f"expected {self.rows} tokens, received {tokens.numel()}.")
        matches = self.token_ids.eq(tokens.unsqueeze(1))
        return torch.where(
            matches,
            self.probabilities,
            torch.zeros((), device=self.probabilities.device),
        ).sum(dim=1)


def filtered_overlap(
    target: FilteredDistribution,
    draft: FilteredDistribution,
) -> Tensor:
    """Return per-row ``sum_x min(p(x), q(x)) = 1 - TV(p, q)``."""

    if target.rows != draft.rows:
        raise ValueError("target and draft must have the same number of rows.")
    if target.token_ids.device != draft.token_ids.device:
        raise ValueError("target and draft supports must share one device.")
    matches = target.token_ids.unsqueeze(2).eq(draft.token_ids.unsqueeze(1))
    q_on_target = torch.sum(
        matches * draft.probabilities.unsqueeze(1),
        dim=2,
    )
    return torch.minimum(target.probabilities, q_on_target).sum(dim=1)


def mixture_distribution(
    static: FilteredDistribution,
    candidate: FilteredDistribution,
    *,
    candidate_weight: float,
) -> FilteredDistribution:
    """Mix two sparse categorical distributions in probability space.

    Duplicate token ids are intentionally retained. ``sample`` treats them as
    equivalent categorical outcomes, while ``probability_of`` and overlap sum
    their masses. This avoids a variable-width per-row coalescing kernel.
    """

    if static.rows != candidate.rows:
        raise ValueError("mixture components must have the same number of rows.")
    if static.token_ids.device != candidate.token_ids.device:
        raise ValueError("mixture components must share one device.")
    if not math.isfinite(candidate_weight) or not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate_weight must lie in [0, 1].")
    if candidate_weight == 0.0:
        return static
    if candidate_weight == 1.0:
        return candidate
    return FilteredDistribution(
        token_ids=torch.cat((static.token_ids, candidate.token_ids), dim=1),
        probabilities=torch.cat(
            (
                static.probabilities * (1.0 - candidate_weight),
                candidate.probabilities * candidate_weight,
            ),
            dim=1,
        ),
    )


@dataclass(frozen=True)
class VerificationResult:
    """Tokens committed by one linear Psi-Spec verification cycle."""

    committed: tuple[int, ...]
    accepted_spec_tokens: int
    rejected_index: int | None
    used_lookahead: bool


def filtered_distribution(
    logits: Tensor, config: SamplingConfig
) -> FilteredDistribution:
    """Match Uno's top-k then top-p filtered categorical distribution."""

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [rows, vocab], got {tuple(logits.shape)}."
        )
    rows, vocab_size = logits.shape
    config.validate(int(vocab_size))

    top_k = vocab_size if config.top_k is None else min(config.top_k, vocab_size)
    values, token_ids = torch.topk(logits, k=top_k, dim=-1, sorted=True)
    probabilities = torch.softmax(values.float() / config.temperature, dim=-1)

    if config.top_p is not None and config.top_p < 1.0:
        cumulative = torch.cumsum(probabilities, dim=-1)
        # Retain the first token crossing top_p, exactly as the Uno runtime.
        probabilities = probabilities.masked_fill(
            (cumulative - probabilities) > config.top_p,
            0.0,
        )
        probabilities = probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

    if rows == 0:
        raise ValueError("at least one logits row is required.")
    return FilteredDistribution(token_ids=token_ids, probabilities=probabilities)


def residual_distribution(
    target: FilteredDistribution,
    draft: FilteredDistribution,
    row: int,
) -> FilteredDistribution:
    """Return normalized ``[p - q]_+`` on the target row's support."""

    if not 0 <= row < target.rows or not 0 <= row < draft.rows:
        raise IndexError("row is outside the target or draft distribution.")
    p_ids = target.token_ids[row]
    p_probs = target.probabilities[row]
    q_ids = draft.token_ids[row]
    q_probs = draft.probabilities[row]
    q_on_p = torch.where(
        p_ids.unsqueeze(1).eq(q_ids.unsqueeze(0)),
        q_probs.unsqueeze(0),
        torch.zeros((), device=q_probs.device),
    ).sum(dim=1)
    residual = torch.clamp(p_probs - q_on_p, min=0.0)
    mass = residual.sum()
    # In exact arithmetic this branch is unreachable after a rejection. It is
    # a numerical guard consistent with the public Uno implementation.
    residual = torch.where(
        mass > 0,
        residual / mass.clamp_min(1e-12),
        p_probs,
    )
    return FilteredDistribution(
        token_ids=p_ids.unsqueeze(0),
        probabilities=residual.unsqueeze(0),
    )


def verify_linear_filtered(
    *,
    free_token: int,
    spec_tokens: Tensor,
    target: FilteredDistribution,
    draft_used: FilteredDistribution,
    lookahead: FilteredDistribution,
    generator: torch.Generator | None = None,
    accept_uniforms: Tensor | None = None,
) -> VerificationResult:
    """Run exact filtered Psi-Spec using the *old* proposal distribution.

    ``draft_used`` must be the distribution that sampled ``spec_tokens``. The
    function is deliberately parameter-free so an online update cannot
    accidentally substitute a newly evaluated proposal distribution.
    """

    spec_tokens = spec_tokens.reshape(-1).to(target.token_ids.device)
    if target.rows != spec_tokens.numel() or draft_used.rows != spec_tokens.numel():
        raise ValueError("target/draft rows must equal the number of spec tokens.")
    if lookahead.rows != 1:
        raise ValueError("lookahead must contain exactly one row.")

    q_sample_prob = draft_used.probability_of(spec_tokens)
    if bool(torch.any(q_sample_prob <= 0).item()):
        raise ValueError("a proposal has zero probability under draft_used.")
    p_sample_prob = target.probability_of(spec_tokens)
    ratios = torch.clamp(p_sample_prob / q_sample_prob, max=1.0)

    if accept_uniforms is None:
        accept_uniforms = torch.rand(
            spec_tokens.numel(),
            device=ratios.device,
            dtype=ratios.dtype,
            generator=generator,
        )
    else:
        accept_uniforms = accept_uniforms.reshape(-1).to(ratios.device)
        if accept_uniforms.numel() != spec_tokens.numel():
            raise ValueError("accept_uniforms must match the number of spec tokens.")
        if bool(torch.any((accept_uniforms < 0) | (accept_uniforms >= 1)).item()):
            raise ValueError("accept_uniforms must lie in [0, 1).")

    committed = [int(free_token)]
    for row, token in enumerate(spec_tokens.tolist()):
        if bool((accept_uniforms[row] < ratios[row]).item()):
            committed.append(int(token))
            continue
        correction = residual_distribution(target, draft_used, row).sample(generator)
        committed.append(int(correction.item()))
        return VerificationResult(
            committed=tuple(committed),
            accepted_spec_tokens=row,
            rejected_index=row,
            used_lookahead=False,
        )

    committed.append(int(lookahead.sample(generator).item()))
    return VerificationResult(
        committed=tuple(committed),
        accepted_spec_tokens=spec_tokens.numel(),
        rejected_index=None,
        used_lookahead=True,
    )


def verify_linear_greedy(
    *,
    free_token: int,
    spec_tokens: Tensor,
    target_logits: Tensor,
    lookahead_logits: Tensor,
) -> VerificationResult:
    """Greedy linear verification, equivalent to delta-distribution Psi-Spec."""

    spec_tokens = spec_tokens.reshape(-1)
    if target_logits.ndim != 2 or target_logits.size(0) != spec_tokens.numel():
        raise ValueError("target logits rows must match spec tokens.")
    target_tokens = torch.argmax(target_logits, dim=-1)
    committed = [int(free_token)]
    for row, (proposal, target_token) in enumerate(
        zip(spec_tokens.tolist(), target_tokens.tolist())
    ):
        if int(proposal) == int(target_token):
            committed.append(int(proposal))
            continue
        committed.append(int(target_token))
        return VerificationResult(
            committed=tuple(committed),
            accepted_spec_tokens=row,
            rejected_index=row,
            used_lookahead=False,
        )
    committed.append(int(torch.argmax(lookahead_logits).item()))
    return VerificationResult(
        committed=tuple(committed),
        accepted_spec_tokens=spec_tokens.numel(),
        rejected_index=None,
        used_lookahead=True,
    )


def verify_replay_filtered(
    *,
    spec_tokens: Tensor,
    target: FilteredDistribution,
    lookahead: FilteredDistribution,
    generator: torch.Generator | None = None,
    accept_uniforms: Tensor | None = None,
) -> VerificationResult:
    """Verify deterministic cache proposals with exact Psi-Spec correction.

    The replay proposal at row ``i`` is the point mass
    ``q_i = delta(spec_tokens[i])``.  A proposal is therefore accepted with
    probability ``p_i(spec_tokens[i])``.  On rejection, ``residual_distribution``
    samples ``[p_i - q_i]_+`` and restores the target distribution exactly.

    Unlike :func:`verify_linear_filtered`, this function has no unverified
    ``free_token``.  Every returned token is produced by the one-pass AR replay
    verifier itself.
    """

    spec_tokens = spec_tokens.reshape(-1).to(target.token_ids.device, dtype=torch.long)
    if spec_tokens.numel() < 1:
        raise ValueError("at least one replay proposal is required.")
    if target.rows != spec_tokens.numel():
        raise ValueError("target rows must equal the number of replay proposals.")
    if lookahead.rows != 1:
        raise ValueError("lookahead must contain exactly one row.")

    draft_used = FilteredDistribution(
        token_ids=spec_tokens.unsqueeze(1),
        probabilities=torch.ones(
            (spec_tokens.numel(), 1),
            device=target.probabilities.device,
            dtype=target.probabilities.dtype,
        ),
    )
    p_sample_prob = target.probability_of(spec_tokens)
    if accept_uniforms is None:
        accept_uniforms = torch.rand(
            spec_tokens.numel(),
            device=p_sample_prob.device,
            dtype=p_sample_prob.dtype,
            generator=generator,
        )
    else:
        accept_uniforms = accept_uniforms.reshape(-1).to(p_sample_prob.device)
        if accept_uniforms.numel() != spec_tokens.numel():
            raise ValueError("accept_uniforms must match the number of replay proposals.")
        if bool(torch.any((accept_uniforms < 0) | (accept_uniforms >= 1)).item()):
            raise ValueError("accept_uniforms must lie in [0, 1).")

    committed: list[int] = []
    for row, token in enumerate(spec_tokens.tolist()):
        if bool((accept_uniforms[row] < p_sample_prob[row]).item()):
            committed.append(int(token))
            continue
        correction = residual_distribution(target, draft_used, row).sample(generator)
        committed.append(int(correction.item()))
        return VerificationResult(
            committed=tuple(committed),
            accepted_spec_tokens=row,
            rejected_index=row,
            used_lookahead=False,
        )

    committed.append(int(lookahead.sample(generator).item()))
    return VerificationResult(
        committed=tuple(committed),
        accepted_spec_tokens=spec_tokens.numel(),
        rejected_index=None,
        used_lookahead=True,
    )


def verify_replay_greedy(
    *,
    spec_tokens: Tensor,
    target_logits: Tensor,
    lookahead_logits: Tensor,
) -> VerificationResult:
    """Verify deterministic replay proposals under greedy target decoding."""

    spec_tokens = spec_tokens.reshape(-1)
    if spec_tokens.numel() < 1:
        raise ValueError("at least one replay proposal is required.")
    if target_logits.ndim != 2 or target_logits.size(0) != spec_tokens.numel():
        raise ValueError("target logits rows must match replay proposals.")
    if lookahead_logits.ndim not in (1, 2):
        raise ValueError("lookahead logits must be one vocabulary row.")
    if lookahead_logits.numel() != target_logits.size(1):
        raise ValueError("lookahead and target vocabulary sizes differ.")

    target_tokens = torch.argmax(target_logits, dim=-1)
    committed: list[int] = []
    for row, (proposal, target_token) in enumerate(
        zip(spec_tokens.tolist(), target_tokens.tolist())
    ):
        if int(proposal) == int(target_token):
            committed.append(int(proposal))
            continue
        committed.append(int(target_token))
        return VerificationResult(
            committed=tuple(committed),
            accepted_spec_tokens=row,
            rejected_index=row,
            used_lookahead=False,
        )
    committed.append(int(torch.argmax(lookahead_logits).item()))
    return VerificationResult(
        committed=tuple(committed),
        accepted_spec_tokens=spec_tokens.numel(),
        rejected_index=None,
        used_lookahead=True,
    )

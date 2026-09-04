"""Reference implementation of lossless linear Psi-Spec verification.

The implementation is deliberately small and eager.  It is the correctness oracle for
later GPU kernels, not a throughput implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
TargetDistribution = Callable[[tuple[int, ...]], ArrayLike]


def probability_vector(values: ArrayLike, *, name: str = "probabilities") -> FloatArray:
    """Validate and normalize one categorical probability vector."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with >=2 entries")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} cannot contain negative entries")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return vector / total


def probability_matrix(
    values: ArrayLike,
    *,
    name: str = "probabilities",
    expected_rows: int | None = None,
) -> FloatArray:
    """Validate and row-normalize a matrix of categorical distributions."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a two-dimensional categorical matrix")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {matrix.shape[0]} rows; expected {expected_rows}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} cannot contain negative entries")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError(f"every row of {name} must have positive mass")
    return matrix / totals


def sample_categorical(probabilities: ArrayLike, rng: np.random.Generator) -> int:
    """Sample one categorical value with deterministic boundary handling."""

    vector = probability_vector(probabilities)
    draw = float(rng.random())
    index = int(np.searchsorted(np.cumsum(vector), draw, side="right"))
    return min(index, vector.size - 1)


def total_variation(left: ArrayLike, right: ArrayLike) -> float:
    p = probability_vector(left, name="left")
    q = probability_vector(right, name="right")
    if p.shape != q.shape:
        raise ValueError("total-variation inputs must have the same shape")
    return 0.5 * float(np.abs(p - q).sum())


def expected_acceptance_probability(target: ArrayLike, draft: ArrayLike) -> float:
    """Return E_q[min(1, p(Y)/q(Y))] = 1 - TV(p, q)."""

    p = probability_vector(target, name="target")
    q = probability_vector(draft, name="draft")
    if p.shape != q.shape:
        raise ValueError("target and draft must have the same shape")
    return float(np.minimum(p, q).sum())


def residual_distribution(target: ArrayLike, draft: ArrayLike) -> FloatArray:
    """Return the normalized speculative-correction law [p-q]_+."""

    p = probability_vector(target, name="target")
    q = probability_vector(draft, name="draft")
    if p.shape != q.shape:
        raise ValueError("target and draft must have the same shape")
    residual = np.maximum(p - q, 0.0)
    mass = float(residual.sum())
    if mass <= 64.0 * np.finfo(np.float64).eps:
        raise ValueError("residual distribution has zero mass")
    return residual / mass


@dataclass(frozen=True)
class SpeculativeStepResult:
    """Outcome after verifying a block that excludes Uno's free AR token."""

    accepted_count: int
    committed_tokens: tuple[int, ...]
    rejection_index: int | None
    correction_token: int | None
    lookahead_token: int | None
    acceptance_uniforms: tuple[float, ...]

    @property
    def all_accepted(self) -> bool:
        return self.rejection_index is None


@dataclass(frozen=True)
class UnoLinearStepResult:
    """One Uno linear iteration: one exact AR token plus a verified draft block."""

    free_token: int
    proposal_tokens: tuple[int, ...]
    committed_tokens: tuple[int, ...]
    verification: SpeculativeStepResult
    target_probabilities: FloatArray
    draft_probabilities: FloatArray


def verify_speculative_block(
    proposal_tokens: Sequence[int],
    target_probabilities: ArrayLike,
    draft_probabilities: ArrayLike,
    rng: np.random.Generator,
) -> SpeculativeStepResult:
    """Verify a proposal block and perform exact residual correction.

    ``target_probabilities`` has one row per proposed token plus a final
    lookahead row. ``draft_probabilities`` must be the distributions that
    actually sampled ``proposal_tokens``; substituting updated distributions
    here is incorrect.
    """

    proposals = tuple(int(token) for token in proposal_tokens)
    draft = probability_matrix(
        draft_probabilities,
        name="draft_probabilities",
        expected_rows=len(proposals),
    )
    target = probability_matrix(
        target_probabilities,
        name="target_probabilities",
        expected_rows=len(proposals) + 1,
    )
    if target.shape[1] != draft.shape[1]:
        raise ValueError("target and draft vocabulary sizes differ")
    vocabulary_size = draft.shape[1]
    if any(token < 0 or token >= vocabulary_size for token in proposals):
        raise ValueError("proposal token is outside the vocabulary")

    uniforms: list[float] = []
    for index, token in enumerate(proposals):
        proposal_mass = float(draft[index, token])
        if proposal_mass <= 0.0:
            raise ValueError("a supplied proposal has zero probability under its draft")
        acceptance_probability = min(
            1.0,
            float(target[index, token]) / proposal_mass,
        )
        uniform = float(rng.random())
        uniforms.append(uniform)
        if uniform < acceptance_probability:
            continue
        correction = sample_categorical(
            residual_distribution(target[index], draft[index]),
            rng,
        )
        return SpeculativeStepResult(
            accepted_count=index,
            committed_tokens=proposals[:index] + (correction,),
            rejection_index=index,
            correction_token=correction,
            lookahead_token=None,
            acceptance_uniforms=tuple(uniforms),
        )

    lookahead = sample_categorical(target[-1], rng)
    return SpeculativeStepResult(
        accepted_count=len(proposals),
        committed_tokens=proposals + (lookahead,),
        rejection_index=None,
        correction_token=None,
        lookahead_token=lookahead,
        acceptance_uniforms=tuple(uniforms),
    )


def uno_linear_step(
    history: Sequence[int],
    target_distribution: TargetDistribution,
    draft_probabilities: ArrayLike,
    rng: np.random.Generator,
) -> UnoLinearStepResult:
    """Run the reference single-sequence Uno linear algorithm.

    The draft matrix is fixed before the free token is sampled, matching the
    one-forward parallel proposal in Uno. The target callback is then queried
    along the hypothetical proposal prefix, exactly as an AR verification
    forward supplies its rows.
    """

    prefix = tuple(int(token) for token in history)
    draft = probability_matrix(draft_probabilities, name="draft_probabilities")
    free_token = sample_categorical(target_distribution(prefix), rng)
    proposals = tuple(sample_categorical(row, rng) for row in draft)

    verifier_history = prefix + (free_token,)
    target_rows: list[FloatArray] = []
    for proposal in proposals:
        target_rows.append(
            probability_vector(
                target_distribution(verifier_history),
                name="target_distribution",
            )
        )
        verifier_history += (proposal,)
    target_rows.append(
        probability_vector(
            target_distribution(verifier_history),
            name="target_distribution",
        )
    )
    target = np.stack(target_rows)
    verification = verify_speculative_block(proposals, target, draft, rng)
    return UnoLinearStepResult(
        free_token=free_token,
        proposal_tokens=proposals,
        committed_tokens=(free_token,) + verification.committed_tokens,
        verification=verification,
        target_probabilities=target,
        draft_probabilities=draft.copy(),
    )


def one_token_output_distribution(
    target: ArrayLike,
    sampling_draft: ArrayLike,
    denominator_draft: ArrayLike | None = None,
) -> FloatArray:
    """Enumerate the output law of one speculative proposal exactly.

    ``denominator_draft`` exists solely for a negative-control audit. Correct
    decoding leaves it unset, so the proposal and acceptance denominator use
    the same distribution.
    """

    p = probability_vector(target, name="target")
    sampled_from = probability_vector(sampling_draft, name="sampling_draft")
    denominator = probability_vector(
        sampling_draft if denominator_draft is None else denominator_draft,
        name="denominator_draft",
    )
    if not (p.shape == sampled_from.shape == denominator.shape):
        raise ValueError("all distributions must have the same shape")

    output = np.zeros_like(p)
    residual: FloatArray | None = None
    for proposal, sampling_mass in enumerate(sampled_from):
        denominator_mass = float(denominator[proposal])
        acceptance = (
            min(1.0, float(p[proposal]) / denominator_mass)
            if denominator_mass > 0.0
            else 0.0
        )
        output[proposal] += float(sampling_mass) * acceptance
        rejection_mass = float(sampling_mass) * (1.0 - acceptance)
        if rejection_mass <= 0.0:
            continue
        if residual is None:
            residual = residual_distribution(p, denominator)
        output += rejection_mass * residual
    return probability_vector(output, name="output")

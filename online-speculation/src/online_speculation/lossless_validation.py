"""Monte Carlo validation utilities for the reference Psi-Spec sampler."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .psi_spec import (
    FloatArray,
    one_token_output_distribution,
    probability_matrix,
    probability_vector,
    total_variation,
    uno_linear_step,
)


@dataclass
class ToyDrafter:
    """A deliberately misspecified proposer with optional post-round updates."""

    vocabulary_size: int
    adaptive: bool
    learning_rate: float = 0.45

    def __post_init__(self) -> None:
        initial = np.linspace(1.0, 0.15, self.vocabulary_size, dtype=np.float64)
        self.state = probability_vector(initial)
        self.update_count = 0

    def propose(self, history: tuple[int, ...], count: int) -> FloatArray:
        rows = []
        phase = (sum(history) + len(history)) % self.vocabulary_size
        for position in range(count):
            rotated = np.roll(self.state, phase + position)
            floor = np.full(self.vocabulary_size, 1.0 / self.vocabulary_size)
            rows.append(0.88 * rotated + 0.12 * floor)
        return probability_matrix(rows)

    def update(self, target_rows: FloatArray, observed_rows: int) -> None:
        if not self.adaptive or observed_rows <= 0:
            return
        target = probability_matrix(target_rows[:observed_rows]).mean(axis=0)
        self.state = probability_vector(
            (1.0 - self.learning_rate) * self.state
            + self.learning_rate * target
        )
        self.update_count += 1


def toy_target(history: tuple[int, ...], vocabulary_size: int) -> FloatArray:
    """A stable, nontrivial AR target with all tokens in support."""

    weighted_sum = sum((index + 1) * (token + 1) for index, token in enumerate(history))
    phase = weighted_sum + 3 * len(history)
    token_ids = np.arange(vocabulary_size, dtype=np.float64)
    logits = 0.45 * np.cos((token_ids + 1.0) * (len(history) + 1.0) + 0.17 * phase)
    preferred = phase % vocabulary_size
    logits[preferred] += 1.05
    logits[(preferred + 1) % vocabulary_size] += 0.2
    logits -= logits.max()
    return probability_vector(np.exp(logits))


def enumerate_target_sequences(
    prompt: tuple[int, ...],
    sequence_length: int,
    vocabulary_size: int,
) -> dict[tuple[int, ...], float]:
    probabilities: dict[tuple[int, ...], float] = {}
    for completion in itertools.product(range(vocabulary_size), repeat=sequence_length):
        history = prompt
        probability = 1.0
        for token in completion:
            distribution = toy_target(history, vocabulary_size)
            probability *= float(distribution[token])
            history += (token,)
        probabilities[completion] = probability
    total = sum(probabilities.values())
    return {sequence: probability / total for sequence, probability in probabilities.items()}


def simulate_completion(
    rng: np.random.Generator,
    *,
    prompt: tuple[int, ...],
    sequence_length: int,
    vocabulary_size: int,
    block_size: int,
    adaptive: bool,
) -> tuple[tuple[int, ...], int, int, int]:
    history = prompt
    output: list[int] = []
    drafter = ToyDrafter(vocabulary_size, adaptive)
    accepted = 0
    rounds = 0
    while len(output) < sequence_length:
        speculative_count = max(1, block_size - 1)
        draft = drafter.propose(history, speculative_count)
        result = uno_linear_step(
            history,
            lambda current: toy_target(current, vocabulary_size),
            draft,
            rng,
        )
        accepted += result.verification.accepted_count
        observed_rows = (
            result.verification.rejection_index + 1
            if result.verification.rejection_index is not None
            else speculative_count
        )
        # The update is intentionally after verification; only the next round sees it.
        drafter.update(result.target_probabilities, observed_rows)
        remaining = sequence_length - len(output)
        committed = result.committed_tokens[:remaining]
        output.extend(committed)
        history += committed
        rounds += 1
    return tuple(output), accepted, rounds, drafter.update_count


def empirical_metrics(
    exact: dict[tuple[int, ...], float],
    counts: Counter[tuple[int, ...]],
    samples: int,
) -> dict[str, float | int | bool]:
    support = sorted(exact)
    expected = np.asarray([exact[key] for key in support], dtype=np.float64)
    empirical = np.asarray([counts[key] / samples for key in support], dtype=np.float64)
    errors = empirical - expected
    variances = expected * (1.0 - expected) / samples
    z_scores = np.abs(errors) / np.sqrt(np.maximum(variances, np.finfo(float).tiny))
    expected_counts = expected * samples
    pearson = float(
        np.sum((np.asarray([counts[key] for key in support]) - expected_counts) ** 2 / expected_counts)
    )
    degrees_of_freedom = max(len(support) - 1, 1)
    tv = 0.5 * float(np.abs(errors).sum())
    tv_limit = max(0.02, 3.0 * 0.5 * math.sqrt((len(support) - 1) / samples))
    pearson_per_dof = pearson / degrees_of_freedom
    passed = tv <= tv_limit and float(z_scores.max()) <= 6.0 and pearson_per_dof <= 2.0
    return {
        "support_size": len(support),
        "total_variation": tv,
        "total_variation_limit": tv_limit,
        "max_absolute_error": float(np.abs(errors).max()),
        "max_standardized_error": float(z_scores.max()),
        "pearson_chi_square_per_dof": pearson_per_dof,
        "passed": passed,
    }


def run_mode(
    *,
    samples: int,
    seed: int,
    prompt: tuple[int, ...],
    sequence_length: int,
    vocabulary_size: int,
    block_size: int,
    adaptive: bool,
    exact: dict[tuple[int, ...], float],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    counts: Counter[tuple[int, ...]] = Counter()
    accepted = 0
    rounds = 0
    updates = 0
    committed_tokens = 0
    for _ in range(samples):
        completion, sample_accepted, sample_rounds, sample_updates = simulate_completion(
            rng,
            prompt=prompt,
            sequence_length=sequence_length,
            vocabulary_size=vocabulary_size,
            block_size=block_size,
            adaptive=adaptive,
        )
        counts[completion] += 1
        accepted += sample_accepted
        rounds += sample_rounds
        updates += sample_updates
        committed_tokens += len(completion)
    metrics = empirical_metrics(exact, counts, samples)
    metrics.update(
        {
            "adaptive": adaptive,
            "samples": samples,
            "seed": seed,
            "mean_accepted_speculative_tokens_per_round": accepted / rounds,
            "mean_committed_tokens_per_round": committed_tokens / rounds,
            "toy_tokens_per_forward": committed_tokens / (2.0 * rounds),
            "mean_updates_per_completion": updates / samples,
        }
    )
    return metrics


def run_validation(
    *,
    samples: int,
    seed: int,
    sequence_length: int,
    vocabulary_size: int,
    block_size: int,
) -> dict[str, object]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least two")
    if block_size < 2:
        raise ValueError("block_size must be at least two")

    prompt = (0,)
    exact = enumerate_target_sequences(prompt, sequence_length, vocabulary_size)
    static = run_mode(
        samples=samples,
        seed=seed,
        prompt=prompt,
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        block_size=block_size,
        adaptive=False,
        exact=exact,
    )
    adaptive = run_mode(
        samples=samples,
        seed=seed + 1,
        prompt=prompt,
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        block_size=block_size,
        adaptive=True,
        exact=exact,
    )

    target = np.asarray([0.65, 0.25, 0.10], dtype=np.float64)
    old_draft = np.asarray([0.05, 0.15, 0.80], dtype=np.float64)
    correct = one_token_output_distribution(target, old_draft)
    wrong = one_token_output_distribution(target, old_draft, target)
    correct_error = total_variation(target, correct)
    wrong_error = total_variation(target, wrong)
    negative_control = {
        "target": target.tolist(),
        "sampling_draft_q_t": old_draft.tolist(),
        "incorrect_denominator_q_t_plus_1": target.tolist(),
        "correct_output": correct.tolist(),
        "incorrect_output": wrong.tolist(),
        "correct_total_variation": correct_error,
        "incorrect_total_variation": wrong_error,
        "passed": correct_error < 1e-12 and wrong_error > 0.25,
    }
    passed = bool(static["passed"] and adaptive["passed"] and negative_control["passed"])
    return {
        "schema_version": 1,
        "experiment": "stage1_lossless_psi_spec_validation",
        "parameters": {
            "samples_per_mode": samples,
            "seed": seed,
            "prompt": list(prompt),
            "sequence_length": sequence_length,
            "vocabulary_size": vocabulary_size,
            "block_size": block_size,
        },
        "exact_target_mass": sum(exact.values()),
        "static_draft": static,
        "post_round_adaptive_draft": adaptive,
        "old_q_negative_control": negative_control,
        "passed": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocabulary-size", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation(
        samples=args.samples,
        seed=args.seed,
        sequence_length=args.sequence_length,
        vocabulary_size=args.vocabulary_size,
        block_size=args.block_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

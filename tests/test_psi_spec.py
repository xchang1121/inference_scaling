from __future__ import annotations

import numpy as np
import pytest

from online_speculation.lossless_validation import run_validation
from online_speculation.psi_spec import (
    expected_acceptance_probability,
    one_token_output_distribution,
    residual_distribution,
    total_variation,
    uno_linear_step,
    verify_speculative_block,
)


def test_expected_acceptance_is_one_minus_total_variation() -> None:
    target = [0.55, 0.30, 0.15]
    draft = [0.10, 0.60, 0.30]

    assert expected_acceptance_probability(target, draft) == pytest.approx(
        1.0 - total_variation(target, draft)
    )


def test_residual_distribution_uses_positive_target_minus_draft() -> None:
    residual = residual_distribution([0.6, 0.3, 0.1], [0.2, 0.5, 0.3])

    assert residual == pytest.approx([1.0, 0.0, 0.0])


def test_one_token_output_is_exact_for_old_draft() -> None:
    target = np.asarray([0.61, 0.27, 0.12])
    draft = np.asarray([0.08, 0.17, 0.75])

    output = one_token_output_distribution(target, draft)

    assert output == pytest.approx(target, abs=1e-15)


def test_using_updated_denominator_is_detectably_wrong() -> None:
    target = np.asarray([0.65, 0.25, 0.10])
    old_draft = np.asarray([0.05, 0.15, 0.80])

    wrong = one_token_output_distribution(target, old_draft, target)

    assert wrong == pytest.approx(old_draft)
    assert total_variation(target, wrong) > 0.5


def test_all_accepted_block_commits_lookahead() -> None:
    proposals = [0, 1]
    draft = [[1.0, 0.0], [0.0, 1.0]]
    target = [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]]

    result = verify_speculative_block(
        proposals,
        target,
        draft,
        np.random.default_rng(7),
    )

    assert result.all_accepted
    assert result.accepted_count == 2
    assert result.committed_tokens[:2] == (0, 1)
    assert result.committed_tokens[-1] == result.lookahead_token


def test_rejected_block_commits_residual_correction() -> None:
    proposals = [1, 1]
    draft = [[0.0, 1.0], [0.0, 1.0]]
    target = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

    result = verify_speculative_block(
        proposals,
        target,
        draft,
        np.random.default_rng(11),
    )

    assert result.rejection_index == 0
    assert result.accepted_count == 0
    assert result.correction_token == 0
    assert result.committed_tokens == (0,)


def test_uno_step_always_commits_free_ar_token() -> None:
    target = lambda history: [1.0, 0.0] if len(history) % 2 == 0 else [0.0, 1.0]
    result = uno_linear_step(
        (0,),
        target,
        [[1.0, 0.0], [1.0, 0.0]],
        np.random.default_rng(3),
    )

    assert result.free_token == 1
    assert result.committed_tokens[0] == 1
    assert len(result.committed_tokens) >= 2


def test_small_monte_carlo_static_and_online_modes_are_lossless() -> None:
    result = run_validation(
        samples=5_000,
        seed=20260905,
        sequence_length=3,
        vocabulary_size=3,
        block_size=4,
    )

    assert result["passed"]

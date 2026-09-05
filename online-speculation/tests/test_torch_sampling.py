from __future__ import annotations

import torch

from online_speculation.torch_sampling import (
    FilteredDistribution,
    SamplingConfig,
    filtered_distribution,
    filtered_overlap,
    mixture_distribution,
    residual_distribution,
    verify_linear_filtered,
    verify_linear_greedy,
    verify_replay_filtered,
    verify_replay_greedy,
)
from online_speculation.hf_uno import _adapter_parameter_name


def _distribution(rows: list[list[float]]) -> FilteredDistribution:
    probabilities = torch.tensor(rows, dtype=torch.float32)
    token_ids = torch.arange(probabilities.size(1)).repeat(probabilities.size(0), 1)
    return FilteredDistribution(token_ids=token_ids, probabilities=probabilities)


def test_filtered_distribution_matches_top_k_then_top_p() -> None:
    logits = torch.log(torch.tensor([[0.40, 0.30, 0.20, 0.10]]))
    result = filtered_distribution(
        logits,
        SamplingConfig(temperature=1.0, top_k=3, top_p=0.65),
    )
    assert result.token_ids.tolist() == [[0, 1, 2]]
    assert torch.allclose(
        result.probabilities,
        torch.tensor([[4 / 7, 3 / 7, 0.0]]),
        atol=1e-6,
    )


def test_residual_distribution_is_positive_part_of_p_minus_q() -> None:
    target = _distribution([[0.7, 0.2, 0.1]])
    draft = _distribution([[0.1, 0.2, 0.7]])
    residual = residual_distribution(target, draft, 0)
    assert torch.allclose(residual.probabilities, torch.tensor([[1.0, 0.0, 0.0]]))


def test_filtered_overlap_handles_different_sparse_supports() -> None:
    target = FilteredDistribution(
        token_ids=torch.tensor([[1, 3, 5]]),
        probabilities=torch.tensor([[0.5, 0.3, 0.2]]),
    )
    draft = FilteredDistribution(
        token_ids=torch.tensor([[0, 1, 5]]),
        probabilities=torch.tensor([[0.4, 0.4, 0.2]]),
    )
    torch.testing.assert_close(filtered_overlap(target, draft), torch.tensor([0.6]))


def test_probability_mixture_preserves_union_mass_and_tv_convexity() -> None:
    static = FilteredDistribution(
        token_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        probabilities=torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]),
    )
    candidate = FilteredDistribution(
        token_ids=torch.tensor([[1, 2], [1, 2], [1, 2]]),
        probabilities=torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
    )
    mixed = mixture_distribution(static, candidate, candidate_weight=0.25)
    assert torch.allclose(
        mixed.probability_of(torch.tensor([0, 1, 2])),
        torch.tensor([0.6, 0.275, 0.125]),
    )

    target = FilteredDistribution(
        token_ids=torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        probabilities=torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
    )
    mixed_tv = 1.0 - filtered_overlap(target, mixed)
    convex_bound = 0.75 * (1.0 - filtered_overlap(target, static)) + 0.25 * (
        1.0 - filtered_overlap(target, candidate)
    )
    assert torch.all(mixed_tv <= convex_bound + 1e-7)


def test_filtered_verifier_accepts_all_and_adds_lookahead() -> None:
    draft = _distribution([[0.6, 0.4], [0.3, 0.7]])
    target = _distribution([[0.6, 0.4], [0.3, 0.7]])
    lookahead = _distribution([[0.0, 1.0]])
    result = verify_linear_filtered(
        free_token=1,
        spec_tokens=torch.tensor([0, 1]),
        target=target,
        draft_used=draft,
        lookahead=lookahead,
        accept_uniforms=torch.tensor([0.99, 0.99]),
    )
    assert result.committed == (1, 0, 1, 1)
    assert result.accepted_spec_tokens == 2
    assert result.used_lookahead


def test_filtered_verifier_rejects_with_residual_correction() -> None:
    draft = _distribution([[0.1, 0.9]])
    target = _distribution([[0.8, 0.2]])
    lookahead = _distribution([[1.0, 0.0]])
    result = verify_linear_filtered(
        free_token=0,
        spec_tokens=torch.tensor([1]),
        target=target,
        draft_used=draft,
        lookahead=lookahead,
        accept_uniforms=torch.tensor([0.5]),
    )
    assert result.committed == (0, 0)
    assert result.accepted_spec_tokens == 0
    assert result.rejected_index == 0
    assert not result.used_lookahead


def test_greedy_verifier_commits_target_at_first_mismatch() -> None:
    result = verify_linear_greedy(
        free_token=2,
        spec_tokens=torch.tensor([1, 0, 2]),
        target_logits=torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        lookahead_logits=torch.tensor([0.0, 0.0, 1.0]),
    )
    assert result.committed == (2, 1, 2)
    assert result.accepted_spec_tokens == 1
    assert result.rejected_index == 1


def test_replay_filtered_accepts_prefix_then_uses_exact_delta_residual() -> None:
    target = _distribution([[0.8, 0.2], [0.3, 0.7]])
    lookahead = _distribution([[0.0, 1.0]])
    result = verify_replay_filtered(
        spec_tokens=torch.tensor([0, 0]),
        target=target,
        lookahead=lookahead,
        accept_uniforms=torch.tensor([0.5, 0.5]),
    )
    assert result.committed == (0, 1)
    assert result.accepted_spec_tokens == 1
    assert result.rejected_index == 1
    assert not result.used_lookahead


def test_replay_filtered_all_accept_adds_target_lookahead_without_free_token() -> None:
    target = _distribution([[1.0, 0.0], [0.0, 1.0]])
    lookahead = _distribution([[1.0, 0.0]])
    result = verify_replay_filtered(
        spec_tokens=torch.tensor([0, 1]),
        target=target,
        lookahead=lookahead,
        accept_uniforms=torch.tensor([0.999, 0.999]),
    )
    assert result.committed == (0, 1, 0)
    assert result.accepted_spec_tokens == 2
    assert result.used_lookahead


def test_replay_greedy_commits_first_target_mismatch_or_lookahead() -> None:
    rejected = verify_replay_greedy(
        spec_tokens=torch.tensor([1, 0, 2]),
        target_logits=torch.tensor(
            [[0.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
        ),
        lookahead_logits=torch.tensor([0.0, 0.0, 1.0]),
    )
    assert rejected.committed == (1, 2)
    assert rejected.accepted_spec_tokens == 1
    assert rejected.rejected_index == 1

    accepted = verify_replay_greedy(
        spec_tokens=torch.tensor([1, 2]),
        target_logits=torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]]),
        lookahead_logits=torch.tensor([3.0, 0.0, 1.0]),
    )
    assert accepted.committed == (1, 2, 0)
    assert accepted.accepted_spec_tokens == 2
    assert accepted.used_lookahead


def test_verifier_rejects_distribution_that_did_not_sample_proposal() -> None:
    target = _distribution([[0.5, 0.5]])
    impossible_old_draft = _distribution([[1.0, 0.0]])
    lookahead = _distribution([[0.5, 0.5]])
    try:
        verify_linear_filtered(
            free_token=0,
            spec_tokens=torch.tensor([1]),
            target=target,
            draft_used=impossible_old_draft,
            lookahead=lookahead,
        )
    except ValueError as error:
        assert "zero probability" in str(error)
    else:
        raise AssertionError("verifier accepted a proposal impossible under draft_used")


def test_public_adapter_key_maps_to_nested_peft_parameter() -> None:
    assert _adapter_parameter_name("model.layers.7.self_attn.q_proj.lora_A.weight") == (
        "base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weight"
    )

from __future__ import annotations

import torch
from torch import nn

from online_speculation.fast_residual import (
    FastResidualConfig,
    FastResidualHead,
    FastResidualLearner,
    assert_optimizer_isolated,
    feedback_from_logits,
)


def _head(*, learning_rate: float = 0.05) -> FastResidualLearner:
    torch.manual_seed(3)
    head = FastResidualHead(
        hidden_size=4,
        vocabulary_size=6,
        rank=2,
        alpha=2.0,
    )
    return FastResidualLearner(
        head,
        FastResidualConfig(
            rank=2,
            alpha=2.0,
            learning_rate=learning_rate,
            validation_stride=2,
            reset_margin=0.05,
            rollback_tolerance=0.0,
        ),
    )


def _feedback(
    *,
    hidden: torch.Tensor | None = None,
    target_token: int = 1,
    position: int = 0,
):
    hidden = torch.tensor([1.0, -0.5, 0.25, 0.75]) if hidden is None else hidden
    base = torch.tensor([2.0, 0.2, -0.5, -1.0, 0.1, -0.2])
    target = torch.full((6,), -2.0)
    target[target_token] = 3.0
    return feedback_from_logits(
        hidden=hidden,
        base_logits=base,
        adjusted_logits=base,
        target_logits=target,
        top_k=3,
        temperature=1.0,
        position=position,
    )


def test_zero_initialized_head_leaves_logits_unchanged() -> None:
    learner = _head()
    hidden = torch.randn(3, 4)
    base = torch.randn(3, 6)
    assert torch.count_nonzero(learner.head(hidden)).item() == 0
    torch.testing.assert_close(learner.corrected_logits(hidden, base), base)


def test_learner_rejects_head_configuration_mismatch() -> None:
    head = FastResidualHead(
        hidden_size=4,
        vocabulary_size=6,
        rank=2,
        alpha=2.0,
    )
    try:
        FastResidualLearner(head, FastResidualConfig(rank=3, alpha=3.0))
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("mismatched head configuration was accepted")


def test_feedback_uses_draft_target_union_and_detaches_old_distribution() -> None:
    hidden = torch.arange(4, dtype=torch.float32, requires_grad=True)
    base = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    target = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    item = feedback_from_logits(
        hidden=hidden,
        base_logits=base,
        adjusted_logits=base,
        target_logits=target,
        top_k=2,
        temperature=1.0,
    )
    assert set(item.token_ids.tolist()) == {0, 1, 4, 5}
    assert not item.hidden.requires_grad
    assert not item.old_probabilities.requires_grad
    torch.testing.assert_close(item.target_probabilities.sum(), torch.tensor(1.0))
    torch.testing.assert_close(item.old_probabilities.sum(), torch.tensor(1.0))


def test_transactional_update_reduces_held_out_distillation_loss() -> None:
    learner = _head(learning_rate=0.05)
    items = [_feedback(position=index) for index in range(8)]
    validation = items[::2]
    before = learner.evaluate(validation).objective
    report = learner.update(items)
    after = learner.evaluate(validation).objective
    assert not report.reset_to_offline
    assert not report.rolled_back
    assert after < before
    assert report.fast_weight_l2 > 0


def test_validation_regression_rolls_back_head_and_optimizer() -> None:
    learner = _head(learning_rate=1.0)
    # Even positions are validation examples preferring token 1. Odd positions
    # are training examples with the same hidden state but prefer token 5.
    items = [
        _feedback(target_token=1 if index % 2 == 0 else 5, position=index)
        for index in range(8)
    ]
    state_before = {
        name: value.detach().clone() for name, value in learner.head.state_dict().items()
    }
    report = learner.update(items)
    assert report.rolled_back
    for name, value in learner.head.state_dict().items():
        torch.testing.assert_close(value, state_before[name])


def test_static_shadow_resets_stale_fast_weights_before_update() -> None:
    learner = _head(learning_rate=0.01)
    items = [_feedback(target_token=0, position=index) for index in range(6)]
    with torch.no_grad():
        learner.head.up.weight.fill_(2.0)
        learner.head.up.weight[0].fill_(-2.0)
    assert learner.evaluate(items).objective > learner.evaluate(
        items,
        zero_correction=True,
    ).objective
    report = learner.update(items)
    assert report.reset_to_offline


def test_decay_moves_fast_weights_toward_offline_snapshot() -> None:
    learner = _head()
    with torch.no_grad():
        learner.head.up.weight.fill_(2.0)
    before = learner.fast_weight_l2()
    learner.decay_toward_offline(0.25)
    assert torch.isclose(torch.tensor(learner.fast_weight_l2()), torch.tensor(before * 0.25))


def test_optimizer_isolation_fails_closed_if_base_is_trainable() -> None:
    learner = _head()
    base = nn.Linear(4, 4)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    report = assert_optimizer_isolated(
        base_model=base,
        head=learner.head,
        optimizer=learner.optimizer,
    )
    assert report["base_optimizer_overlap"] == 0
    assert report["fast_trainable_parameters"] == 20

    next(base.parameters()).requires_grad_(True)
    try:
        assert_optimizer_isolated(
            base_model=base,
            head=learner.head,
            optimizer=learner.optimizer,
        )
    except RuntimeError as error:
        assert "trainable" in str(error)
    else:
        raise AssertionError("trainable base parameters were not rejected")

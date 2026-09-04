from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from online_speculation.fast_residual import FastResidualConfig
from online_speculation.hf_online_uno import (
    HfOnlineUnoRunner,
    OnlineRuntimeConfig,
    choose_deferred_action,
    feedback_weights,
    summarize_online_runs,
)
from online_speculation.hf_uno import HfUnoRuntime
from online_speculation.torch_sampling import SamplingConfig


class _Cache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        if length < 0:
            self.length = max(0, self.length + length)
        else:
            self.length = min(self.length, length)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=6, hidden_size=4)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: _Cache | None = None,
        use_cache: bool,
        output_hidden_states: bool = False,
    ):
        del use_cache, output_hidden_states
        cache = _Cache() if past_key_values is None else past_key_values
        cache.length += input_ids.size(1)
        next_ids = (input_ids + 1) % self.config.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -2.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 3.0)
        hidden = torch.stack(
            (
                input_ids.float() / 6.0,
                torch.ones_like(input_ids, dtype=torch.float32),
                (input_ids % 2).float(),
                (input_ids % 3).float() / 3.0,
            ),
            dim=-1,
        )
        return SimpleNamespace(
            logits=logits + self.anchor * 0.0,
            past_key_values=cache,
            hidden_states=(hidden,),
        )

    @contextmanager
    def disable_adapter(self):
        try:
            yield
        finally:
            # Reproduce PEFT restoring a trainable adapter after its context.
            self.anchor.requires_grad_(True)


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(str(token) for token in token_ids)


class _Router:
    def __init__(self) -> None:
        self.mask = None

    def set_token_mask(self, mask: torch.Tensor) -> None:
        self.mask = mask.clone()


def _runtime() -> HfUnoRuntime:
    model = _TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return HfUnoRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        router=_Router(),
        device=torch.device("cpu"),
        sampling=SamplingConfig(temperature=1.0, top_k=3, top_p=0.95),
        mask_token_id=6,
        stop_token_ids=[],
        ignore_stop=True,
    )


def test_feedback_weights_cover_full_on_policy_and_discounted_modes() -> None:
    assert feedback_weights(
        speculative_tokens=4,
        rejected_index=1,
        supervision="full",
        tail_discount=0.25,
        position_discount=1.0,
    ) == [1.0, 1.0, 1.0, 1.0]
    assert feedback_weights(
        speculative_tokens=4,
        rejected_index=1,
        supervision="on_policy",
        tail_discount=0.25,
        position_discount=1.0,
    ) == [1.0, 1.0, 0.0, 0.0]
    assert feedback_weights(
        speculative_tokens=4,
        rejected_index=1,
        supervision="discounted_tail",
        tail_discount=0.25,
        position_discount=1.0,
    ) == [1.0, 1.0, 0.25, 0.0625]


def test_deferred_action_requires_future_margin_and_prefers_best_shadow() -> None:
    assert (
        choose_deferred_action(
            active_tv=0.30,
            candidate_tv=0.25,
            static_tv=0.28,
            promotion_margin=0.01,
            reset_margin=0.01,
        )
        == "promote_candidate"
    )
    assert (
        choose_deferred_action(
            active_tv=0.30,
            candidate_tv=0.29,
            static_tv=0.20,
            promotion_margin=0.01,
            reset_margin=0.01,
        )
        == "reset_to_static"
    )
    assert (
        choose_deferred_action(
            active_tv=0.30,
            candidate_tv=0.295,
            static_tv=0.30,
            promotion_margin=0.01,
            reset_margin=0.01,
        )
        == "keep_active"
    )


def test_online_runner_executes_real_cache_loop_and_updates_only_fast_head() -> None:
    runtime = _runtime()
    result = HfOnlineUnoRunner(runtime).generate(
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        max_new_tokens=20,
        seed=11,
        initialization_seed=19,
        config=OnlineRuntimeConfig(
            block_size=4,
            update_stride=2,
            feedback_top_k=3,
            supervision="on_policy",
            fast=FastResidualConfig(
                rank=2,
                alpha=2.0,
                learning_rate=0.01,
                validation_stride=2,
            ),
        ),
    )
    assert result.metrics.output_tokens == 20
    assert result.metrics.decoder_forwards == 2 * result.metrics.cycles
    assert result.metrics.method == "uno_online_fast_residual_hf_fallback"
    assert result.diagnostics.feedback_items_created > 0
    assert result.diagnostics.update_attempts > 0
    assert result.diagnostics.parameter_isolation["base_optimizer_overlap"] == 0
    assert result.diagnostics.parameter_isolation["fast_trainable_parameters"] == 20
    assert not next(runtime.model.parameters()).requires_grad


def test_base_context_refreezes_parameters_after_adapter_restore() -> None:
    runtime = _runtime()
    with runtime._base_context():
        assert not next(runtime.model.parameters()).requires_grad
    assert not next(runtime.model.parameters()).requires_grad


def test_deferred_runner_validates_candidate_on_future_verifier_rows() -> None:
    runtime = _runtime()
    result = HfOnlineUnoRunner(runtime).generate(
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        max_new_tokens=50,
        seed=13,
        initialization_seed=23,
        config=OnlineRuntimeConfig(
            block_size=4,
            update_stride=2,
            feedback_top_k=3,
            supervision="on_policy",
            activation_mode="deferred",
            feedback_interval=1,
            candidate_evaluation_interval=2,
            promotion_margin=0.0,
            future_reset_margin=0.0,
            fast=FastResidualConfig(
                rank=2,
                alpha=2.0,
                learning_rate=0.01,
                validation_stride=2,
            ),
        ),
    )
    diagnostics = result.diagnostics
    assert result.metrics.output_tokens == 50
    assert result.metrics.method == "uno_deferred_fast_residual_hf_fallback"
    assert diagnostics.activation_mode == "deferred"
    assert 0 < diagnostics.candidate_evaluation_cycles < result.metrics.cycles
    assert (
        diagnostics.active_head_evaluation_cycles + diagnostics.static_head_skip_cycles
        == result.metrics.cycles
    )
    assert diagnostics.static_head_skip_cycles > 0
    assert diagnostics.candidate_promotion_attempts > 0
    assert len(diagnostics.promotion_events) == diagnostics.candidate_promotion_attempts
    assert (
        diagnostics.candidate_promotions
        + diagnostics.candidate_rejections
        + diagnostics.future_static_resets
        == diagnostics.candidate_promotion_attempts
    )


def test_candidate_evaluation_interval_must_fit_update_window() -> None:
    with pytest.raises(ValueError, match="candidate_evaluation_interval"):
        OnlineRuntimeConfig(
            block_size=4,
            update_stride=2,
            feedback_top_k=3,
            candidate_evaluation_interval=3,
            fast=FastResidualConfig(rank=2, alpha=2.0),
        ).validate(vocabulary_size=6)


def _fake_run(label: str, repetition: int, tps: float, tpf: float) -> dict:
    diagnostics = None
    if label != "static":
        diagnostics = {
            "update_fraction_of_decode": 0.1,
            "update_attempts": 2,
            "updates_rolled_back": 0,
            "static_shadow_resets": 0,
            "update_seconds": 1.0,
            "feedback_materialization_seconds": 0.2,
            "head_forward_seconds": 0.1,
        }
    return {
        "label": label,
        "repetition": repetition,
        "prompt_index": 0,
        "result": {
            "metrics": {
                "decode_tokens_per_second": tps,
                "decoder_tokens_per_forward": tpf,
                "spec_acceptance_rate": 0.5,
                "peak_memory_allocated_bytes": 1_000,
                "decode_seconds": 10.0,
            },
            "diagnostics": diagnostics,
        },
    }


def test_online_summary_uses_paired_static_ratios() -> None:
    runs = [
        _fake_run("static", 0, 10.0, 1.2),
        _fake_run("online_s10", 0, 12.0, 1.5),
        _fake_run("static", 1, 20.0, 1.2),
        _fake_run("online_s10", 1, 24.0, 1.5),
    ]
    summary = summarize_online_runs(
        runs,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )
    assert summary["online_s10"]["paired_decode_speed_ratio"]["estimate"] == 1.2
    assert summary["online_s10"]["paired_tpf_ratio"]["estimate"] == 1.25

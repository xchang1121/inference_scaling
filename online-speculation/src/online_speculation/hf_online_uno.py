"""Real-checkpoint Online Uno prototype on the Hugging Face KV-cache fallback.

The implementation adds a request-local low-rank logit residual after Uno's
frozen diffusion draft.  Model forward passes remain inference-only; verifier
feedback trains only the detached residual head after exact verification.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import Tensor

from .fast_residual import (
    FastResidualConfig,
    FastResidualHead,
    FastResidualLearner,
    FastUpdateReport,
    ResidualFeedback,
    assert_optimizer_isolated,
    feedback_batch_from_logits,
)
from .hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
    HfUnoRuntime,
    RunMetrics,
    _cache_length,
    _crop_cache_by,
    _dtype,
    _first_stop_length,
    _generator,
    _parse_ints,
    _sha256,
    _sync,
    load_runtime,
)
from .stage2_analysis import bootstrap_interval
from .torch_sampling import (
    SamplingConfig,
    filtered_distribution,
    filtered_overlap,
    verify_linear_filtered,
)


Supervision = Literal["full", "on_policy", "discounted_tail"]
ActivationMode = Literal["immediate", "deferred"]


@dataclass(frozen=True)
class OnlineRuntimeConfig:
    block_size: int = 8
    update_stride: int = 10
    feedback_top_k: int = 50
    supervision: Supervision = "on_policy"
    tail_discount: float = 0.25
    position_discount: float = 0.97
    decay_factor_per_cycle: float = 1.0
    activation_mode: ActivationMode = "immediate"
    feedback_interval: int = 1
    promotion_margin: float = 0.002
    future_reset_margin: float = 0.005
    fast: FastResidualConfig = field(default_factory=FastResidualConfig)

    def validate(self, *, vocabulary_size: int) -> None:
        if self.block_size < 2:
            raise ValueError("online Uno requires block_size >= 2.")
        if self.update_stride < 1:
            raise ValueError("update_stride must be positive.")
        if self.feedback_top_k < 2 or self.feedback_top_k > vocabulary_size:
            raise ValueError("feedback_top_k must lie in [2, vocabulary_size].")
        if self.supervision not in ("full", "on_policy", "discounted_tail"):
            raise ValueError(f"unknown supervision mode: {self.supervision}.")
        if not 0 <= self.tail_discount <= 1:
            raise ValueError("tail_discount must lie in [0, 1].")
        if not 0 < self.position_discount <= 1:
            raise ValueError("position_discount must lie in (0, 1].")
        if not 0 <= self.decay_factor_per_cycle <= 1:
            raise ValueError("decay_factor_per_cycle must lie in [0, 1].")
        if self.activation_mode not in ("immediate", "deferred"):
            raise ValueError(f"unknown activation mode: {self.activation_mode}.")
        if self.feedback_interval < 1 or self.feedback_interval > self.update_stride:
            raise ValueError("feedback_interval must lie in [1, update_stride].")
        if self.promotion_margin < 0 or self.future_reset_margin < 0:
            raise ValueError("deferred decision margins cannot be negative.")
        self.fast.validate()


@dataclass(frozen=True)
class OnlineDiagnostics:
    activation_mode: str
    parameter_isolation: dict[str, int]
    feedback_cycles: int
    feedback_items_created: int
    feedback_items_discarded_at_end: int
    update_attempts: int
    updates_applied: int
    updates_rolled_back: int
    static_shadow_resets: int
    candidate_promotion_attempts: int
    candidate_promotions: int
    candidate_rejections: int
    future_static_resets: int
    head_forward_seconds: float
    candidate_head_forward_seconds: float
    feedback_materialization_seconds: float
    update_seconds: float
    update_fraction_of_decode: float
    final_fast_weight_l2: float
    max_fast_weight_l2: float
    update_reports: tuple[dict[str, Any], ...]
    promotion_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OnlineRunResult:
    metrics: RunMetrics
    diagnostics: OnlineDiagnostics


def choose_deferred_action(
    *,
    active_tv: float,
    candidate_tv: float,
    static_tv: float,
    promotion_margin: float,
    reset_margin: float,
) -> str:
    """Choose the best future-validated state with explicit improvement margins."""

    values = (active_tv, candidate_tv, static_tv)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("deferred TV evidence must be finite.")
    if promotion_margin < 0 or reset_margin < 0:
        raise ValueError("deferred decision margins cannot be negative.")
    if (
        candidate_tv + promotion_margin < active_tv
        and candidate_tv <= static_tv
    ):
        return "promote_candidate"
    if static_tv + reset_margin < active_tv and static_tv < candidate_tv:
        return "reset_to_static"
    return "keep_active"


def feedback_weights(
    *,
    speculative_tokens: int,
    rejected_index: int | None,
    supervision: Supervision,
    tail_discount: float,
    position_discount: float,
) -> list[float]:
    if speculative_tokens < 1:
        raise ValueError("speculative_tokens must be positive.")
    if rejected_index is not None and not 0 <= rejected_index < speculative_tokens:
        raise ValueError("rejected_index lies outside the speculative block.")
    weights = []
    for position in range(speculative_tokens):
        weight = position_discount**position
        if rejected_index is not None and position > rejected_index:
            if supervision == "on_policy":
                weight = 0.0
            elif supervision == "discounted_tail":
                weight *= tail_discount ** (position - rejected_index)
            elif supervision != "full":
                raise ValueError(f"unknown supervision mode: {supervision}.")
        weights.append(float(weight))
    return weights


class HfOnlineUnoRunner:
    """Attach a fresh fast residual learner to each generation request."""

    def __init__(self, runtime: HfUnoRuntime) -> None:
        self.runtime = runtime

    def _new_learner(
        self,
        config: OnlineRuntimeConfig,
        *,
        initialization_seed: int,
    ) -> tuple[FastResidualLearner, dict[str, int]]:
        hidden_size = int(self.runtime.model.config.hidden_size)
        device = self.runtime.device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(initialization_seed)
            head = FastResidualHead(
                hidden_size=hidden_size,
                vocabulary_size=self.runtime.vocab_size,
                rank=config.fast.rank,
                alpha=config.fast.alpha,
                device=device,
            )
        learner = FastResidualLearner(head, config.fast)
        isolation = assert_optimizer_isolated(
            base_model=self.runtime.model,
            head=head,
            optimizer=learner.optimizer,
        )
        return learner, isolation

    def _draft_forward_with_last_hidden(
        self,
        *,
        draft_input: Tensor,
        cache: object,
    ) -> tuple[object, Tensor]:
        """Capture the lm-head input even when remote code drops hidden states."""

        captured: list[Tensor] = []
        getter = getattr(self.runtime.model, "get_output_embeddings", None)
        output_module = getter() if callable(getter) else None
        handle = None
        if output_module is not None:

            def capture_hidden(module: object, inputs: tuple[Tensor, ...]) -> None:
                del module
                if not inputs:
                    raise RuntimeError("output embedding hook received no inputs.")
                captured.append(inputs[0])

            handle = output_module.register_forward_pre_hook(capture_hidden)
        try:
            with torch.inference_mode():
                output = self.runtime.model(
                    input_ids=draft_input,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=output_module is None,
                )
        finally:
            if handle is not None:
                handle.remove()

        if captured:
            last_hidden = captured[-1]
        else:
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError(
                    "model exposed neither lm-head input nor final hidden states."
                )
            last_hidden = hidden_states[-1]
        expected_shape = (
            int(draft_input.size(0)),
            int(draft_input.size(1)),
            int(self.runtime.model.config.hidden_size),
        )
        if tuple(last_hidden.shape) != expected_shape:
            raise RuntimeError(
                "captured draft hidden shape differs from model configuration: "
                f"{tuple(last_hidden.shape)} != {expected_shape}."
            )
        return output, last_hidden

    def _materialize_feedback(
        self,
        *,
        hidden_rows: Tensor,
        base_logits: Tensor,
        adjusted_logits: Tensor,
        target_logits: Tensor,
        rejected_index: int | None,
        config: OnlineRuntimeConfig,
    ) -> list[ResidualFeedback]:
        weights = feedback_weights(
            speculative_tokens=base_logits.size(0),
            rejected_index=rejected_index,
            supervision=config.supervision,
            tail_discount=config.tail_discount,
            position_discount=config.position_discount,
        )
        return feedback_batch_from_logits(
            hidden_rows=hidden_rows,
            base_logits=base_logits,
            adjusted_logits=adjusted_logits,
            target_logits=target_logits,
            top_k=config.feedback_top_k,
            temperature=self.runtime.sampling.temperature,
            weights=weights,
        )

    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        seed: int,
        initialization_seed: int,
        config: OnlineRuntimeConfig,
    ) -> OnlineRunResult:
        if max_new_tokens < 2:
            raise ValueError("online generation requires at least two output tokens.")
        if self.runtime.sampling.temperature <= 0:
            raise ValueError("online fast residual currently requires stochastic sampling.")
        config.validate(vocabulary_size=self.runtime.vocab_size)
        learner, isolation = self._new_learner(
            config,
            initialization_seed=initialization_seed,
        )
        generator = _generator(self.runtime.device, seed)
        base_memory = (
            torch.cuda.memory_allocated(self.runtime.device)
            if self.runtime.device.type == "cuda"
            else 0
        )
        if self.runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.runtime.device)

        cache, seed_token, prefill_seconds = self.runtime._prefill(input_ids, generator)
        output_tokens = [seed_token]
        stopped = (
            not self.runtime.ignore_stop and seed_token in self.runtime.stop_token_ids
        )
        cycles = 0
        cycles_since_update = 0
        committed_cycle_tokens = 0
        accepted_spec_tokens = 0
        attempted_spec_tokens = 0
        lookaheads = 0
        buffer: list[ResidualFeedback] = []
        candidate: FastResidualLearner | None = None
        update_reports: list[FastUpdateReport] = []
        promotion_events: list[dict[str, Any]] = []
        feedback_items_created = 0
        feedback_cycles = 0
        update_seconds = 0.0
        feedback_seconds = 0.0
        head_seconds_cpu = 0.0
        candidate_head_seconds_cpu = 0.0
        head_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        candidate_head_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        max_fast_weight_l2 = 0.0
        candidate_promotion_attempts = 0
        candidate_promotions = 0
        candidate_rejections = 0
        future_static_resets = 0
        future_active_tv = torch.zeros((), device=self.runtime.device)
        future_candidate_tv = torch.zeros((), device=self.runtime.device)
        future_static_tv = torch.zeros((), device=self.runtime.device)
        future_weight = torch.zeros((), device=self.runtime.device)

        _sync(self.runtime.device)
        decode_start = time.perf_counter()
        while len(output_tokens) < max_new_tokens and not stopped:
            prefix_cache_length = _cache_length(cache)
            noise = torch.randint(
                1,
                self.runtime.mask_token_id,
                (1, config.block_size - 1),
                device=self.runtime.device,
                dtype=torch.long,
                generator=generator,
            )
            seed_tensor = torch.tensor(
                [[seed_token]],
                device=self.runtime.device,
                dtype=torch.long,
            )
            draft_input = torch.cat((seed_tensor, noise), dim=1)
            lora_mask = torch.ones(
                (1, config.block_size),
                device=self.runtime.device,
                dtype=torch.float32,
            )
            lora_mask[:, 0] = 0.0
            self.runtime.router.set_token_mask(lora_mask)
            draft_output, last_hidden = self._draft_forward_with_last_hidden(
                draft_input=draft_input,
                cache=cache,
            )
            cache = draft_output.past_key_values
            if _cache_length(cache) != prefix_cache_length + config.block_size:
                raise RuntimeError("online draft cache did not advance by block_size.")
            _crop_cache_by(cache, config.block_size - 1)

            draft_logits = draft_output.logits[0]
            hidden_rows = last_hidden[0, 1:]
            free_token = int(self.runtime._sample_logits(draft_logits[0:1], generator).item())
            if self.runtime.device.type == "cuda":
                head_start = torch.cuda.Event(enable_timing=True)
                head_end = torch.cuda.Event(enable_timing=True)
                head_start.record()
                with torch.no_grad():
                    adjusted_logits = learner.corrected_logits(
                        hidden_rows,
                        draft_logits[1:],
                    )
                head_end.record()
                head_events.append((head_start, head_end))
            else:
                head_start_time = time.perf_counter()
                with torch.no_grad():
                    adjusted_logits = learner.corrected_logits(
                        hidden_rows,
                        draft_logits[1:],
                    )
                head_seconds_cpu += time.perf_counter() - head_start_time
            draft_used = filtered_distribution(adjusted_logits, self.runtime.sampling)
            spec_tokens = draft_used.sample(generator)
            proposal = torch.cat(
                (
                    torch.tensor(
                        [free_token],
                        device=self.runtime.device,
                        dtype=torch.long,
                    ),
                    spec_tokens,
                )
            ).unsqueeze(0)

            with torch.inference_mode(), self.runtime._base_context():
                verify_output = self.runtime.model(
                    input_ids=proposal,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = verify_output.past_key_values
            verify_logits = verify_output.logits[0]
            target = filtered_distribution(verify_logits[:-1], self.runtime.sampling)
            lookahead = filtered_distribution(verify_logits[-1:], self.runtime.sampling)
            verification = verify_linear_filtered(
                free_token=free_token,
                spec_tokens=spec_tokens,
                target=target,
                draft_used=draft_used,
                lookahead=lookahead,
                generator=generator,
            )

            if config.activation_mode == "deferred" and candidate is not None:
                if self.runtime.device.type == "cuda":
                    candidate_start = torch.cuda.Event(enable_timing=True)
                    candidate_end = torch.cuda.Event(enable_timing=True)
                    candidate_start.record()
                    with torch.no_grad():
                        candidate_logits = candidate.corrected_logits(
                            hidden_rows,
                            draft_logits[1:],
                        )
                    candidate_end.record()
                    candidate_head_events.append((candidate_start, candidate_end))
                else:
                    candidate_start_time = time.perf_counter()
                    with torch.no_grad():
                        candidate_logits = candidate.corrected_logits(
                            hidden_rows,
                            draft_logits[1:],
                        )
                    candidate_head_seconds_cpu += (
                        time.perf_counter() - candidate_start_time
                    )
                candidate_distribution = filtered_distribution(
                    candidate_logits,
                    self.runtime.sampling,
                )
                static_distribution = filtered_distribution(
                    draft_logits[1:].float(),
                    self.runtime.sampling,
                )
                evidence_weights = torch.tensor(
                    feedback_weights(
                        speculative_tokens=config.block_size - 1,
                        rejected_index=verification.rejected_index,
                        supervision=config.supervision,
                        tail_discount=config.tail_discount,
                        position_discount=config.position_discount,
                    ),
                    device=self.runtime.device,
                )
                active_tv = 1.0 - filtered_overlap(target, draft_used)
                candidate_tv = 1.0 - filtered_overlap(
                    target,
                    candidate_distribution,
                )
                static_tv = 1.0 - filtered_overlap(target, static_distribution)
                future_active_tv += torch.sum(evidence_weights * active_tv)
                future_candidate_tv += torch.sum(evidence_weights * candidate_tv)
                future_static_tv += torch.sum(evidence_weights * static_tv)
                future_weight += evidence_weights.sum()
                del candidate_logits, candidate_distribution, static_distribution

            committed = list(verification.committed)
            if not self.runtime.ignore_stop:
                committed = committed[
                    : _first_stop_length(committed, self.runtime.stop_token_ids)
                ]
            remaining = max_new_tokens - len(output_tokens)
            committed = committed[:remaining]
            if not committed:
                raise RuntimeError("online Uno cycle committed no tokens.")
            _crop_cache_by(cache, config.block_size + 1 - len(committed))
            expected_cache_length = prefix_cache_length + len(committed)
            if _cache_length(cache) != expected_cache_length:
                raise RuntimeError("online post-verification cache frontier is inconsistent.")

            if (cycles + 1) % config.feedback_interval == 0:
                feedback_start = time.perf_counter()
                round_feedback = self._materialize_feedback(
                    hidden_rows=hidden_rows,
                    base_logits=draft_logits[1:],
                    adjusted_logits=adjusted_logits,
                    target_logits=verify_logits[:-1],
                    rejected_index=verification.rejected_index,
                    config=config,
                )
                feedback_seconds += time.perf_counter() - feedback_start
                feedback_items_created += len(round_feedback)
                feedback_cycles += 1
                buffer.extend(round_feedback)

            output_tokens.extend(committed)
            seed_token = committed[-1]
            cycles += 1
            cycles_since_update += 1
            committed_cycle_tokens += len(committed)
            visible_specs = min(
                verification.accepted_spec_tokens,
                max(0, len(committed) - 1),
            )
            accepted_spec_tokens += visible_specs
            attempted_spec_tokens += min(
                config.block_size - 1,
                max(0, len(committed) - 1),
            )
            lookaheads += int(
                verification.used_lookahead
                and len(committed) == config.block_size + 1
            )
            stopped = (
                not self.runtime.ignore_stop and seed_token in self.runtime.stop_token_ids
            )

            will_continue = len(output_tokens) < max_new_tokens and not stopped
            if (
                will_continue
                and buffer
                and cycles_since_update >= config.update_stride
            ):
                _sync(self.runtime.device)
                update_start = time.perf_counter()
                if config.activation_mode == "deferred":
                    if candidate is not None:
                        if float(future_weight.item()) <= 0:
                            raise RuntimeError(
                                "deferred candidate received no future evidence."
                            )
                        active_mean_tv = float(
                            (future_active_tv / future_weight).item()
                        )
                        candidate_mean_tv = float(
                            (future_candidate_tv / future_weight).item()
                        )
                        static_mean_tv = float(
                            (future_static_tv / future_weight).item()
                        )
                        action = choose_deferred_action(
                            active_tv=active_mean_tv,
                            candidate_tv=candidate_mean_tv,
                            static_tv=static_mean_tv,
                            promotion_margin=config.promotion_margin,
                            reset_margin=config.future_reset_margin,
                        )
                        candidate_promotion_attempts += 1
                        if action == "promote_candidate":
                            learner = candidate
                            candidate_promotions += 1
                        elif action == "reset_to_static":
                            learner.reset_to_offline()
                            future_static_resets += 1
                        else:
                            candidate_rejections += 1
                        isolation = assert_optimizer_isolated(
                            base_model=self.runtime.model,
                            head=learner.head,
                            optimizer=learner.optimizer,
                        )
                        promotion_events.append(
                            {
                                "cycle": cycles,
                                "action": action,
                                "future_rows_weight": float(future_weight.item()),
                                "active_filtered_tv": active_mean_tv,
                                "candidate_filtered_tv": candidate_mean_tv,
                                "static_filtered_tv": static_mean_tv,
                                "active_fast_weight_l2_after_action": (
                                    learner.fast_weight_l2()
                                ),
                            }
                        )
                        candidate = None
                    future_active_tv.zero_()
                    future_candidate_tv.zero_()
                    future_static_tv.zero_()
                    future_weight.zero_()
                    candidate = learner.clone()
                    report = candidate.update(buffer)
                    if report.rolled_back:
                        candidate = None
                else:
                    report = learner.update(buffer)
                _sync(self.runtime.device)
                update_seconds += time.perf_counter() - update_start
                update_reports.append(report)
                max_fast_weight_l2 = max(
                    max_fast_weight_l2,
                    report.fast_weight_l2,
                    learner.fast_weight_l2(),
                )
                buffer.clear()
                cycles_since_update = 0
            if config.decay_factor_per_cycle < 1.0:
                learner.decay_toward_offline(config.decay_factor_per_cycle)
                if candidate is not None:
                    candidate.decay_toward_offline(config.decay_factor_per_cycle)

            del draft_output, verify_output, target, lookahead, draft_used

        _sync(self.runtime.device)
        decode_seconds = time.perf_counter() - decode_start
        head_forward_seconds = head_seconds_cpu
        if self.runtime.device.type == "cuda":
            head_forward_seconds = sum(
                start.elapsed_time(end) for start, end in head_events
            ) / 1_000.0
        candidate_head_forward_seconds = candidate_head_seconds_cpu
        if self.runtime.device.type == "cuda":
            candidate_head_forward_seconds = sum(
                start.elapsed_time(end) for start, end in candidate_head_events
            ) / 1_000.0

        metrics = self.runtime._metrics(
            method=(
                "uno_deferred_fast_residual_hf_fallback"
                if config.activation_mode == "deferred"
                else "uno_online_fast_residual_hf_fallback"
            ),
            block_size=config.block_size,
            input_ids=input_ids,
            output_tokens=output_tokens,
            forwards=2 * cycles,
            cycles=cycles,
            committed_cycle_tokens=committed_cycle_tokens,
            accepted_spec_tokens=accepted_spec_tokens,
            attempted_spec_tokens=attempted_spec_tokens,
            lookaheads=lookaheads,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            base_memory=base_memory,
            stopped=stopped,
        )
        applied = sum(not report.rolled_back for report in update_reports)
        diagnostics = OnlineDiagnostics(
            activation_mode=config.activation_mode,
            parameter_isolation=isolation,
            feedback_cycles=feedback_cycles,
            feedback_items_created=feedback_items_created,
            feedback_items_discarded_at_end=len(buffer),
            update_attempts=len(update_reports),
            updates_applied=applied,
            updates_rolled_back=len(update_reports) - applied,
            static_shadow_resets=sum(
                report.reset_to_offline for report in update_reports
            ),
            candidate_promotion_attempts=candidate_promotion_attempts,
            candidate_promotions=candidate_promotions,
            candidate_rejections=candidate_rejections,
            future_static_resets=future_static_resets,
            head_forward_seconds=head_forward_seconds,
            candidate_head_forward_seconds=candidate_head_forward_seconds,
            feedback_materialization_seconds=feedback_seconds,
            update_seconds=update_seconds,
            update_fraction_of_decode=(
                update_seconds / decode_seconds if decode_seconds else 0.0
            ),
            final_fast_weight_l2=learner.fast_weight_l2(),
            max_fast_weight_l2=max_fast_weight_l2,
            update_reports=tuple(asdict(report) for report in update_reports),
            promotion_events=tuple(promotion_events),
        )
        return OnlineRunResult(metrics=metrics, diagnostics=diagnostics)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(statistics.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_online_runs(
    runs: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["label"]), []).append(run)
    if "static" not in grouped:
        raise ValueError("online benchmark requires a static baseline.")
    result: dict[str, Any] = {}
    static_by_key = {
        (int(run["repetition"]), int(run["prompt_index"])): run
        for run in grouped["static"]
    }
    for label_index, (label, items) in enumerate(grouped.items()):
        metrics = [item["result"]["metrics"] for item in items]
        result[label] = {
            "runs": len(items),
            "decode_tokens_per_second": _summary(
                [float(metric["decode_tokens_per_second"]) for metric in metrics]
            ),
            "decoder_tokens_per_forward": _summary(
                [float(metric["decoder_tokens_per_forward"]) for metric in metrics]
            ),
            "spec_acceptance_rate": _summary(
                [float(metric["spec_acceptance_rate"]) for metric in metrics]
            ),
            "peak_memory_allocated_bytes": _summary(
                [float(metric["peak_memory_allocated_bytes"]) for metric in metrics]
            ),
        }
        if label == "static":
            continue
        ratios = []
        tpf_ratios = []
        acceptance_deltas = []
        peak_memory_deltas = []
        update_fractions = []
        feedback_fractions = []
        head_fractions = []
        candidate_head_fractions = []
        explicit_online_fractions = []
        rollback_fractions = []
        reset_fractions = []
        update_attempts = []
        for item in items:
            key = (int(item["repetition"]), int(item["prompt_index"]))
            if key not in static_by_key:
                raise ValueError(f"online run {key} has no paired static baseline.")
            metric = item["result"]["metrics"]
            static_metric = static_by_key[key]["result"]["metrics"]
            ratios.append(
                float(metric["decode_tokens_per_second"])
                / float(static_metric["decode_tokens_per_second"])
            )
            tpf_ratios.append(
                float(metric["decoder_tokens_per_forward"])
                / float(static_metric["decoder_tokens_per_forward"])
            )
            acceptance_deltas.append(
                float(metric["spec_acceptance_rate"])
                - float(static_metric["spec_acceptance_rate"])
            )
            peak_memory_deltas.append(
                float(metric["peak_memory_allocated_bytes"])
                - float(static_metric["peak_memory_allocated_bytes"])
            )
            diagnostics = item["result"]["diagnostics"]
            decode_seconds = float(metric["decode_seconds"])
            update_fractions.append(float(diagnostics["update_fraction_of_decode"]))
            feedback_fractions.append(
                float(diagnostics["feedback_materialization_seconds"]) / decode_seconds
            )
            head_fractions.append(
                float(diagnostics["head_forward_seconds"]) / decode_seconds
            )
            candidate_head_seconds = float(
                diagnostics.get("candidate_head_forward_seconds", 0.0)
            )
            candidate_head_fractions.append(candidate_head_seconds / decode_seconds)
            explicit_online_fractions.append(
                (
                    float(diagnostics["update_seconds"])
                    + float(diagnostics["feedback_materialization_seconds"])
                    + float(diagnostics["head_forward_seconds"])
                    + candidate_head_seconds
                )
                / decode_seconds
            )
            attempts = int(diagnostics["update_attempts"])
            update_attempts.append(float(attempts))
            rollback_fractions.append(
                int(diagnostics["updates_rolled_back"]) / attempts if attempts else 0.0
            )
            reset_fractions.append(
                int(diagnostics["static_shadow_resets"]) / attempts if attempts else 0.0
            )
        local_seed = bootstrap_seed + 10_000 * label_index
        result[label].update(
            {
                "paired_decode_speed_ratio": bootstrap_interval(
                    np.asarray(ratios),
                    samples=bootstrap_samples,
                    seed=local_seed + 1,
                ),
                "paired_tpf_ratio": bootstrap_interval(
                    np.asarray(tpf_ratios),
                    samples=bootstrap_samples,
                    seed=local_seed + 2,
                ),
                "paired_acceptance_rate_delta": bootstrap_interval(
                    np.asarray(acceptance_deltas),
                    samples=bootstrap_samples,
                    seed=local_seed + 3,
                ),
                "paired_peak_memory_delta_bytes": bootstrap_interval(
                    np.asarray(peak_memory_deltas),
                    samples=bootstrap_samples,
                    seed=local_seed + 4,
                ),
                "update_attempts": _summary(update_attempts),
                "update_fraction_of_decode": _summary(update_fractions),
                "feedback_fraction_of_decode": _summary(feedback_fractions),
                "head_fraction_of_decode": _summary(head_fractions),
                "candidate_head_fraction_of_decode": _summary(
                    candidate_head_fractions
                ),
                "explicit_online_fraction_of_decode": _summary(
                    explicit_online_fractions
                ),
                "rollback_fraction": _summary(rollback_fractions),
                "static_shadow_reset_fraction": _summary(reset_fractions),
            }
        )
    return result


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--update-strides", type=_parse_ints, default=[5, 10, 20])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--feedback-top-k", type=int, default=50)
    parser.add_argument(
        "--supervision",
        choices=("full", "on_policy", "discounted_tail"),
        default="on_policy",
    )
    parser.add_argument("--tail-discount", type=float, default=0.25)
    parser.add_argument("--position-discount", type=float, default=0.97)
    parser.add_argument("--decay-factor", type=float, default=1.0)
    parser.add_argument(
        "--activation-mode",
        choices=("immediate", "deferred"),
        default="immediate",
    )
    parser.add_argument("--feedback-interval", type=int, default=1)
    parser.add_argument("--promotion-margin", type=float, default=0.002)
    parser.add_argument("--future-reset-margin", type=float, default=0.005)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--mask-token-id", type=int, default=64256)
    parser.add_argument("--stop-token-ids", type=_parse_ints, default=[64019, 1])
    parser.add_argument("--ignore-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=30_000)
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.repetitions < 2:
        raise ValueError("at least two repetitions are required for paired analysis.")
    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve()
    base_weight = model_path / "model-00000-of-00001.safetensors"
    adapter_weight = adapter_path / "adapter_model.safetensors"
    for required in (base_weight, adapter_weight):
        if not required.is_file():
            raise FileNotFoundError(required)
    hashes = {
        "base_weight_sha256": _sha256(base_weight),
        "adapter_weight_sha256": _sha256(adapter_weight),
    }
    if not args.skip_hash_check:
        if hashes["base_weight_sha256"] != BASE_WEIGHT_SHA256:
            raise RuntimeError("base checkpoint SHA-256 does not match the pinned revision.")
        if hashes["adapter_weight_sha256"] != ADAPTER_WEIGHT_SHA256:
            raise RuntimeError("Uno adapter SHA-256 does not match the pinned revision.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    sampling = SamplingConfig(
        temperature=float(args.temperature),
        top_k=int(args.top_k) if args.top_k > 0 else None,
        top_p=float(args.top_p),
    )
    runtime = load_runtime(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        dtype=_dtype(args.dtype),
        sampling=sampling,
        mask_token_id=args.mask_token_id,
        stop_token_ids=args.stop_token_ids,
        ignore_stop=args.ignore_stop,
    )
    online_runner = HfOnlineUnoRunner(runtime)
    prompts = args.prompt or [
        "Explain in three concise paragraphs why speculative decoding can be lossless."
    ]
    encoded_prompts = [runtime.encode_prompt(prompt) for prompt in prompts]
    routing_probe = runtime.routing_probe(
        encoded_prompts[0],
        block_size=args.block_size,
        seed=args.seed,
    )
    fast_config = FastResidualConfig(
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
    )

    runtime.generate_uno(
        encoded_prompts[0],
        max_new_tokens=max(2, args.warmup_tokens),
        block_size=args.block_size,
        seed=args.seed - 1,
    )
    online_runner.generate(
        encoded_prompts[0],
        max_new_tokens=max(args.block_size + 2, args.warmup_tokens),
        seed=args.seed - 1,
        initialization_seed=args.seed + 999_999,
        config=OnlineRuntimeConfig(
            block_size=args.block_size,
            # Stride one guarantees at least one backward before formal runs
            # when max_new_tokens is at least block_size + 2.
            update_stride=1,
            feedback_top_k=args.feedback_top_k,
            supervision=args.supervision,
            tail_discount=args.tail_discount,
            position_discount=args.position_discount,
            decay_factor_per_cycle=args.decay_factor,
            fast=fast_config,
        ),
    )

    online_prefix = "deferred" if args.activation_mode == "deferred" else "online"
    methods: list[tuple[str, int | None]] = [("static", None)] + [
        (f"{online_prefix}_s{stride}", stride) for stride in args.update_strides
    ]
    runs: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        for prompt_index, input_ids in enumerate(encoded_prompts):
            run_seed = args.seed + 1_000 * repetition + prompt_index
            order_offset = (repetition + prompt_index) % len(methods)
            ordered_methods = methods[order_offset:] + methods[:order_offset]
            for label, stride in ordered_methods:
                if stride is None:
                    metrics = runtime.generate_uno(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        block_size=args.block_size,
                        seed=run_seed,
                    )
                    result: dict[str, Any] = {
                        "metrics": asdict(metrics),
                        "diagnostics": None,
                    }
                else:
                    online_result = online_runner.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        seed=run_seed,
                        initialization_seed=run_seed + 999_999,
                        config=OnlineRuntimeConfig(
                            block_size=args.block_size,
                            update_stride=stride,
                            feedback_top_k=args.feedback_top_k,
                            supervision=args.supervision,
                            tail_discount=args.tail_discount,
                            position_discount=args.position_discount,
                            decay_factor_per_cycle=args.decay_factor,
                            activation_mode=args.activation_mode,
                            feedback_interval=args.feedback_interval,
                            promotion_margin=args.promotion_margin,
                            future_reset_margin=args.future_reset_margin,
                            fast=fast_config,
                        ),
                    )
                    result = asdict(online_result)
                runs.append(
                    {
                        "label": label,
                        "repetition": repetition,
                        "prompt_index": prompt_index,
                        "seed": run_seed,
                        "result": result,
                    }
                )
                run_metrics = result["metrics"]
                print(
                    f"{label} rep={repetition} prompt={prompt_index} "
                    f"TPF={run_metrics['decoder_tokens_per_forward']:.3f} "
                    f"decode_TPS={run_metrics['decode_tokens_per_second']:.2f}",
                    flush=True,
                )

    summary = summarize_online_runs(
        runs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    result = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_backend": "huggingface_pytorch_kv_cache_online_fast_residual",
        "claim_scope": {
            "sampling": "exact filtered linear Psi-Spec using saved pre-update q",
            "online_parameters": "request-local rank-r logit residual only",
            "base_and_offline_uno": "frozen and excluded from optimizer",
            "performance": "actual wall-clock on HF fallback, not official Nano-vLLM",
        },
        "checkpoint": {
            "base_id": "IFM/K2-Horizon-0.9B",
            "base_revision": BASE_REVISION,
            "adapter_id": "IFM/K2-Horizon-0.9B-Uno",
            "adapter_revision": ADAPTER_REVISION,
            **hashes,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "sampling": {
            **asdict(sampling),
            "mask_token_id": args.mask_token_id,
            "stop_token_ids": args.stop_token_ids,
            "ignore_stop": args.ignore_stop,
        },
        "design": {
            "prompts": prompts,
            "block_size": args.block_size,
            "update_strides": args.update_strides,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "feedback_top_k": args.feedback_top_k,
            "supervision": args.supervision,
            "tail_discount": args.tail_discount,
            "position_discount": args.position_discount,
            "decay_factor_per_cycle": args.decay_factor,
            "activation_mode": args.activation_mode,
            "feedback_interval": args.feedback_interval,
            "promotion_margin": args.promotion_margin,
            "future_reset_margin": args.future_reset_margin,
            "fast": asdict(fast_config),
            "bootstrap_samples": args.bootstrap_samples,
            "method_order": (
                "cyclic Latin rotation by (repetition + prompt_index) modulo method count"
            ),
        },
        "routing_probe": routing_probe,
        "adapter_load_report": runtime.adapter_load_report,
        "runs": runs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()

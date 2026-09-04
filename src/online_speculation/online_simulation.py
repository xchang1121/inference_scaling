"""Controlled distribution-drift simulation for online Uno policies.

The target is a non-stationary first-order Markov language model.  Uno's draft
rows are tabular multi-step marginals, while verification observes exact target
conditionals along the proposed path.  Every update happens only after exact
Psi-Spec verification has completed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .psi_spec import total_variation, uno_linear_step
from .stage2_analysis import bootstrap_interval


FloatArray = NDArray[np.float64]
Supervision = Literal["full", "on_policy", "discounted_tail"]


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def make_transition_matrix(vocabulary_size: int, shift: int) -> FloatArray:
    """Build a structured Markov target whose modes rotate across regimes."""

    if vocabulary_size < 5:
        raise ValueError("vocabulary_size must be at least five.")
    matrix = np.full(
        (vocabulary_size, vocabulary_size),
        0.08 / (vocabulary_size - 3),
        dtype=np.float64,
    )
    for state in range(vocabulary_size):
        preferred = (3 * state + 1 + shift) % vocabulary_size
        secondary = (5 * state + 2 + 2 * shift) % vocabulary_size
        tertiary = (state + 3 * shift) % vocabulary_size
        # Accumulation handles rare index collisions and normalization below.
        matrix[state, preferred] += 0.58
        matrix[state, secondary] += 0.22
        matrix[state, tertiary] += 0.12
    return matrix / matrix.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class DriftSchedule:
    """Token-indexed deployment regimes shared by every strategy."""

    regime_0: FloatArray
    regime_1: FloatArray
    regime_2: FloatArray
    in_domain_end: int = 2_000
    shift_a_end: int = 5_000
    gradual_end: int = 7_000
    shift_b_end: int = 10_000
    total_tokens: int = 12_000

    @classmethod
    def create(cls, vocabulary_size: int, total_tokens: int = 12_000) -> "DriftSchedule":
        if total_tokens < 1_200:
            raise ValueError("total_tokens must be at least 1,200.")
        scale = total_tokens / 12_000
        boundaries = [
            round(2_000 * scale),
            round(5_000 * scale),
            round(7_000 * scale),
            round(10_000 * scale),
        ]
        return cls(
            make_transition_matrix(vocabulary_size, 0),
            make_transition_matrix(vocabulary_size, 2),
            make_transition_matrix(vocabulary_size, 5),
            *boundaries,
            total_tokens,
        )

    @property
    def vocabulary_size(self) -> int:
        return int(self.regime_0.shape[0])

    def matrix_at(self, token_index: int) -> FloatArray:
        if token_index < self.in_domain_end:
            return self.regime_0
        if token_index < self.shift_a_end:
            return self.regime_1
        if token_index < self.gradual_end:
            width = max(1, self.gradual_end - self.shift_a_end)
            alpha = (token_index - self.shift_a_end) / width
            return (1.0 - alpha) * self.regime_1 + alpha * self.regime_2
        if token_index < self.shift_b_end:
            return self.regime_2
        return self.regime_0

    def segment_at(self, token_index: int) -> str:
        if token_index < self.in_domain_end:
            return "in_domain"
        if token_index < self.shift_a_end:
            return "abrupt_shift_a"
        if token_index < self.gradual_end:
            return "gradual_a_to_b"
        if token_index < self.shift_b_end:
            return "shift_b"
        return "return_in_domain"

    def segment_bounds(self) -> dict[str, tuple[int, int]]:
        return {
            "in_domain": (0, self.in_domain_end),
            "abrupt_shift_a": (self.in_domain_end, self.shift_a_end),
            "gradual_a_to_b": (self.shift_a_end, self.gradual_end),
            "shift_b": (self.gradual_end, self.shift_b_end),
            "return_in_domain": (self.shift_b_end, self.total_tokens),
        }


def offline_marginal_draft(
    transition: FloatArray,
    speculative_tokens: int,
    smoothing: float = 0.02,
) -> FloatArray:
    """Return P^(i+2) marginals for Uno's speculative future positions."""

    if speculative_tokens < 1:
        raise ValueError("speculative_tokens must be positive.")
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must lie in [0, 1).")
    vocabulary_size = transition.shape[0]
    rows = np.empty(
        (vocabulary_size, speculative_tokens, vocabulary_size),
        dtype=np.float64,
    )
    power = transition.copy()
    for position in range(speculative_tokens):
        power = power @ transition
        rows[:, position, :] = power
    rows = (1.0 - smoothing) * rows + smoothing / vocabulary_size
    return rows / rows.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class FeedbackItem:
    start_state: int
    position: int
    target: FloatArray
    draft_used: FloatArray
    weight: float


@dataclass(frozen=True)
class UpdateReport:
    items: int
    effective_weight: float
    mean_tv_before: float
    mean_tv_after: float
    mean_forward_kl_before: float
    mean_forward_kl_after: float


class TabularFastDrafter:
    """Offline logits plus request-local tabular fast-weight corrections."""

    def __init__(self, offline_probabilities: FloatArray) -> None:
        probabilities = np.asarray(offline_probabilities, dtype=np.float64)
        if probabilities.ndim != 3 or np.any(probabilities <= 0):
            raise ValueError("offline probabilities must be a positive rank-three tensor.")
        self.offline_probabilities = probabilities / probabilities.sum(
            axis=-1,
            keepdims=True,
        )
        self.logits = np.log(self.offline_probabilities)

    def distributions(self, start_state: int) -> FloatArray:
        return _softmax(self.logits[int(start_state)]).copy()

    @staticmethod
    def _forward_kl(target: FloatArray, draft: FloatArray) -> float:
        return float(
            np.sum(target * (np.log(target.clip(1e-12)) - np.log(draft.clip(1e-12))))
        )

    def _batch_metrics(self, items: Sequence[FeedbackItem]) -> tuple[float, float]:
        tv_values = []
        kl_values = []
        for item in items:
            q = _softmax(self.logits[item.start_state, item.position])
            tv_values.append(total_variation(item.target, q))
            kl_values.append(self._forward_kl(item.target, q))
        return float(np.mean(tv_values)), float(np.mean(kl_values))

    def update(
        self,
        items: Sequence[FeedbackItem],
        *,
        learning_rate: float,
        tv_weight: float,
        forward_kl_weight: float,
        old_q_weight: float,
        gradient_clip: float,
    ) -> UpdateReport:
        if not items:
            raise ValueError("cannot update from an empty feedback batch.")
        tv_before, kl_before = self._batch_metrics(items)
        effective_weight = 0.0
        for item in items:
            if item.weight <= 0:
                continue
            row = self.logits[item.start_state, item.position]
            q = _softmax(row)
            kl_gradient = q - item.target
            signs = np.sign(q - item.target)
            tv_gradient = 0.5 * q * (signs - float(np.dot(signs, q)))
            old_q_gradient = q - item.draft_used
            gradient = (
                forward_kl_weight * kl_gradient
                + tv_weight * tv_gradient
                + old_q_weight * old_q_gradient
            )
            norm = float(np.linalg.norm(gradient))
            if norm > gradient_clip:
                gradient *= gradient_clip / norm
            row -= learning_rate * item.weight * gradient
            row -= float(row.mean())
            np.clip(row, -16.0, 16.0, out=row)
            effective_weight += item.weight
        tv_after, kl_after = self._batch_metrics(items)
        return UpdateReport(
            items=len(items),
            effective_weight=effective_weight,
            mean_tv_before=tv_before,
            mean_tv_after=tv_after,
            mean_forward_kl_before=kl_before,
            mean_forward_kl_after=kl_after,
        )


def supervision_weights(
    speculative_tokens: int,
    rejection_index: int | None,
    supervision: Supervision,
    tail_discount: float,
) -> FloatArray:
    if speculative_tokens < 1:
        raise ValueError("speculative_tokens must be positive.")
    if not 0 <= tail_discount <= 1:
        raise ValueError("tail_discount must lie in [0, 1].")
    weights = np.ones(speculative_tokens, dtype=np.float64)
    if rejection_index is None or supervision == "full":
        return weights
    if not 0 <= rejection_index < speculative_tokens:
        raise ValueError("rejection_index is outside the speculative block.")
    if supervision == "on_policy":
        weights[rejection_index + 1 :] = 0.0
        return weights
    if supervision == "discounted_tail":
        for position in range(rejection_index + 1, speculative_tokens):
            weights[position] = tail_discount ** (position - rejection_index)
        return weights
    raise ValueError(f"unknown supervision mode: {supervision}.")


def expected_committed_tokens(target_rows: FloatArray, draft_rows: FloatArray) -> float:
    """Approximate E[2 + accepted prefix] from per-position TV overlaps."""

    if target_rows.shape != draft_rows.shape or target_rows.ndim != 2:
        raise ValueError("target and draft rows must have the same rank-two shape.")
    acceptances = np.minimum(target_rows, draft_rows).sum(axis=1)
    survival = np.cumprod(acceptances)
    return 2.0 + float(survival.sum())


@dataclass(frozen=True)
class CostModel:
    """Forward-equivalent proxy calibrated from the Stage-2 B=8 result."""

    forward_pair_cost: float = 2.0715279487947216
    update_fixed_cost: float = 0.35
    update_item_cost: float = 0.002

    def update_cost(self, items: int) -> float:
        return self.update_fixed_cost + self.update_item_cost * items


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    stride: int | None
    supervision: Supervision = "discounted_tail"
    adaptive: bool = False
    tail_discount: float = 0.25
    learning_rate: float = 0.35
    tv_weight: float = 0.5
    forward_kl_weight: float = 1.0
    old_q_weight: float = 0.15
    gradient_clip: float = 1.0
    position_discount: float = 0.97

    def validate(self) -> None:
        if self.stride is not None and self.stride < 1:
            raise ValueError("stride must be positive or None.")
        if self.adaptive and self.stride is None:
            raise ValueError("adaptive strategy requires an initial stride.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass
class AdaptiveStrideController:
    """Confidence-gated stride controller using a static shadow proposal."""

    candidates: tuple[int, ...] = (1, 5, 10, 20)
    initial_stride: int = 10
    window_rounds: int = 100
    confidence_z: float = 1.645
    positive_margin: float = 0.005
    tv_floor: float = 0.04
    max_update_fraction: float = 0.30
    current_stride: int = field(init=False)
    events: list[dict[str, float | int | str]] = field(default_factory=list)
    _differences: list[float] = field(default_factory=list)
    _tv_values: list[float] = field(default_factory=list)
    _update_costs: list[float] = field(default_factory=list)
    _total_costs: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_stride not in self.candidates:
            raise ValueError("initial_stride must be one of candidates.")
        self.current_stride = self.initial_stride

    def observe(
        self,
        *,
        token_position: int,
        current_proxy_efficiency: float,
        static_proxy_efficiency: float,
        mean_tv: float,
        update_cost: float,
        total_cost: float,
    ) -> dict[str, float | int | str] | None:
        self._differences.append(current_proxy_efficiency - static_proxy_efficiency)
        self._tv_values.append(mean_tv)
        self._update_costs.append(update_cost)
        self._total_costs.append(total_cost)
        if len(self._differences) < self.window_rounds:
            return None

        differences = np.asarray(self._differences, dtype=np.float64)
        mean_difference = float(differences.mean())
        standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
        lower_bound = mean_difference - self.confidence_z * standard_error
        mean_tv = float(np.mean(self._tv_values))
        update_fraction = float(sum(self._update_costs) / sum(self._total_costs))
        old_stride = self.current_stride
        index = self.candidates.index(old_stride)
        reason = "hold"
        if (
            lower_bound > self.positive_margin
            and mean_tv > self.tv_floor
            and update_fraction < self.max_update_fraction
            and index > 0
        ):
            self.current_stride = self.candidates[index - 1]
            reason = "online_lcb_above_static_increase_frequency"
        elif (
            (lower_bound < 0.0 or update_fraction >= self.max_update_fraction)
            and index < len(self.candidates) - 1
        ):
            self.current_stride = self.candidates[index + 1]
            reason = "no_net_gain_or_excess_cost_decrease_frequency"

        event: dict[str, float | int | str] = {
            "token_position": token_position,
            "old_stride": old_stride,
            "new_stride": self.current_stride,
            "mean_proxy_efficiency_delta": mean_difference,
            "lower_confidence_bound": lower_bound,
            "mean_feedback_tv": mean_tv,
            "update_cost_fraction": update_fraction,
            "reason": reason,
        }
        self.events.append(event)
        self._differences.clear()
        self._tv_values.clear()
        self._update_costs.clear()
        self._total_costs.clear()
        return event


@dataclass
class _Accumulator:
    tokens: int = 0
    rounds: int = 0
    accepted: int = 0
    attempted: int = 0
    cost: float = 0.0
    update_cost: float = 0.0
    updates: int = 0
    items_updated: int = 0
    tv_sum: float = 0.0
    static_tv_sum: float = 0.0
    feedback_rows: int = 0

    def summary(self) -> dict[str, float | int]:
        return {
            "tokens": self.tokens,
            "rounds": self.rounds,
            "forwards": 2 * self.rounds,
            "tpf": self.tokens / (2 * self.rounds) if self.rounds else 0.0,
            "tokens_per_cost": self.tokens / self.cost if self.cost else 0.0,
            "mean_tokens_per_round": self.tokens / self.rounds if self.rounds else 0.0,
            "spec_acceptance_rate": self.accepted / self.attempted if self.attempted else 0.0,
            "mean_feedback_tv": self.tv_sum / self.feedback_rows if self.feedback_rows else 0.0,
            "mean_static_feedback_tv": (
                self.static_tv_sum / self.feedback_rows if self.feedback_rows else 0.0
            ),
            "dynamic_regret_proxy": self.tv_sum,
            "static_regret_proxy": self.static_tv_sum,
            "cost": self.cost,
            "update_cost": self.update_cost,
            "update_cost_fraction": self.update_cost / self.cost if self.cost else 0.0,
            "updates": self.updates,
            "items_updated": self.items_updated,
        }


def _feedback_items(
    *,
    start_state: int,
    target_rows: FloatArray,
    draft_used: FloatArray,
    rejection_index: int | None,
    strategy: StrategyConfig,
) -> list[FeedbackItem]:
    weights = supervision_weights(
        target_rows.shape[0],
        rejection_index,
        strategy.supervision,
        strategy.tail_discount,
    )
    items = []
    for position, weight in enumerate(weights):
        weight *= strategy.position_discount**position
        if weight <= 0:
            continue
        items.append(
            FeedbackItem(
                start_state=start_state,
                position=position,
                target=target_rows[position].copy(),
                draft_used=draft_used[position].copy(),
                weight=float(weight),
            )
        )
    return items


def simulate_strategy(
    *,
    schedule: DriftSchedule,
    offline_draft: FloatArray,
    block_size: int,
    strategy: StrategyConfig,
    cost_model: CostModel,
    seed: int,
    trace_window_tokens: int = 250,
) -> dict[str, object]:
    """Run one lossless adaptive proposal trajectory through the drift schedule."""

    strategy.validate()
    speculative_tokens = block_size - 1
    if speculative_tokens != offline_draft.shape[1]:
        raise ValueError("offline draft positions do not match block_size - 1.")
    rng = np.random.default_rng(seed)
    drafter = TabularFastDrafter(offline_draft)
    controller = (
        AdaptiveStrideController(initial_stride=int(strategy.stride))
        if strategy.adaptive
        else None
    )
    buffer: list[FeedbackItem] = []
    total = _Accumulator()
    segments: dict[str, _Accumulator] = defaultdict(_Accumulator)
    update_reports: list[UpdateReport] = []
    controller_events: list[dict[str, float | int | str]] = []
    trace: list[dict[str, float | int | str]] = []
    trace_accumulator = _Accumulator()
    next_trace_boundary = trace_window_tokens
    generated = 0
    state = int(rng.integers(0, schedule.vocabulary_size))
    rounds_since_update = 0
    negative_log_likelihood = 0.0

    while generated < schedule.total_tokens:
        start_position = generated
        round_start_state = state
        segment_name = schedule.segment_at(start_position)
        segment = segments[segment_name]
        q_used = drafter.distributions(round_start_state)
        q_static = offline_draft[round_start_state]

        def target_distribution(
            history: tuple[int, ...],
            *,
            _start_position: int = start_position,
        ) -> FloatArray:
            offset = len(history) - 1
            matrix = schedule.matrix_at(_start_position + offset)
            return matrix[int(history[-1])]

        step = uno_linear_step((state,), target_distribution, q_used, rng)
        target_rows = step.target_probabilities[:-1]
        committed = list(step.committed_tokens)
        remaining = schedule.total_tokens - generated
        committed = committed[:remaining]

        for token in committed:
            target = schedule.matrix_at(generated)[state]
            negative_log_likelihood -= math.log(max(float(target[token]), 1e-12))
            state = int(token)
            generated += 1

        rejection = step.verification.rejection_index
        attempted = (
            speculative_tokens if rejection is None else step.verification.accepted_count + 1
        )
        accepted = step.verification.accepted_count
        feedback_tv = np.asarray(
            [total_variation(p, q) for p, q in zip(target_rows, q_used)],
            dtype=np.float64,
        )
        static_tv = np.asarray(
            [total_variation(p, q) for p, q in zip(target_rows, q_static)],
            dtype=np.float64,
        )

        round_items: list[FeedbackItem] = []
        if strategy.stride is not None:
            round_items = _feedback_items(
                start_state=round_start_state,
                target_rows=target_rows,
                draft_used=q_used,
                rejection_index=rejection,
                strategy=strategy,
            )
            buffer.extend(round_items)

        rounds_since_update += 1
        stride = (
            controller.current_stride
            if controller is not None
            else strategy.stride
        )
        will_continue = generated < schedule.total_tokens
        should_update = (
            stride is not None
            and bool(buffer)
            and rounds_since_update >= stride
            and will_continue
        )
        update_cost = 0.0
        update_report = None
        if should_update:
            update_report = drafter.update(
                buffer,
                learning_rate=strategy.learning_rate,
                tv_weight=strategy.tv_weight,
                forward_kl_weight=strategy.forward_kl_weight,
                old_q_weight=strategy.old_q_weight,
                gradient_clip=strategy.gradient_clip,
            )
            update_reports.append(update_report)
            update_cost = cost_model.update_cost(len(buffer))
            buffer.clear()
            rounds_since_update = 0

        round_cost = cost_model.forward_pair_cost + update_cost
        current_expected = expected_committed_tokens(target_rows, q_used)
        static_expected = expected_committed_tokens(target_rows, q_static)
        if controller is not None:
            event = controller.observe(
                token_position=generated,
                current_proxy_efficiency=current_expected / round_cost,
                static_proxy_efficiency=static_expected / cost_model.forward_pair_cost,
                mean_tv=float(feedback_tv.mean()),
                update_cost=update_cost,
                total_cost=round_cost,
            )
            if event is not None:
                controller_events.append(event)

        for accumulator in (total, segment, trace_accumulator):
            accumulator.tokens += len(committed)
            accumulator.rounds += 1
            accumulator.accepted += accepted
            accumulator.attempted += attempted
            accumulator.cost += round_cost
            accumulator.update_cost += update_cost
            accumulator.tv_sum += float(feedback_tv.sum())
            accumulator.static_tv_sum += float(static_tv.sum())
            accumulator.feedback_rows += speculative_tokens
            if update_report is not None:
                accumulator.updates += 1
                accumulator.items_updated += update_report.items

        if generated >= next_trace_boundary:
            trace_summary = trace_accumulator.summary()
            trace.append(
                {
                    "token_end": generated,
                    "segment": segment_name,
                    "current_stride": (
                        controller.current_stride
                        if controller is not None
                        else (strategy.stride or 0)
                    ),
                    **trace_summary,
                }
            )
            trace_accumulator = _Accumulator()
            while next_trace_boundary <= generated:
                next_trace_boundary += trace_window_tokens

    if trace_accumulator.rounds:
        trace.append(
            {
                "token_end": generated,
                "segment": schedule.segment_at(max(0, generated - 1)),
                "current_stride": (
                    controller.current_stride
                    if controller is not None
                    else (strategy.stride or 0)
                ),
                **trace_accumulator.summary(),
            }
        )

    summary = total.summary()
    summary.update(
        {
            "negative_log_likelihood_per_token": negative_log_likelihood / generated,
            "mean_update_tv_improvement": (
                float(
                    np.mean(
                        [report.mean_tv_before - report.mean_tv_after for report in update_reports]
                    )
                )
                if update_reports
                else 0.0
            ),
            "mean_update_kl_improvement": (
                float(
                    np.mean(
                        [
                            report.mean_forward_kl_before
                            - report.mean_forward_kl_after
                            for report in update_reports
                        ]
                    )
                )
                if update_reports
                else 0.0
            ),
            "final_fast_weight_l2": float(
                np.linalg.norm(drafter.logits - np.log(drafter.offline_probabilities))
            ),
            "final_stride": (
                controller.current_stride
                if controller is not None
                else (strategy.stride or 0)
            ),
        }
    )
    ordered_segments = {}
    for name in schedule.segment_bounds():
        ordered_segments[name] = segments[name].summary()
    return {
        "strategy": asdict(strategy),
        "seed": seed,
        "summary": summary,
        "segments": ordered_segments,
        "controller_events": controller_events,
        "trace": trace,
    }


def aggregate_outcomes(
    outcomes: Sequence[dict[str, object]],
    *,
    cost_model: CostModel,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for outcome in outcomes:
        seed = int(outcome["seed"])
        label = str(outcome["strategy"]["name"])
        by_seed[seed][label] = outcome
    labels = [str(outcome["strategy"]["name"]) for outcome in outcomes]
    labels = list(dict.fromkeys(labels))
    if "static" not in labels:
        raise ValueError("outcomes require a static baseline.")
    if any(set(methods) != set(labels) for methods in by_seed.values()):
        raise ValueError("every seed must contain every strategy.")

    aggregate: dict[str, object] = {}
    for label_index, label in enumerate(labels):
        summaries = [methods[label]["summary"] for methods in by_seed.values()]
        tpf = np.asarray([summary["tpf"] for summary in summaries], dtype=np.float64)
        efficiency = np.asarray(
            [summary["tokens_per_cost"] for summary in summaries],
            dtype=np.float64,
        )
        regret_ratio = np.asarray(
            [
                summary["dynamic_regret_proxy"] / summary["static_regret_proxy"]
                for summary in summaries
            ],
            dtype=np.float64,
        )
        seed_offset = bootstrap_seed + 100_000 * label_index
        method_result: dict[str, object] = {
            "seeds": len(summaries),
            "tpf": bootstrap_interval(
                tpf,
                samples=bootstrap_samples,
                seed=seed_offset + 1,
            ),
            "tokens_per_cost": bootstrap_interval(
                efficiency,
                samples=bootstrap_samples,
                seed=seed_offset + 2,
            ),
            "dynamic_to_static_regret_ratio": bootstrap_interval(
                regret_ratio,
                samples=bootstrap_samples,
                seed=seed_offset + 3,
            ),
            "median_updates": float(np.median([summary["updates"] for summary in summaries])),
            "median_update_cost_fraction": float(
                np.median([summary["update_cost_fraction"] for summary in summaries])
            ),
            "median_spec_acceptance_rate": float(
                np.median([summary["spec_acceptance_rate"] for summary in summaries])
            ),
        }
        if label != "static":
            static_efficiency = np.asarray(
                [methods["static"]["summary"]["tokens_per_cost"] for methods in by_seed.values()],
                dtype=np.float64,
            )
            static_tpf = np.asarray(
                [methods["static"]["summary"]["tpf"] for methods in by_seed.values()],
                dtype=np.float64,
            )
            efficiency_ratio = efficiency / static_efficiency
            tpf_ratio = tpf / static_tpf
            method_result["paired_efficiency_ratio_over_static"] = bootstrap_interval(
                efficiency_ratio,
                samples=bootstrap_samples,
                seed=seed_offset + 4,
            )
            method_result["paired_tpf_ratio_over_static"] = bootstrap_interval(
                tpf_ratio,
                samples=bootstrap_samples,
                seed=seed_offset + 5,
            )
            break_even_multipliers = []
            sensitivity: dict[str, list[float]] = {
                str(multiplier): [] for multiplier in (0.0, 0.5, 1.0, 2.0, 4.0)
            }
            for summary, static_value in zip(summaries, static_efficiency):
                base_cost = float(summary["rounds"]) * cost_model.forward_pair_cost
                update_cost = float(summary["update_cost"])
                allowed_cost = float(summary["tokens"]) / static_value - base_cost
                break_even_multipliers.append(
                    max(0.0, allowed_cost / update_cost) if update_cost else math.inf
                )
                for multiplier in (0.0, 0.5, 1.0, 2.0, 4.0):
                    counterfactual_efficiency = float(summary["tokens"]) / (
                        base_cost + multiplier * update_cost
                    )
                    sensitivity[str(multiplier)].append(
                        counterfactual_efficiency / static_value
                    )
            finite_break_even = np.asarray(
                [value for value in break_even_multipliers if math.isfinite(value)],
                dtype=np.float64,
            )
            method_result["median_update_cost_break_even_multiplier"] = (
                float(np.median(finite_break_even)) if finite_break_even.size else None
            )
            method_result["update_cost_sensitivity_median_efficiency_ratio"] = {
                multiplier: float(np.median(values))
                for multiplier, values in sensitivity.items()
            }

        segment_results = {}
        for segment in next(iter(by_seed.values()))[label]["segments"]:
            segment_efficiency = np.asarray(
                [methods[label]["segments"][segment]["tokens_per_cost"] for methods in by_seed.values()],
                dtype=np.float64,
            )
            static_segment_efficiency = np.asarray(
                [
                    methods["static"]["segments"][segment]["tokens_per_cost"]
                    for methods in by_seed.values()
                ],
                dtype=np.float64,
            )
            segment_results[segment] = {
                "tokens_per_cost_median": float(np.median(segment_efficiency)),
                "paired_efficiency_ratio_over_static_median": float(
                    np.median(segment_efficiency / static_segment_efficiency)
                ),
            }
        method_result["segments"] = segment_results
        aggregate[label] = method_result

    online_labels = [label for label in labels if label != "static"]
    primary_label = "stride10_discounted"
    if primary_label not in online_labels:
        raise ValueError(f"primary strategy {primary_label!r} is missing.")
    best_label = max(
        online_labels,
        key=lambda label: aggregate[label]["paired_efficiency_ratio_over_static"][
            "estimate"
        ],
    )
    return {
        "methods": aggregate,
        "decision": {
            "preregistered_primary_strategy": primary_label,
            "primary_learning_success": (
                aggregate[primary_label]["dynamic_to_static_regret_ratio"]["ci_95_high"]
                < 1.0
            ),
            "primary_tpf_success": (
                aggregate[primary_label]["paired_tpf_ratio_over_static"]["ci_95_low"]
                > 1.0
            ),
            "primary_proxy_system_success": (
                aggregate[primary_label]["paired_efficiency_ratio_over_static"][
                    "ci_95_low"
                ]
                > 1.0
            ),
            "exploratory_best_default_cost_strategy": best_label,
            "any_exploratory_learning_success": any(
                aggregate[label]["dynamic_to_static_regret_ratio"]["ci_95_high"] < 1.0
                for label in online_labels
            ),
            "any_exploratory_tpf_success": any(
                aggregate[label]["paired_tpf_ratio_over_static"]["ci_95_low"] > 1.0
                for label in online_labels
            ),
            "exploratory_best_proxy_system_success": (
                aggregate[best_label]["paired_efficiency_ratio_over_static"]["ci_95_low"]
                > 1.0
            ),
            "selection_warning": (
                "The exploratory best strategy is selected on the same seeds used for its "
                "interval; only the preregistered primary strategy is confirmatory."
            ),
            "real_gpu_online_speedup_tested": False,
        },
    }


def default_strategies() -> list[StrategyConfig]:
    return [
        StrategyConfig("static", stride=None),
        StrategyConfig("per_round_full", stride=1, supervision="full"),
        StrategyConfig("stride5_full", stride=5, supervision="full"),
        StrategyConfig("stride10_full", stride=10, supervision="full"),
        StrategyConfig("stride20_full", stride=20, supervision="full"),
        StrategyConfig("stride10_on_policy", stride=10, supervision="on_policy"),
        StrategyConfig(
            "stride10_discounted",
            stride=10,
            supervision="discounted_tail",
        ),
        StrategyConfig(
            "adaptive_discounted",
            stride=10,
            supervision="discounted_tail",
            adaptive=True,
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=12_000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=20260905)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--vocabulary-size", type=int, default=8)
    parser.add_argument("--trace-window-tokens", type=int, default=250)
    parser.add_argument("--bootstrap-samples", type=int, default=30_000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.seeds < 2:
        raise ValueError("at least two seeds are required for paired analysis.")
    if args.block_size < 2:
        raise ValueError("block_size must be at least two.")
    schedule = DriftSchedule.create(args.vocabulary_size, args.tokens)
    offline = offline_marginal_draft(
        schedule.regime_0,
        args.block_size - 1,
    )
    cost_model = CostModel()
    strategies = default_strategies()
    outcomes: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for seed_index in range(args.seeds):
        seed = args.first_seed + seed_index
        for strategy in strategies:
            outcome = simulate_strategy(
                schedule=schedule,
                offline_draft=offline,
                block_size=args.block_size,
                strategy=strategy,
                cost_model=cost_model,
                seed=seed,
                trace_window_tokens=args.trace_window_tokens,
            )
            # Full traces are retained for one representative paired seed. The
            # per-seed summary and per-segment metrics are retained for all runs.
            if seed_index > 0:
                outcome["trace"] = []
                outcome["controller_events"] = []
            outcomes.append(outcome)
            summary = outcome["summary"]
            print(
                f"seed={seed} strategy={strategy.name} "
                f"TPF={summary['tpf']:.3f} efficiency={summary['tokens_per_cost']:.3f} "
                f"updates={summary['updates']}",
                flush=True,
            )

    aggregate = aggregate_outcomes(
        outcomes,
        cost_model=cost_model,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.first_seed,
    )
    result = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": "nonstationary_markov_online_uno",
        "claim_scope": {
            "sampling": "exact NumPy Psi-Spec with post-verification updates",
            "learning": "request-local tabular logit fast weights",
            "cost": (
                "forward-equivalent proxy calibrated from Stage-2 static B=8; "
                "update costs are synthetic sensitivity parameters"
            ),
            "gpu_timing": False,
        },
        "configuration": {
            "tokens": args.tokens,
            "seeds": args.seeds,
            "first_seed": args.first_seed,
            "block_size": args.block_size,
            "vocabulary_size": args.vocabulary_size,
            "trace_window_tokens": args.trace_window_tokens,
            "bootstrap_samples": args.bootstrap_samples,
            "cost_model": asdict(cost_model),
            "cost_calibration": {
                "source": "Stage-2 Uno-1B HF fallback B=8 paired medians",
                "formula": "forward_pair_cost = 2 * TPF / decode_speedup",
                "stage2_tpf": 1.4006916996047432,
                "stage2_decode_speedup": 1.3523271075533483,
            },
            "schedule_bounds": schedule.segment_bounds(),
            "strategies": [asdict(strategy) for strategy in strategies],
        },
        "target_matrices": {
            "regime_0": schedule.regime_0.tolist(),
            "regime_1": schedule.regime_1.tolist(),
            "regime_2": schedule.regime_2.tolist(),
        },
        "outcomes": outcomes,
        "aggregate": aggregate,
        "elapsed_seconds": time.perf_counter() - start_time,
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

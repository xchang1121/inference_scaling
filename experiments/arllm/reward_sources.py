"""Rewards selected by source name for the AR search methods.

Algorithms receive only ``reward(prompt, completion)`` or a batched callable.
This module maps each ``--reward`` source to that callable, keeping dataset
access (answer parsing, verifier references) out of the algorithm layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from experiments.arllm.common import answer_counts, configured_verifier_reward, fraction_text
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.reward_factory import model_reward_from_config
from inference_scaling.arllm.rewards import ConsilienceReward, SequenceLogProbabilityReward
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, TokenSequence
from inference_scaling.shared.evaluation import (
    CumulativeConsensusReward,
    GSM8KProblem,
    extract_numeric_answer,
    modal_answer,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import TokenVerifierReward

REWARD_SOURCES = (
    "self_consistency",
    "frozen_consensus",
    "log_probability",
    "sequence_log_probability",
    "negative_entropy",
    "self_certainty",
    "consilience",
    "verifier",
)
# Per-decision min-max normalized confidence statistics (batch-coupled rewards).
NORMALIZED_CONFIDENCE_SOURCES = frozenset({"log_probability", "negative_entropy", "self_certainty"})
REWARD_TARGET_NAMES = {
    "self_consistency": "cumulative_consensus",
    "frozen_consensus": "independent_pilot_frozen_consensus",
    "log_probability": "normalized_mean_log_probability",
    "sequence_log_probability": "model_sequence_log_probability",
    "negative_entropy": "normalized_negative_entropy",
    "self_certainty": "normalized_self_certainty",
    "consilience": "model_consilience",
    "verifier": "configured_verifier",
}


def minmax_rewards(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize confidence rewards within one decision batch."""

    if not values:
        raise ValueError("reward normalization requires at least one value")
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-12):
        return (0.0,) * len(values)
    scale = upper - lower
    return tuple((float(value) - lower) / scale for value in values)


def confidence_rewards(
    backend: Any,
    prompt: TokenSequence,
    sequences: Sequence[TokenSequence],
    *,
    sampling: SamplingConfig,
    source: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    statistics_batch = backend.score_statistics_batch(
        [ScoreRequest(prompt, tuple(sequences), sampling)]
    )
    statistic = {
        "log_probability": "mean_logprob",
        "negative_entropy": "mean_negative_entropy",
        "self_certainty": "mean_self_certainty",
    }.get(source)
    if statistic is None:
        raise ValueError(f"{source!r} is not a confidence reward")
    raw = tuple(float(getattr(item, statistic)) for item in statistics_batch)
    return raw, minmax_rewards(raw)


def consilience_reward(
    backend: Any,
    sampling: SamplingConfig,
    config: dict[str, Any],
) -> ConsilienceReward:
    reward = model_reward_from_config(backend, config, source="consilience", sampling=sampling)
    assert isinstance(reward, ConsilienceReward)
    return reward


def frozen_consensus_reward(
    backend: Any,
    prompt: TokenSequence,
    *,
    maximum: int,
    sampling: SamplingConfig,
    samples: int,
    seeds: SeedStream,
    problem_index: int,
) -> tuple[Callable[[TokenSequence, TokenSequence], float], dict[str, Any]]:
    """Construct a pointwise reward from an independent base-model pilot pool."""

    if samples <= 0:
        raise ValueError("frozen-consensus pilot_samples must be positive")
    requests = [
        GenerationRequest(
            prompt,
            maximum,
            sampling,
            seeds.derive("frozen_consensus", problem_index, pilot_index),
            f"frozen-consensus:{problem_index}:pilot:{pilot_index}",
        )
        for pilot_index in range(samples)
    ]
    pilots = backend.sample_batch(requests)
    answers = [
        extract_numeric_answer(backend.decode(sample.token_ids)) for sample in pilots
    ]
    reference = modal_answer(answers)

    def reward(_prompt: TokenSequence, generated: TokenSequence) -> float:
        prediction = extract_numeric_answer(backend.decode(generated))
        return float(reference is not None and prediction == reference)

    return reward, {
        "pilot_samples": samples,
        "pilot_answer_counts": answer_counts(answers),
        "frozen_consensus_answer": fraction_text(reference),
    }


@dataclass
class SearchReward:
    """The reward callables and provenance of one conditional-IS run."""

    pointwise: Callable[[TokenSequence, TokenSequence], float] | None = None
    batch: Callable[[TokenSequence, Sequence[TokenSequence]], Sequence[float]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    model_reward: ConsilienceReward | SequenceLogProbabilityReward | None = None
    verifier: TokenVerifierReward | None = None


def conditional_search_reward(
    *,
    method: str,
    source: str,
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    config: dict[str, Any],
    seeds: SeedStream,
    maximum: int,
    sampling: SamplingConfig,
) -> SearchReward:
    """Build the reward of a conditional or iterated conditional IS method.

    Iterated IS needs a fixed pointwise reward; the other conditional methods
    use batched model rewards, which keep the same per-sequence definition.
    """

    iterated = method == "iterated_conditional_is"
    reward = SearchReward()
    if source == "self_consistency":
        if iterated:
            raise ValueError(
                "iterated_conditional_is requires a fixed pointwise reward; "
                "use frozen_consensus, sequence_log_probability, Consilience, "
                "or verifier"
            )
        reward.batch = CumulativeConsensusReward(backend.decode)
    elif source == "frozen_consensus":
        reward.pointwise, reward.diagnostics = frozen_consensus_reward(
            backend,
            prompt,
            maximum=maximum,
            sampling=sampling,
            samples=int(config.get("iterated_is", {}).get("pilot_samples", 8)),
            seeds=seeds,
            problem_index=problem.index,
        )
    elif source in {"sequence_log_probability", "consilience"}:
        model_reward = (
            consilience_reward(backend, sampling, config)
            if source == "consilience"
            else model_reward_from_config(
                backend, config, source="sequence_log_probability", sampling=sampling,
            )
        )
        reward.model_reward = model_reward
        if iterated:
            reward.pointwise = model_reward
        else:
            reward.batch = model_reward.batch
        reward.diagnostics["model_reward"] = model_reward.describe()
    elif source != "verifier":
        if iterated:
            raise ValueError(
                "iterated_conditional_is currently accepts only fixed pointwise "
                "frozen_consensus, sequence_log_probability, Consilience, or "
                "verifier rewards"
            )

        def normalized_confidence(
            reward_prompt: TokenSequence,
            generated_sequences: Sequence[TokenSequence],
        ) -> tuple[float, ...]:
            _, normalized = confidence_rewards(
                backend, reward_prompt, generated_sequences, sampling=sampling, source=source,
            )
            return normalized

        reward.batch = normalized_confidence
    if source == "verifier":
        reward.verifier = configured_verifier_reward(backend, problem, config)
        reward.pointwise = reward.verifier
        reward.diagnostics["verifier"] = reward.verifier.describe()
    return reward


__all__ = [
    "NORMALIZED_CONFIDENCE_SOURCES",
    "REWARD_SOURCES",
    "REWARD_TARGET_NAMES",
    "SearchReward",
    "conditional_search_reward",
    "confidence_rewards",
    "consilience_reward",
    "frozen_consensus_reward",
    "minmax_rewards",
]

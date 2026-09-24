"""Every sequence reward by source name, independent of any dataset.

A dataset enters a reward only through what the caller binds: an answer rule
and the text that carries the answer (answer-agreement rewards), pilot texts
(frozen rewards) and a verifier bound to its reference (``verifier``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import thinking_format_from_backend, output_settings_from_config
from inference_scaling.arllm.rewards.intrinsic import (
    ConsilienceReward,
    SequenceLogProbabilityReward,
    confidence_rewards,
)
from inference_scaling.shared.rewards.consensus import (
    AnswerRule,
    CumulativeConsensusReward,
    Decode,
    frozen_consensus_reward,
    pilot_agreement_reward,
)
from inference_scaling.shared.rewards.verifier import TokenBatchReward, TokenReward, TokenVerifierReward


MODEL_REWARD_SOURCES = ("consilience", "sequence_log_probability")
# Source name -> the target named in diagnostics.
REWARD_TARGET_NAMES = {
    "self_consistency": "cumulative_consensus",
    "frozen_consensus": "independent_pilot_frozen_consensus",
    "pilot_agreement": "independent_pilot_agreement",
    "log_probability": "normalized_mean_log_probability",
    "sequence_log_probability": "model_sequence_log_probability",
    "negative_entropy": "normalized_negative_entropy",
    "self_certainty": "normalized_self_certainty",
    "consilience": "model_consilience",
    "verifier": "configured_verifier",
}
REWARD_SOURCES = tuple(REWARD_TARGET_NAMES)
# Confidence statistics min-max normalized within each decision batch.
NORMALIZED_CONFIDENCE_SOURCES = frozenset({"log_probability", "negative_entropy", "self_certainty"})
# Rewards whose value for a sequence depends on the other sequences scored with it.
BATCH_DEPENDENT_SOURCES = NORMALIZED_CONFIDENCE_SOURCES | {"self_consistency"}
# Rewards that compare a sequence with independently sampled, fixed pilot texts.
PILOT_SOURCES = frozenset({"frozen_consensus", "pilot_agreement"})


@dataclass
class SequenceReward:
    """One reward source bound to a problem: what algorithms call and its provenance."""

    pointwise: TokenReward | None = None
    batch: TokenBatchReward | None = None
    model_reward: ConsilienceReward | SequenceLogProbabilityReward | None = None
    verifier: TokenVerifierReward | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_reward(
    source: str,
    *,
    backend: Any,
    config: Mapping[str, Any],
    sampling: SamplingConfig | None,
    rule: AnswerRule | None = None,
    decode: Decode | None = None,
    pilots: Sequence[str] = (),
    verifier: TokenVerifierReward | None = None,
) -> SequenceReward:
    """Bind one reward source; model rewards expose both a pointwise and a batch call."""

    if source in {"self_consistency", *PILOT_SOURCES}:
        if rule is None or decode is None:
            raise ValueError(f"{source} requires an answer rule and a decoder")
        if source == "self_consistency":
            return SequenceReward(batch=CumulativeConsensusReward(rule, decode))
        build = frozen_consensus_reward if source == "frozen_consensus" else pilot_agreement_reward
        return SequenceReward(pointwise=build(rule, decode, pilots))
    if source in NORMALIZED_CONFIDENCE_SOURCES:
        if sampling is None:
            raise ValueError(f"{source} requires the sampling policy it scores")
        confidence = partial(confidence_rewards, backend, sampling=sampling, source=source)
        return SequenceReward(batch=lambda prompt, sequences: confidence(prompt, sequences)[1])
    if source == "verifier":
        if verifier is None:
            raise ValueError("the verifier reward requires a verifier bound to its reference")
        return SequenceReward(pointwise=verifier, verifier=verifier,
                              diagnostics={"verifier": verifier.describe()})
    if source not in MODEL_REWARD_SOURCES:
        raise ValueError(f"unknown reward source {source!r}")
    model = model_reward_from_config(backend, config, source=source, sampling=sampling)
    return SequenceReward(pointwise=model, batch=model.batch, model_reward=model,
                          diagnostics={"model_reward": model.describe()})


def model_reward_from_config(
    backend: Any,
    config: Mapping[str, Any],
    *,
    source: str,
    sampling: SamplingConfig | None = None,
) -> ConsilienceReward | SequenceLogProbabilityReward:
    common = config.get("reward", {})
    if source == "sequence_log_probability":
        return SequenceLogProbabilityReward(backend, sampling, scale=float(common.get("logprob_scale", 1.0)))
    if source != "consilience":
        raise ValueError(f"unknown model reward source: {source}")
    options = common.get("consilience", {})
    setting = options.get
    output = output_settings_from_config(config)
    scope = str(options.get("scope", "thinking"))
    if scope not in {"thinking", "full"}:
        raise ValueError("Consilience scope must be thinking or full")
    format_ = (
        thinking_format_from_backend(backend, output)
        if scope == "thinking" else None
    )
    window_tokens = setting("window_tokens", None)
    return ConsilienceReward(
        backend,
        # Confidence is a fixed property of the reference model, independent
        # of the proposal policy used to obtain the sequence.
        SamplingConfig(temperature=float(options.get("score_temperature", 1.0))),
        top_k=int(setting("top_k", 5)),
        window_fraction=float(setting("window_fraction", 0.2)),
        window_tokens=None if window_tokens is None else int(window_tokens),
        skip_fraction=float(setting("skip_fraction", 0.05)),
        initial_penalty=float(setting("initial_penalty", 3.0)),
        scale=float(setting("scale", 1.0)),
        thinking_format=format_,
        scope="thinking" if scope == "thinking" else "full",
    )


def reward_temperature_from_config(
    config: Mapping[str, Any], *, source: str, default: float = 1.0
) -> float:
    if "temperature" in config.get("reward", {}):
        return float(config["reward"]["temperature"])
    if source == "consilience":
        return 2.0
    return float(config.get("conditional_is", {}).get("reward_temperature", default))


__all__ = [
    "BATCH_DEPENDENT_SOURCES",
    "MODEL_REWARD_SOURCES",
    "NORMALIZED_CONFIDENCE_SOURCES",
    "PILOT_SOURCES",
    "REWARD_SOURCES",
    "REWARD_TARGET_NAMES",
    "SequenceReward",
    "build_reward",
    "model_reward_from_config",
    "reward_temperature_from_config",
]

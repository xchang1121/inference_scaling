"""Bounded verifier-replay drafts and a past-feedback-only cost router.

The cache stores continuations that a target model has already verified.  A
cache proposal is still checked by the target model; it is never returned as a
response merely because it was observed before.  Consequently a stale or
wrong replay hurts efficiency, not the target distribution, provided the
verifier uses the actual saved replay proposal distribution.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


TokenTuple = tuple[int, ...]
FloatMatrix = NDArray[np.float64]


def _tokens(values: Sequence[int], *, name: str, allow_empty: bool) -> TokenTuple:
    tokens = tuple(int(value) for value in values)
    if not allow_empty and not tokens:
        raise ValueError(f"{name} cannot be empty")
    if any(value < 0 for value in tokens):
        raise ValueError(f"{name} cannot contain negative token IDs")
    return tokens


@dataclass(frozen=True)
class ReplayCacheConfig:
    min_suffix_length: int = 8
    max_suffix_length: int = 32
    max_continuation_length: int = 16
    max_entries: int = 100_000
    max_alternatives_per_key: int = 4
    min_observations: int = 1
    min_confidence: float = 0.75

    def validate(self) -> None:
        if self.min_suffix_length < 1:
            raise ValueError("min_suffix_length must be positive")
        if self.max_suffix_length < self.min_suffix_length:
            raise ValueError("max_suffix_length cannot be smaller than the minimum")
        if self.max_continuation_length < 1:
            raise ValueError("max_continuation_length must be positive")
        if self.max_entries < 1 or self.max_alternatives_per_key < 1:
            raise ValueError("cache bounds must be positive")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in (0, 1]")


@dataclass(frozen=True)
class ReplayCandidate:
    token_ids: TokenTuple
    matched_suffix_length: int
    observations: int
    total_observations: int
    confidence: float
    namespace: str


@dataclass(frozen=True)
class ReplayCacheStats:
    namespace: str
    entries: int
    alternatives: int
    observations: int
    evicted_entries: int
    evicted_alternatives: int


class VerifierReplayCache:
    """Exact-suffix continuation cache populated only from verified outputs."""

    def __init__(self, *, namespace: str, config: ReplayCacheConfig) -> None:
        if not namespace.strip():
            raise ValueError("cache namespace cannot be empty")
        config.validate()
        self.namespace = namespace
        self.config = config
        self._entries: OrderedDict[TokenTuple, Counter[TokenTuple]] = OrderedDict()
        self._observations = 0
        self._evicted_entries = 0
        self._evicted_alternatives = 0

    def _record(self, key: TokenTuple, continuation: TokenTuple) -> None:
        counter = self._entries.get(key)
        if counter is None:
            counter = Counter()
            self._entries[key] = counter
        else:
            self._entries.move_to_end(key)
        counter[continuation] += 1
        self._observations += 1

        if len(counter) > self.config.max_alternatives_per_key:
            victim = min(
                counter,
                key=lambda candidate: (
                    counter[candidate],
                    len(candidate),
                    candidate,
                ),
            )
            del counter[victim]
            self._evicted_alternatives += 1

        while len(self._entries) > self.config.max_entries:
            self._entries.popitem(last=False)
            self._evicted_entries += 1

    def observe_sequence(
        self,
        *,
        prompt_tokens: Sequence[int],
        verified_completion_tokens: Sequence[int],
    ) -> int:
        """Add target-verified continuations after a request has completed.

        No item from an unfinished response is visible to ``lookup`` unless the
        caller explicitly closes that response by invoking this method.
        """

        prompt = _tokens(prompt_tokens, name="prompt_tokens", allow_empty=False)
        completion = _tokens(
            verified_completion_tokens,
            name="verified_completion_tokens",
            allow_empty=False,
        )
        sequence = prompt + completion
        records = 0
        for start in range(len(prompt), len(sequence)):
            continuation = sequence[
                start : start + self.config.max_continuation_length
            ]
            maximum = min(self.config.max_suffix_length, start)
            for suffix_length in range(self.config.min_suffix_length, maximum + 1):
                key = sequence[start - suffix_length : start]
                self._record(key, continuation)
                records += 1
        return records

    def lookup(
        self,
        context_tokens: Sequence[int],
        *,
        max_tokens: int,
        min_tokens: int = 1,
    ) -> ReplayCandidate | None:
        """Return the most frequent continuation at the longest eligible suffix."""

        context = _tokens(context_tokens, name="context_tokens", allow_empty=False)
        if max_tokens < 1 or min_tokens < 1 or min_tokens > max_tokens:
            raise ValueError("lookup token bounds are invalid")
        maximum = min(self.config.max_suffix_length, len(context))
        for suffix_length in range(maximum, self.config.min_suffix_length - 1, -1):
            key = context[-suffix_length:]
            counter = self._entries.get(key)
            if not counter:
                continue
            eligible = [
                (continuation, count)
                for continuation, count in counter.items()
                if len(continuation) >= min_tokens
            ]
            if not eligible:
                continue
            continuation, count = max(
                eligible,
                key=lambda item: (item[1], len(item[0]), item[0]),
            )
            total = sum(counter.values())
            confidence = count / total
            if count < self.config.min_observations:
                continue
            if confidence + 1e-15 < self.config.min_confidence:
                continue
            self._entries.move_to_end(key)
            return ReplayCandidate(
                token_ids=continuation[:max_tokens],
                matched_suffix_length=suffix_length,
                observations=count,
                total_observations=total,
                confidence=confidence,
                namespace=self.namespace,
            )
        return None

    def clear(self) -> None:
        self._entries.clear()

    def begin_causal_session(
        self,
        *,
        prompt_tokens: Sequence[int],
    ) -> CausalVerifierReplaySession:
        """Create a request-local overlay populated only by verified past tokens."""

        return CausalVerifierReplaySession(
            global_cache=self,
            prompt_tokens=prompt_tokens,
        )

    def _merge_from(self, source: VerifierReplayCache) -> int:
        """Merge a closed request-local overlay without exposing it early."""

        if source is self:
            raise ValueError("cannot merge a replay cache into itself")
        if source.namespace != self.namespace or source.config != self.config:
            raise ValueError("replay cache merge requires identical namespace and config")
        records = 0
        for key, counter in source._entries.items():
            for continuation, count in counter.items():
                for _ in range(count):
                    self._record(key, continuation)
                    records += 1
        return records

    def stats(self) -> ReplayCacheStats:
        return ReplayCacheStats(
            namespace=self.namespace,
            entries=len(self._entries),
            alternatives=sum(len(counter) for counter in self._entries.values()),
            observations=self._observations,
            evicted_entries=self._evicted_entries,
            evicted_alternatives=self._evicted_alternatives,
        )


class CausalVerifierReplaySession:
    """Request-local cache overlay with a strict verified-past publication lag.

    The global cache remains unchanged until :meth:`close` publishes the
    overlay.  During the request, a continuation becomes locally visible only
    after all of its tokens have already been committed by the target model.
    Thus an in-flight request can exploit repetitions in its own verified past
    without revealing partial request state to concurrent requests.
    """

    def __init__(
        self,
        *,
        global_cache: VerifierReplayCache,
        prompt_tokens: Sequence[int],
    ) -> None:
        self.global_cache = global_cache
        self.prompt_tokens = _tokens(
            prompt_tokens,
            name="prompt_tokens",
            allow_empty=False,
        )
        self._local_cache = VerifierReplayCache(
            namespace=global_cache.namespace,
            config=global_cache.config,
        )
        self._sequence = list(self.prompt_tokens)
        self._next_start = len(self.prompt_tokens)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def verified_completion_tokens(self) -> int:
        return len(self._sequence) - len(self.prompt_tokens)

    @property
    def local_records(self) -> int:
        return self._local_cache.stats().observations

    def _record_start(self, start: int) -> int:
        continuation = tuple(
            self._sequence[
                start : start + self.global_cache.config.max_continuation_length
            ]
        )
        if not continuation:
            return 0
        maximum = min(self.global_cache.config.max_suffix_length, start)
        records = 0
        for suffix_length in range(
            self.global_cache.config.min_suffix_length,
            maximum + 1,
        ):
            key = tuple(self._sequence[start - suffix_length : start])
            self._local_cache._record(key, continuation)
            records += 1
        return records

    def append_verified(self, token_ids: Sequence[int]) -> int:
        """Append committed target tokens and publish only full local horizons."""

        if self._closed:
            raise RuntimeError("cannot append to a closed causal replay session")
        tokens = _tokens(
            token_ids,
            name="verified_token_ids",
            allow_empty=False,
        )
        self._sequence.extend(tokens)
        records = 0
        horizon = self.global_cache.config.max_continuation_length
        while self._next_start + horizon <= len(self._sequence):
            records += self._record_start(self._next_start)
            self._next_start += 1
        return records

    def lookup(
        self,
        context_tokens: Sequence[int],
        *,
        max_tokens: int,
        min_tokens: int = 1,
    ) -> ReplayCandidate | None:
        """Query the private verified-past overlay and the closed global cache."""

        if self._closed:
            raise RuntimeError("cannot query a closed causal replay session")
        local = self._local_cache.lookup(
            context_tokens,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        global_candidate = self.global_cache.lookup(
            context_tokens,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        if local is None:
            return global_candidate
        if global_candidate is None:
            return local
        return max(
            (local, global_candidate),
            key=lambda candidate: (
                candidate.matched_suffix_length,
                candidate.observations,
                candidate.confidence,
                len(candidate.token_ids),
                candidate.token_ids,
            ),
        )

    def close(self, *, publish: bool = True) -> int:
        """Seal tail continuations and optionally atomically publish the overlay."""

        if self._closed:
            raise RuntimeError("causal replay session is already closed")
        while self._next_start < len(self._sequence):
            self._record_start(self._next_start)
            self._next_start += 1
        self._closed = True
        if not publish:
            return 0
        return self.global_cache._merge_from(self._local_cache)


@dataclass(frozen=True)
class ReplayRouteConfig:
    min_match_length: int = 8
    min_proposal_tokens: int = 2
    min_cache_confidence: float = 0.75
    exploration_trials_per_match_length: int = 1
    probe_interval: int = 64
    ema_decay: float = 0.9
    throughput_margin: float = 0.02

    def validate(self) -> None:
        if self.min_match_length < 1 or self.min_proposal_tokens < 1:
            raise ValueError("route length thresholds must be positive")
        if not 0.0 < self.min_cache_confidence <= 1.0:
            raise ValueError("min_cache_confidence must lie in (0, 1]")
        if self.exploration_trials_per_match_length < 0:
            raise ValueError("exploration trials cannot be negative")
        if self.probe_interval < 1:
            raise ValueError("probe_interval must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if self.throughput_margin < 0.0:
            raise ValueError("throughput_margin cannot be negative")


@dataclass(frozen=True)
class ReplayRouteDecision:
    use_replay: bool
    reason: str
    static_tokens_per_forward: float | None
    replay_tokens_per_forward: float | None
    match_length: int
    decision_index: int


@dataclass
class _RouteBucket:
    trials: int = 0
    replay_tpf_ema: float | None = None
    last_probe_decision: int = 0


class CostAwareReplayRouter:
    """Choose one-forward replay or two-forward Uno using only past timings."""

    def __init__(self, *, namespace: str, config: ReplayRouteConfig) -> None:
        if not namespace.strip():
            raise ValueError("router namespace cannot be empty")
        config.validate()
        self.namespace = namespace
        self.config = config
        self._static_tpf_ema: float | None = None
        self._buckets: dict[int, _RouteBucket] = {}
        self._decision_index = 0

    def _ema(self, old: float | None, value: float) -> float:
        if old is None:
            return value
        return self.config.ema_decay * old + (1.0 - self.config.ema_decay) * value

    def observe_static(self, *, committed_tokens: int, forwards: int = 2) -> None:
        if committed_tokens < 1 or forwards < 1:
            raise ValueError("static observations must have positive work")
        value = committed_tokens / forwards
        self._static_tpf_ema = self._ema(self._static_tpf_ema, value)

    def observe_replay(
        self,
        *,
        matched_suffix_length: int,
        committed_tokens: int,
        forwards: int = 1,
    ) -> None:
        if matched_suffix_length < 1 or committed_tokens < 1 or forwards < 1:
            raise ValueError("replay observations must have positive work")
        bucket = self._buckets.setdefault(matched_suffix_length, _RouteBucket())
        bucket.trials += 1
        bucket.replay_tpf_ema = self._ema(
            bucket.replay_tpf_ema,
            committed_tokens / forwards,
        )

    def decide(self, candidate: ReplayCandidate) -> ReplayRouteDecision:
        if candidate.namespace != self.namespace:
            raise ValueError("candidate and router namespaces differ")
        self._decision_index += 1
        bucket = self._buckets.setdefault(
            candidate.matched_suffix_length,
            _RouteBucket(),
        )

        use_replay = False
        reason = "ineligible"
        if self._static_tpf_ema is None:
            reason = "static-uninitialized"
        elif candidate.matched_suffix_length < self.config.min_match_length:
            reason = "short-match"
        elif len(candidate.token_ids) < self.config.min_proposal_tokens:
            reason = "short-proposal"
        elif candidate.confidence + 1e-15 < self.config.min_cache_confidence:
            reason = "low-confidence"
        elif bucket.trials < self.config.exploration_trials_per_match_length:
            use_replay = True
            reason = "explore"
        elif (
            bucket.replay_tpf_ema is not None
            and bucket.replay_tpf_ema
            >= self._static_tpf_ema * (1.0 + self.config.throughput_margin)
        ):
            use_replay = True
            reason = "exploit"
        elif self._decision_index - bucket.last_probe_decision >= self.config.probe_interval:
            use_replay = True
            reason = "periodic-probe"

        if use_replay:
            bucket.last_probe_decision = self._decision_index
        return ReplayRouteDecision(
            use_replay=use_replay,
            reason=reason,
            static_tokens_per_forward=self._static_tpf_ema,
            replay_tokens_per_forward=bucket.replay_tpf_ema,
            match_length=candidate.matched_suffix_length,
            decision_index=self._decision_index,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "config": asdict(self.config),
            "decision_index": self._decision_index,
            "static_tokens_per_forward_ema": self._static_tpf_ema,
            "buckets": {
                str(match_length): asdict(bucket)
                for match_length, bucket in sorted(self._buckets.items())
            },
        }


@dataclass(frozen=True)
class GreedyReplayVerification:
    accepted_count: int
    committed_tokens: TokenTuple
    rejection_index: int | None
    correction_token: int | None
    lookahead_token: int | None

    @property
    def all_accepted(self) -> bool:
        return self.rejection_index is None


def verify_greedy_replay(
    proposal_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    lookahead_token: int,
) -> GreedyReplayVerification:
    """Verify deterministic replay drafts against target argmax tokens."""

    proposals = _tokens(
        proposal_tokens,
        name="proposal_tokens",
        allow_empty=False,
    )
    targets = _tokens(target_tokens, name="target_tokens", allow_empty=False)
    if len(proposals) != len(targets):
        raise ValueError("one target token is required per replay proposal")
    if int(lookahead_token) < 0:
        raise ValueError("lookahead_token cannot be negative")
    for index, (proposal, target) in enumerate(zip(proposals, targets)):
        if proposal != target:
            return GreedyReplayVerification(
                accepted_count=index,
                committed_tokens=proposals[:index] + (target,),
                rejection_index=index,
                correction_token=target,
                lookahead_token=None,
            )
    return GreedyReplayVerification(
        accepted_count=len(proposals),
        committed_tokens=proposals + (int(lookahead_token),),
        rejection_index=None,
        correction_token=None,
        lookahead_token=int(lookahead_token),
    )


def delta_draft_probabilities(
    proposal_tokens: Sequence[int],
    *,
    vocabulary_size: int,
) -> FloatMatrix:
    """Materialize the saved categorical law of deterministic cache drafts."""

    proposals = _tokens(
        proposal_tokens,
        name="proposal_tokens",
        allow_empty=False,
    )
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least two")
    if any(token >= vocabulary_size for token in proposals):
        raise ValueError("proposal token lies outside the vocabulary")
    probabilities = np.zeros((len(proposals), vocabulary_size), dtype=np.float64)
    probabilities[np.arange(len(proposals)), proposals] = 1.0
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ):
        raise RuntimeError("failed to construct deterministic replay distributions")
    return probabilities


def independent_match_tpf(*, token_match_probability: float, proposals: int) -> float:
    """One-forward replay TPF under an explicit i.i.d. match approximation."""

    if not 0.0 <= token_match_probability <= 1.0:
        raise ValueError("token_match_probability must lie in [0, 1]")
    if proposals < 1:
        raise ValueError("proposals must be positive")
    probability = float(token_match_probability)
    if math.isclose(probability, 1.0):
        return float(proposals + 1)
    return float((1.0 - probability ** (proposals + 1)) / (1.0 - probability))

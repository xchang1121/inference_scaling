"""Verifier-replay fast path with exact Uno fallback for the HF runtime.

This module is the checkpoint-level reference implementation of VR-Uno.  A
cache hit is drafted by a deterministic point mass and verified in one base-AR
forward.  A miss, an ineligible hit, or a router rejection executes the same
two-forward linear Uno cycle as :mod:`online_speculation.hf_uno`.

The Hugging Face backend is intentionally a correctness and algorithmic
prototype.  It does not claim the fused Nano-vLLM wall-clock performance of the
official Uno runtime.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .hf_uno import (
    HfUnoRuntime,
    RunMetrics,
    _cache_length,
    _crop_cache_by,
    _first_stop_length,
    _generator,
    _sync,
)
from .replay_cache import (
    CostAwareReplayRouter,
    ReplayCandidate,
    VerifierReplayCache,
)
from .torch_sampling import (
    VerificationResult,
    filtered_distribution,
    verify_linear_filtered,
    verify_linear_greedy,
    verify_replay_filtered,
    verify_replay_greedy,
)


@dataclass(frozen=True)
class ReplayRuntimeConfig:
    """Execution controls that do not alter the target sampling distribution."""

    block_size: int = 8
    observe_after_request: bool = True
    causal_within_request: bool = False

    def validate(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be positive.")


@dataclass(frozen=True)
class ReplayDiagnostics:
    """Auditable accounting for one hybrid replay/Uno request."""

    namespace: str
    exactness_mode: str
    replay_cycles: int
    static_cycles: int
    cache_hit_cycles: int
    cache_miss_cycles: int
    invalid_candidate_cycles: int
    replay_forwards: int
    uno_forwards: int
    replay_committed_tokens: int
    static_committed_tokens: int
    replay_accepted_tokens: int
    replay_attempted_tokens: int
    static_accepted_tokens: int
    static_attempted_tokens: int
    replay_lookaheads: int
    static_lookaheads: int
    lookup_seconds: float
    cache_update_seconds: float
    cache_update_in_decode_seconds: float
    cache_close_seconds: float
    causal_session_enabled: bool
    causal_records_created: int
    route_reason_counts: dict[str, int]
    cache_records_added: int
    cache_before: dict[str, Any]
    cache_after: dict[str, Any]
    router_after: dict[str, object]

    @property
    def replay_tokens_per_forward(self) -> float:
        if self.replay_forwards == 0:
            return 0.0
        return self.replay_committed_tokens / self.replay_forwards

    @property
    def static_tokens_per_forward(self) -> float:
        if self.uno_forwards == 0:
            return 0.0
        return self.static_committed_tokens / self.uno_forwards


@dataclass(frozen=True)
class ReplayRunResult:
    metrics: RunMetrics
    diagnostics: ReplayDiagnostics


@dataclass(frozen=True)
class _CycleOutcome:
    cache: object
    verification: VerificationResult
    kind: str
    forwards: int
    frontier_rows: int
    speculative_tokens: int
    matched_suffix_length: int | None = None
    target_predictions: Tensor | None = None


class HfReplayUnoRunner:
    """Run one-pass verifier replay with a lossless two-pass Uno fallback."""

    def __init__(
        self,
        runtime: HfUnoRuntime,
        *,
        replay_cache: VerifierReplayCache,
        router: CostAwareReplayRouter,
    ) -> None:
        if replay_cache.namespace != router.namespace:
            raise ValueError("replay cache and router namespaces must be identical.")
        self.runtime = runtime
        self.replay_cache = replay_cache
        self.router = router

    def _static_cycle(
        self,
        *,
        cache: object,
        seed_token: int,
        block_size: int,
        generator: torch.Generator,
        retain_target_predictions: bool = False,
    ) -> _CycleOutcome:
        runtime = self.runtime
        prefix_cache_length = _cache_length(cache)
        seed_tensor = torch.tensor(
            [[seed_token]],
            device=runtime.device,
            dtype=torch.long,
        )
        if block_size > 1:
            noise = torch.randint(
                1,
                runtime.mask_token_id,
                (1, block_size - 1),
                device=runtime.device,
                dtype=torch.long,
                generator=generator,
            )
            draft_input = torch.cat((seed_tensor, noise), dim=1)
        else:
            draft_input = seed_tensor

        lora_mask = torch.ones(
            (1, block_size),
            device=runtime.device,
            dtype=torch.float32,
        )
        lora_mask[:, 0] = 0.0
        runtime.router.set_token_mask(lora_mask)
        with torch.inference_mode():
            draft_output = runtime.model(
                input_ids=draft_input,
                past_key_values=cache,
                use_cache=True,
            )
        cache = draft_output.past_key_values
        if _cache_length(cache) != prefix_cache_length + block_size:
            raise RuntimeError("static draft cache did not advance by block_size.")
        _crop_cache_by(cache, block_size - 1)

        draft_logits = draft_output.logits[0]
        free_token = int(runtime._sample_logits(draft_logits[0:1], generator).item())
        draft_used = None
        if block_size > 1:
            if runtime.sampling.temperature <= 0:
                spec_tokens = torch.argmax(draft_logits[1:], dim=-1)
            else:
                draft_used = filtered_distribution(draft_logits[1:], runtime.sampling)
                spec_tokens = draft_used.sample(generator)
        else:
            spec_tokens = torch.empty((0,), device=runtime.device, dtype=torch.long)

        proposal = torch.cat(
            (
                torch.tensor([free_token], device=runtime.device, dtype=torch.long),
                spec_tokens,
            )
        ).unsqueeze(0)
        with torch.inference_mode(), runtime._base_context():
            verify_output = runtime.model(
                input_ids=proposal,
                past_key_values=cache,
                use_cache=True,
            )
        cache = verify_output.past_key_values
        verify_logits = verify_output.logits[0]
        # Launch the optional tail reduction before the verifier's existing
        # host transfer so its GPU work belongs to this completed cycle.
        target_predictions = (
            torch.argmax(verify_logits, dim=-1)
            if retain_target_predictions else None
        )
        if _cache_length(cache) != prefix_cache_length + block_size + 1:
            raise RuntimeError("static verifier cache frontier is inconsistent.")

        if block_size > 1 and runtime.sampling.temperature <= 0:
            verification = verify_linear_greedy(
                free_token=free_token,
                spec_tokens=spec_tokens,
                target_logits=verify_logits[:-1],
                lookahead_logits=verify_logits[-1],
            )
        elif block_size > 1:
            if draft_used is None:
                raise RuntimeError("stochastic Uno cycle lost its saved proposal law.")
            verification = verify_linear_filtered(
                free_token=free_token,
                spec_tokens=spec_tokens,
                target=filtered_distribution(verify_logits[:-1], runtime.sampling),
                draft_used=draft_used,
                lookahead=filtered_distribution(verify_logits[-1:], runtime.sampling),
                generator=generator,
            )
        else:
            lookahead_token = int(
                runtime._sample_logits(verify_logits[-1:], generator).item()
            )
            verification = VerificationResult(
                committed=(free_token, lookahead_token),
                accepted_spec_tokens=0,
                rejected_index=None,
                used_lookahead=True,
            )

        return _CycleOutcome(
            cache=cache,
            verification=verification,
            kind="static",
            forwards=2,
            frontier_rows=block_size + 1,
            speculative_tokens=block_size - 1,
            target_predictions=target_predictions,
        )

    def _replay_cycle(
        self,
        *,
        cache: object,
        seed_token: int,
        candidate: ReplayCandidate,
        generator: torch.Generator,
    ) -> _CycleOutcome:
        runtime = self.runtime
        prefix_cache_length = _cache_length(cache)
        spec_tokens = torch.tensor(
            candidate.token_ids,
            device=runtime.device,
            dtype=torch.long,
        )
        seed_tensor = torch.tensor(
            [seed_token],
            device=runtime.device,
            dtype=torch.long,
        )
        replay_input = torch.cat((seed_tensor, spec_tokens)).unsqueeze(0)
        with torch.inference_mode(), runtime._base_context():
            verify_output = runtime.model(
                input_ids=replay_input,
                past_key_values=cache,
                use_cache=True,
            )
        cache = verify_output.past_key_values
        frontier_rows = 1 + spec_tokens.numel()
        if _cache_length(cache) != prefix_cache_length + frontier_rows:
            raise RuntimeError("replay verifier cache frontier is inconsistent.")

        verify_logits = verify_output.logits[0]
        if runtime.sampling.temperature <= 0:
            verification = verify_replay_greedy(
                spec_tokens=spec_tokens,
                target_logits=verify_logits[:-1],
                lookahead_logits=verify_logits[-1],
            )
        else:
            verification = verify_replay_filtered(
                spec_tokens=spec_tokens,
                target=filtered_distribution(verify_logits[:-1], runtime.sampling),
                lookahead=filtered_distribution(verify_logits[-1:], runtime.sampling),
                generator=generator,
            )
        return _CycleOutcome(
            cache=cache,
            verification=verification,
            kind="replay",
            forwards=1,
            frontier_rows=frontier_rows,
            speculative_tokens=spec_tokens.numel(),
            matched_suffix_length=candidate.matched_suffix_length,
        )

    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        seed: int,
        config: ReplayRuntimeConfig,
    ) -> ReplayRunResult:
        """Generate from the unchanged target distribution and then close cache data."""

        config.validate()
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) < 1:
            raise ValueError("input_ids must contain one non-empty prompt.")

        runtime = self.runtime
        generator = _generator(runtime.device, seed)
        base_memory = (
            torch.cuda.memory_allocated(runtime.device)
            if runtime.device.type == "cuda"
            else 0
        )
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)

        prompt_tokens = tuple(int(token) for token in input_ids[0].tolist())
        cache_before = asdict(self.replay_cache.stats())
        causal_session = (
            self.replay_cache.begin_causal_session(prompt_tokens=prompt_tokens)
            if config.causal_within_request
            else None
        )
        cache, seed_token, prefill_seconds = runtime._prefill(input_ids, generator)
        output_tokens = [seed_token]
        stopped = not runtime.ignore_stop and seed_token in runtime.stop_token_ids

        cycles = 0
        forwards = 0
        committed_cycle_tokens = 0
        accepted_spec_tokens = 0
        attempted_spec_tokens = 0
        lookaheads = 0
        replay_cycles = 0
        static_cycles = 0
        cache_hit_cycles = 0
        cache_miss_cycles = 0
        invalid_candidate_cycles = 0
        replay_committed_tokens = 0
        static_committed_tokens = 0
        replay_accepted_tokens = 0
        replay_attempted_tokens = 0
        static_accepted_tokens = 0
        static_attempted_tokens = 0
        replay_lookaheads = 0
        static_lookaheads = 0
        lookup_seconds = 0.0
        cache_update_seconds = 0.0
        cache_update_in_decode_seconds = 0.0
        cache_close_seconds = 0.0
        route_reasons: Counter[str] = Counter()

        _sync(runtime.device)
        decode_start = time.perf_counter()
        if causal_session is not None:
            cache_update_start = time.perf_counter()
            causal_session.append_verified((seed_token,))
            elapsed = time.perf_counter() - cache_update_start
            cache_update_seconds += elapsed
            cache_update_in_decode_seconds += elapsed
        while len(output_tokens) < max_new_tokens and not stopped:
            prefix_cache_length = _cache_length(cache)
            candidate = None
            if config.block_size > 1:
                lookup_start = time.perf_counter()
                lookup_source = (
                    causal_session
                    if causal_session is not None
                    else self.replay_cache
                )
                candidate = lookup_source.lookup(
                    prompt_tokens + tuple(output_tokens),
                    max_tokens=config.block_size - 1,
                )
                lookup_seconds += time.perf_counter() - lookup_start

            use_replay = False
            if candidate is None:
                cache_miss_cycles += 1
                route_reasons[
                    "block-size-one" if config.block_size == 1 else "cache-miss"
                ] += 1
            elif any(
                token < 0 or token >= runtime.vocab_size
                for token in candidate.token_ids
            ):
                cache_hit_cycles += 1
                invalid_candidate_cycles += 1
                route_reasons["invalid-candidate-token"] += 1
            else:
                cache_hit_cycles += 1
                decision = self.router.decide(candidate)
                route_reasons[decision.reason] += 1
                use_replay = decision.use_replay

            if use_replay:
                if candidate is None:
                    raise RuntimeError("replay route selected without a candidate.")
                outcome = self._replay_cycle(
                    cache=cache,
                    seed_token=seed_token,
                    candidate=candidate,
                    generator=generator,
                )
            else:
                outcome = self._static_cycle(
                    cache=cache,
                    seed_token=seed_token,
                    block_size=config.block_size,
                    generator=generator,
                )

            committed = list(outcome.verification.committed)
            if not runtime.ignore_stop:
                committed = committed[
                    : _first_stop_length(committed, runtime.stop_token_ids)
                ]
            remaining = max_new_tokens - len(output_tokens)
            committed = committed[:remaining]
            if not committed:
                raise RuntimeError("hybrid Uno cycle committed no tokens.")

            tokens_to_remove = outcome.frontier_rows - len(committed)
            if tokens_to_remove < 0:
                raise RuntimeError("cycle committed beyond its KV frontier.")
            _crop_cache_by(outcome.cache, tokens_to_remove)
            expected_cache_length = prefix_cache_length + len(committed)
            if _cache_length(outcome.cache) != expected_cache_length:
                raise RuntimeError("post-cycle cache frontier is inconsistent.")
            cache = outcome.cache

            if outcome.kind == "static":
                visible_accepted = min(
                    outcome.verification.accepted_spec_tokens,
                    max(0, len(committed) - 1),
                )
                visible_attempted = min(
                    outcome.speculative_tokens,
                    max(0, len(committed) - 1),
                )
            else:
                visible_accepted = min(
                    outcome.verification.accepted_spec_tokens,
                    len(committed),
                )
                visible_attempted = min(
                    outcome.speculative_tokens,
                    len(committed),
                )
            used_visible_lookahead = int(
                outcome.verification.used_lookahead
                and len(committed) == outcome.frontier_rows
            )

            output_tokens.extend(committed)
            if causal_session is not None:
                cache_update_start = time.perf_counter()
                causal_session.append_verified(committed)
                elapsed = time.perf_counter() - cache_update_start
                cache_update_seconds += elapsed
                cache_update_in_decode_seconds += elapsed
            seed_token = committed[-1]
            stopped = not runtime.ignore_stop and seed_token in runtime.stop_token_ids
            cycles += 1
            forwards += outcome.forwards
            committed_cycle_tokens += len(committed)
            accepted_spec_tokens += visible_accepted
            attempted_spec_tokens += visible_attempted
            lookaheads += used_visible_lookahead

            if outcome.kind == "replay":
                replay_cycles += 1
                replay_committed_tokens += len(committed)
                replay_accepted_tokens += visible_accepted
                replay_attempted_tokens += visible_attempted
                replay_lookaheads += used_visible_lookahead
                if outcome.matched_suffix_length is None:
                    raise RuntimeError("replay cycle lost its match-length bucket.")
                self.router.observe_replay(
                    matched_suffix_length=outcome.matched_suffix_length,
                    committed_tokens=len(committed),
                    forwards=1,
                )
            else:
                static_cycles += 1
                static_committed_tokens += len(committed)
                static_accepted_tokens += visible_accepted
                static_attempted_tokens += visible_attempted
                static_lookaheads += used_visible_lookahead
                self.router.observe_static(
                    committed_tokens=len(committed),
                    forwards=2,
                )

        _sync(runtime.device)
        decode_seconds = time.perf_counter() - decode_start
        cache_records_added = 0
        causal_records_created = 0
        if causal_session is not None:
            cache_update_start = time.perf_counter()
            cache_records_added = causal_session.close(
                publish=config.observe_after_request,
            )
            cache_close_seconds = time.perf_counter() - cache_update_start
            cache_update_seconds += cache_close_seconds
            causal_records_created = causal_session.local_records
        elif config.observe_after_request:
            cache_update_start = time.perf_counter()
            cache_records_added = self.replay_cache.observe_sequence(
                prompt_tokens=prompt_tokens,
                verified_completion_tokens=output_tokens,
            )
            cache_close_seconds = time.perf_counter() - cache_update_start
            cache_update_seconds = cache_close_seconds
        cache_after = asdict(self.replay_cache.stats())

        metrics = runtime._metrics(
            method="uno_verifier_replay_hf_fallback",
            block_size=config.block_size,
            input_ids=input_ids,
            output_tokens=output_tokens,
            forwards=forwards,
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
        diagnostics = ReplayDiagnostics(
            namespace=self.replay_cache.namespace,
            exactness_mode=(
                "greedy-argmax-equivalence"
                if runtime.sampling.temperature <= 0
                else "filtered-psi-spec-delta-correction"
            ),
            replay_cycles=replay_cycles,
            static_cycles=static_cycles,
            cache_hit_cycles=cache_hit_cycles,
            cache_miss_cycles=cache_miss_cycles,
            invalid_candidate_cycles=invalid_candidate_cycles,
            replay_forwards=replay_cycles,
            uno_forwards=2 * static_cycles,
            replay_committed_tokens=replay_committed_tokens,
            static_committed_tokens=static_committed_tokens,
            replay_accepted_tokens=replay_accepted_tokens,
            replay_attempted_tokens=replay_attempted_tokens,
            static_accepted_tokens=static_accepted_tokens,
            static_attempted_tokens=static_attempted_tokens,
            replay_lookaheads=replay_lookaheads,
            static_lookaheads=static_lookaheads,
            lookup_seconds=lookup_seconds,
            cache_update_seconds=cache_update_seconds,
            cache_update_in_decode_seconds=cache_update_in_decode_seconds,
            cache_close_seconds=cache_close_seconds,
            causal_session_enabled=causal_session is not None,
            causal_records_created=causal_records_created,
            route_reason_counts=dict(sorted(route_reasons.items())),
            cache_records_added=cache_records_added,
            cache_before=cache_before,
            cache_after=cache_after,
            router_after=self.router.snapshot(),
        )
        return ReplayRunResult(metrics=metrics, diagnostics=diagnostics)

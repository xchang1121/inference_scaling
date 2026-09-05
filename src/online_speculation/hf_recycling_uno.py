"""Uno refill + one-forward verifier-tail recycling on a real HF KV cache."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor

from .hf_replay_uno import HfReplayUnoRunner, _CycleOutcome
from .hf_uno import (
    HfUnoRuntime, RunMetrics, _cache_length, _crop_cache_by,
    _first_stop_length, _generator, _sync,
)
from .recycling import (
    RecyclingConfig, RecyclingController, tail_after_commit, verify_target_draws,
)


@dataclass(frozen=True)
class RecyclingRunResult:
    metrics: RunMetrics
    diagnostics: dict[str, object]


class HfRecyclingUnoRunner:
    def __init__(self, runtime: HfUnoRuntime) -> None:
        self.runtime = runtime

    def _recycle_cycle(
        self, *, cache: object, seed_token: int, candidates: Tensor,
        generator: torch.Generator,
    ) -> _CycleOutcome:
        runtime = self.runtime
        length_before = _cache_length(cache)
        inputs = torch.cat((
            torch.tensor([seed_token], device=runtime.device, dtype=torch.long),
            candidates,
        )).unsqueeze(0)
        with torch.inference_mode(), runtime._base_context():
            output = runtime.model(
                input_ids=inputs, past_key_values=cache, use_cache=True,
            )
        logits = output.logits[0]
        if _cache_length(output.past_key_values) != length_before + inputs.size(1):
            raise RuntimeError("recycling forward KV frontier mismatch")
        # Greedy target draws also supply the next candidate predictions.
        # Stochastic target draws are fresh and independent across rows; the
        # next proposal remains a deterministic argmax of these past logits.
        draws = runtime._sample_logits(logits, generator)
        predictions = (
            draws if runtime.sampling.temperature <= 0
            else torch.argmax(logits, dim=-1)
        )
        verification = verify_target_draws(candidates, draws)
        return _CycleOutcome(
            cache=output.past_key_values,
            verification=verification,
            kind="recycle",
            forwards=1,
            frontier_rows=inputs.size(1),
            speculative_tokens=candidates.numel(),
            target_predictions=predictions,
        )

    def generate(
        self, input_ids: Tensor, *, max_new_tokens: int, seed: int,
        config: RecyclingConfig,
    ) -> RecyclingRunResult:
        config.validate()
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) < 1:
            raise ValueError("one nonempty prompt is required")
        runtime = self.runtime
        controller = RecyclingController(config)
        generator = _generator(runtime.device, seed)
        memory_before = (
            torch.cuda.memory_allocated(runtime.device)
            if runtime.device.type == "cuda" else 0
        )
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        cache, seed_token, prefill_seconds = runtime._prefill(input_ids, generator)
        output_tokens = [seed_token]
        candidates = torch.empty(0, device=runtime.device, dtype=torch.long)
        stopped = not runtime.ignore_stop and seed_token in runtime.stop_token_ids
        counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        shapes: Counter[str] = Counter()
        depth = forwards = cycles = accepts = attempts = lookaheads = 0
        online_seconds = 0.0
        _sync(runtime.device)
        decode_start = time.perf_counter()
        while len(output_tokens) < max_new_tokens and not stopped:
            cycle_start = time.perf_counter()
            prior_length = _cache_length(cache)
            candidate_count = candidates.numel()
            use_recycle, reason = controller.decide(
                candidates=candidate_count, depth=depth,
            )
            reasons[reason] += 1
            online_seconds += time.perf_counter() - cycle_start
            if use_recycle:
                outcome = self._recycle_cycle(
                    cache=cache, seed_token=seed_token, candidates=candidates,
                    generator=generator,
                )
                depth += 1
                shapes[f"recycle:{candidate_count + 1}"] += 1
            else:
                # The existing static kernel is reused without a cache/router.
                outcome = HfReplayUnoRunner._static_cycle(
                    self, cache=cache, seed_token=seed_token,
                    block_size=config.block_size, generator=generator,
                    retain_target_predictions=config.policy not in {"disabled", "scaled"},
                    noise_prefix=candidates if config.policy == "warmstart" else None,
                    noise_lora_scale=config.noise_lora_scale,
                )
                if config.policy == "warmstart":
                    counts["warmstart_input_tokens"] += min(
                        candidate_count, config.block_size - 1,
                    )
                depth = 0
                shapes[f"refill:{config.block_size}"] += 1
            committed = list(outcome.verification.committed)
            if not runtime.ignore_stop:
                committed = committed[:_first_stop_length(committed, runtime.stop_token_ids)]
            committed = committed[:max_new_tokens - len(output_tokens)]
            if not committed:
                raise RuntimeError("a recycling cycle must commit at least one token")
            _crop_cache_by(outcome.cache, outcome.frontier_rows - len(committed))
            if _cache_length(outcome.cache) != prior_length + len(committed):
                raise RuntimeError("recycling rollback failed")
            cache = outcome.cache
            output_tokens.extend(committed)
            seed_token = committed[-1]
            stopped = not runtime.ignore_stop and seed_token in runtime.stop_token_ids
            online_start = time.perf_counter()
            if outcome.target_predictions is not None:
                candidates = tail_after_commit(
                    outcome.target_predictions, committed_tokens=len(committed),
                    refill=not use_recycle, max_candidates=config.block_size - 1,
                )
            else:
                candidates = candidates[:0]
            online_seconds += time.perf_counter() - online_start
            elapsed = time.perf_counter() - cycle_start
            online_start = time.perf_counter()
            controller.observe(
                recycle=use_recycle, candidates=candidate_count,
                tokens=len(committed), seconds=elapsed,
            )
            online_seconds += time.perf_counter() - online_start
            kind = "recycle" if use_recycle else "refill"
            counts[kind + "_cycles"] += 1
            counts[kind + "_tokens"] += len(committed)
            counts[kind + "_accepted"] += min(
                outcome.verification.accepted_spec_tokens,
                max(0, len(committed) - int(not use_recycle)),
            )
            forwards += outcome.forwards
            cycles += 1
            accepts += min(
                outcome.verification.accepted_spec_tokens,
                max(0, len(committed) - int(not use_recycle)),
            )
            attempts += min(
                outcome.speculative_tokens,
                max(0, len(committed) - int(not use_recycle)),
            )
            lookaheads += int(
                outcome.verification.used_lookahead
                and len(committed) == outcome.frontier_rows
            )
        _sync(runtime.device)
        decode_seconds = time.perf_counter() - decode_start
        metrics = runtime._metrics(
            method="recycling_uno_hf", block_size=config.block_size,
            input_ids=input_ids, output_tokens=output_tokens, forwards=forwards,
            cycles=cycles, committed_cycle_tokens=len(output_tokens) - 1,
            accepted_spec_tokens=accepts, attempted_spec_tokens=attempts,
            lookaheads=lookaheads, prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds, base_memory=memory_before, stopped=stopped,
        )
        return RecyclingRunResult(metrics=metrics, diagnostics={
            **dict(counts),
            "route_reasons": dict(reasons),
            "forward_shapes": dict(shapes),
            "controller": controller.snapshot(),
            "online_host_seconds": online_seconds,
            "online_cost_included_in_decode": True,
            "request_local": True,
            "optimizer_steps": 0,
            "model_parameters_frozen": all(
                not parameter.requires_grad for parameter in runtime.model.parameters()
            ),
            "numerical_scope": "exact with respect to the computed target distributions",
        })

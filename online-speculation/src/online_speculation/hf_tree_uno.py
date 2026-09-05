"""Packed-tree reference runtime with ancestor masks and compacted dynamic KV."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor

from .feedback_budget import FeedbackBudgetController
from .hf_uno import (
    HfUnoRuntime, RunMetrics, _cache_length, _crop_cache_by,
    _first_stop_length, _generator, _sync,
)
from .tree_uno import (
    CandidateTree, RankCalibrator, TreeBudgetController, TreeConfig, build_tree, walk_target_draws,
)
from .tree_feedback import nested_lengths_from_walk


def tree_attention_mask(tree: CandidateTree, prefix_length: int, *, device: torch.device) -> Tensor:
    """Boolean SDPA mask: True means visible, including self and all ancestors."""
    count = len(tree.nodes)
    mask = torch.zeros((count, prefix_length + count), device=device, dtype=torch.bool)
    mask[:, :prefix_length] = True
    # Small CPU-built topology is transferred as indices; no model data copied.
    rows, cols = [], []
    for index in range(count):
        ancestors = tree.ancestor_indices(index)
        rows.extend([index] * len(ancestors))
        cols.extend(prefix_length + a for a in ancestors)
    mask[rows, cols] = True
    return mask[None, None]


def compact_tree_cache(cache: object, prefix_length: int, path: tuple[int, ...], nodes: int) -> None:
    """Move only the newly written tree tail, preserving the long prefix in-place.

    Full-attention DynamicCache only. Sliding/quantized/offloaded layouts are
    deliberately not silently treated as ordinary KV tensors.
    """
    if _cache_length(cache) != prefix_length + nodes:
        raise RuntimeError("tree cache frontier differs from expected packed length")
    if any(not 0 <= index < nodes for index in path):
        raise ValueError("tree path index out of bounds")
    if path and path != tuple(range(len(path))):
        if not hasattr(cache, "layers"):
            raise TypeError("packed tree requires transformers DynamicCache.layers")
        layers = cache.layers
        if any(getattr(layer, "is_sliding", False) or not hasattr(layer, "keys") for layer in layers):
            raise TypeError("packed tree supports only full-attention dynamic KV layers")
        indices = torch.tensor(path, dtype=torch.long, device=layers[0].keys.device) + prefix_length
        with torch.inference_mode():
            for layer in layers:
                for states in (layer.keys, layer.values):
                    selected = states.index_select(-2, indices)
                    states[..., prefix_length:prefix_length + len(path), :].copy_(selected)
    _crop_cache_by(cache, nodes - len(path))


@dataclass(frozen=True)
class TreeRunResult:
    metrics: RunMetrics
    diagnostics: dict[str, object]


class HfTreeUnoRunner:
    def __init__(self, runtime: HfUnoRuntime) -> None:
        self.runtime = runtime

    def generate(
        self, input_ids: Tensor, *, max_new_tokens: int, seed: int, config: TreeConfig,
    ) -> TreeRunResult:
        config.validate()
        if max_new_tokens < 1 or input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) < 1:
            raise ValueError("one nonempty prompt and positive token budget required")
        runtime = self.runtime
        generator = _generator(runtime.device, seed)
        calibrator = RankCalibrator(config)
        controller = FeedbackBudgetController(config, seed=seed) if config.feedback_budget else TreeBudgetController(config)
        memory_before = torch.cuda.memory_allocated(runtime.device) if runtime.device.type == "cuda" else 0
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        cache, seed_token, prefill_seconds = runtime._prefill(input_ids, generator)
        output_tokens = [seed_token]
        stopped = not runtime.ignore_stop and seed_token in runtime.stop_token_ids
        cycles = accepted = attempted = lookaheads = 0
        shapes: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        online_seconds = compact_seconds = 0.0
        _sync(runtime.device)
        decode_start = time.perf_counter()
        with torch.inference_mode():
            while len(output_tokens) < max_new_tokens and not stopped:
                cycle_start = time.perf_counter()
                prefix_before = _cache_length(cache)
                noise = torch.randint(
                    1, runtime.mask_token_id, (1, config.block_size - 1),
                    device=runtime.device, dtype=torch.long, generator=generator,
                )
                draft_input = torch.cat((
                    torch.tensor([[seed_token]], device=runtime.device, dtype=torch.long), noise,
                ), dim=1)
                lora_mask = torch.ones_like(draft_input, dtype=torch.float32)
                lora_mask[:, 0] = 0
                runtime.router.set_token_mask(lora_mask)
                draft = runtime.model(input_ids=draft_input, past_key_values=cache, use_cache=True)
                cache = draft.past_key_values
                _crop_cache_by(cache, config.block_size - 1)
                prefix_length = _cache_length(cache)
                if prefix_length != prefix_before + 1:
                    raise RuntimeError("tree draft did not preserve exactly one base seed KV")
                free = runtime._sample_logits(draft.logits[0, :1], generator)
                logits = draft.logits[0, 1:].float()
                k = min(config.top_k, logits.size(-1))
                if k != config.top_k:
                    raise ValueError("tree top_k exceeds the vocabulary")
                _, ids = logits.topk(k, dim=-1, sorted=True)
                # Softmax's shifted reduction avoids cancellation when a large
                # dominant logit makes exp(top - logsumexp) sum slightly > 1.
                probabilities = torch.softmax(logits, dim=-1).gather(-1, ids)
                # One small transfer supplies the free token, ranks and masses.
                packed = torch.cat((free.double(), ids.reshape(-1).double(), probabilities.reshape(-1).double()))
                host = packed.tolist()
                host_start = time.perf_counter()
                width = config.block_size - 1
                candidate_ids = [list(map(int, host[1+d*k:1+(d+1)*k])) for d in range(width)]
                offset = 1 + width * k
                prior = [host[offset+d*k:offset+(d+1)*k] for d in range(width)]
                full_tree = build_tree(
                    int(host[0]), candidate_ids, calibrator.weights(prior),
                    nodes=max(config.node_budgets or (config.nodes,)), include_spine=config.include_spine,
                )
                if isinstance(controller, FeedbackBudgetController):
                    budget, reason = controller.choose(full_tree, remaining=max_new_tokens - len(output_tokens))
                else:
                    budget, reason = controller.choose(full_tree)
                tree = CandidateTree(full_tree.nodes[:budget])
                reasons[reason] += 1
                count = len(tree.nodes)
                tree_inputs = torch.tensor([[n.token for n in tree.nodes]], device=runtime.device)
                positions = torch.tensor([[prefix_length + n.depth for n in tree.nodes]], device=runtime.device)
                mask = tree_attention_mask(tree, prefix_length, device=runtime.device)
                online_seconds += time.perf_counter() - host_start
                with runtime._base_context():
                    verify = runtime.model(
                        input_ids=tree_inputs, attention_mask=mask, position_ids=positions,
                        past_key_values=cache, use_cache=True,
                    )
                cache = verify.past_key_values
                draws = runtime._sample_logits(verify.logits[0], generator).tolist()
                host_start = time.perf_counter()
                walk = walk_target_draws(tree, draws)
                committed = list(walk.committed)
                if not runtime.ignore_stop:
                    committed = committed[:_first_stop_length(committed, runtime.stop_token_ids)]
                committed = committed[:max_new_tokens - len(output_tokens)]
                if isinstance(controller, FeedbackBudgetController):
                    visible_rewards = nested_lengths_from_walk(
                        walk, [n for n in config.node_budgets if n <= budget],
                        verified_nodes=count, output_limit=len(committed),
                    )
                    controller.observe_rewards(budget, visible_rewards)
                keep = walk.path_indices[:len(committed) - 1]
                compact_start = time.perf_counter()
                compact_tree_cache(cache, prefix_length, keep, count)
                compact_seconds += time.perf_counter() - compact_start
                if _cache_length(cache) != prefix_before + len(committed):
                    raise RuntimeError("tree rollback failed logical KV invariant")
                # Only used outputs before request truncation teach the next cycle.
                calibrator.observe(walk.observations[:max(0, len(committed) - 1)], candidate_ids)
                online_seconds += time.perf_counter() - host_start
                output_tokens.extend(committed)
                seed_token = committed[-1]
                stopped = not runtime.ignore_stop and seed_token in runtime.stop_token_ids
                cycles += 1
                shapes[str(count)] += 1
                accepted += min(len(walk.path_indices) - 1, max(0, len(committed) - 1))
                attempted += max(0, len(committed) - 1)
                lookaheads += int(walk.used_leaf_lookahead and len(committed) == len(walk.committed))
                if config.node_budgets:
                    # Include asynchronous tail compaction in the cost label.
                    # This extra sync is controller overhead, not excluded time.
                    _sync(runtime.device)
                controller.observe(budget, tokens=len(committed), seconds=time.perf_counter() - cycle_start)
        _sync(runtime.device)
        decode_seconds = time.perf_counter() - decode_start
        metrics = runtime._metrics(
            method="budgeted_tree_uno_hf", block_size=config.block_size,
            input_ids=input_ids, output_tokens=output_tokens, forwards=2*cycles,
            cycles=cycles, committed_cycle_tokens=len(output_tokens) - 1,
            accepted_spec_tokens=accepted, attempted_spec_tokens=attempted,
            lookaheads=lookaheads, prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds, base_memory=memory_before, stopped=stopped,
        )
        return TreeRunResult(metrics, {
            "tree_shapes": dict(shapes), "online_host_seconds": online_seconds,
            "kv_compact_host_seconds": compact_seconds, "rank_calibrator": calibrator.snapshot(),
            "budget_controller": controller.snapshot(), "budget_reasons": dict(reasons),
            "cycle_sync_for_cost": bool(config.node_budgets),
            "all_online_costs_in_decode": True, "request_local": True, "optimizer_steps": 0,
            "model_parameters_frozen": all(not p.requires_grad for p in runtime.model.parameters()),
            "exactness": "one target draw per reached tree node; computed-distribution scope",
        })

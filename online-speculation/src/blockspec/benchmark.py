"""Balanced request streams with complete online costs and unchanged-base checks.

This is a development evaluator, not a benchmark-suite claim. No results or
tokens are written to disk. Each online stream starts at the same offline
checkpoint; learned weights and Adam state persist across its requests.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
import time

import torch

from .checkpoint import adapter_state, base_fingerprint
from .decoding import generate_ar, generate_speculative
from .online import OnlineConfig, OnlineLearner, synchronize
from .tree import generate_tree


@dataclass(frozen=True)
class BenchmarkConfig:
    tokens: int = 128
    block_size: int = 4
    repeats: int = 2
    warmup_tokens: int = 8
    seed: int = 271828
    sampler: str = "linear"
    top_k: int = 4
    prefix_budget: int = 16
    eos_id: int | None = None

    def __post_init__(self):
        if min(self.tokens, self.warmup_tokens, self.top_k, self.prefix_budget) < 1:
            raise ValueError("positive token and tree budgets required")
        if self.block_size < 2 or self.repeats < 2 or self.repeats % 2:
            raise ValueError("block >=2 and a positive even repeat count required")
        if self.sampler not in ("linear", "tree"):
            raise ValueError("unknown sampler")


def continuation_prompts(sequences, *, count, length):
    """First N sufficiently long records, first L tokens; no hidden resampling.

    These are continuation prefixes of existing records, not newly composed
    instruction prompts. The caller must use a development or held-out split.
    """
    if count < 1 or length < 1:
        raise ValueError("positive prompt count and length required")
    eligible = [s for s in sequences if s.ndim == 1 and len(s) >= length]
    if len(eligible) < count:
        raise ValueError("not enough sufficiently long records")
    return [s[:length].reshape(1, -1).clone() for s in eligible[:count]]


def compare_tokens(reference, actual):
    common = 0
    for left, right in zip(reference, actual):
        if left != right:
            break
        common += 1
    return {"identical": reference == actual, "common_prefix": common,
            "reference_tokens": len(reference), "actual_tokens": len(actual)}


def aggregate(rows, *, setup_seconds=0.0):
    """Token-weighted throughput, never the arithmetic mean of request TPS."""
    if not rows or not math.isfinite(setup_seconds) or setup_seconds < 0:
        raise ValueError("nonempty measurements and nonnegative setup time required")
    fields = ("tokens", "seconds", "decode_forwards", "rounds", "accepted", "proposed",
              "updates", "update_seconds")
    result = {key: sum(row[key] for row in rows) for key in fields}
    if result["seconds"] <= 0 or result["tokens"] < 0:
        raise ValueError("invalid measured tokens or elapsed time")
    result.update(requests=len(rows), setup_seconds=setup_seconds,
                  tps=result["tokens"] / result["seconds"],
                  tps_including_learner_setup=result["tokens"] / (result["seconds"] + setup_seconds))
    result["tokens_per_round"] = result["tokens"] / result["rounds"] if result["rounds"] else 0.0
    return result


def benchmark_streams(model, prompts, config=BenchmarkConfig(), online_config=OnlineConfig(), *, progress=None):
    if not prompts or any(p.ndim != 2 or p.shape[0] != 1 or p.shape[1] < 1 for p in prompts):
        raise ValueError("nonempty batch-one prompts required")
    device = next(model.parameters()).device
    prompts = [p.to(device) for p in prompts]
    initial = adapter_state(model)
    frozen = base_fingerprint(model)
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    original_dtypes = {name: p.dtype for name, p in model.named_parameters()}
    measurements, generated, setup, adapter_changes = {}, {}, {}, []
    trainable = 0
    spec = generate_tree if config.sampler == "tree" else generate_speculative
    options = {"block_size": config.block_size}
    if config.sampler == "tree":
        options.update(top_k=config.top_k, prefix_budget=config.prefix_budget)

    def restore():
        model.load_state_dict(initial, strict=False)
        model.train_adapters_only()

    def rng(seed):
        return torch.Generator(device=device).manual_seed(seed)

    try:
        # Warm kernels, not the reported online adapter/Adam trajectory. Lazy Adam
        # allocation still occurs in each fresh measured learner's first update.
        generate_ar(model, prompts[0], config.warmup_tokens, eos_id=config.eos_id)
        spec(model, prompts[0], config.warmup_tokens, **options, eos_id=config.eos_id,
             generator=rng(config.seed))
        warm = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1,
                                                 learning_rate=online_config.learning_rate,
                                                 loss=online_config.loss,
                                                 clip_norm=online_config.clip_norm,
                                                 train_last_layers=online_config.train_last_layers))
        spec(model, prompts[0], max(4, config.warmup_tokens), **options, learner=warm,
             eos_id=config.eos_id, generator=rng(config.seed))
        del warm
        restore()
        synchronize(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for repeat in range(config.repeats):
            order = ("ar", "static", "online") if repeat % 2 == 0 else ("online", "static", "ar")
            for arm in order:
                restore()
                learner = None
                setup[(repeat, arm)] = 0.0
                if arm == "online":
                    synchronize(model)
                    start = time.perf_counter()
                    learner = OnlineLearner(model, online_config)
                    trainable = sum(p.numel() for p in learner.parameters)
                    synchronize(model)
                    setup[(repeat, arm)] = time.perf_counter() - start
                for request, prompt in enumerate(prompts):
                    generator = rng(config.seed + repeat * len(prompts) + request)
                    if arm == "ar":
                        output = generate_ar(model, prompt, config.tokens, eos_id=config.eos_id,
                                             generator=generator)
                    else:
                        output = spec(model, prompt, config.tokens, **options, learner=learner,
                                      eos_id=config.eos_id, generator=generator)
                    key = (repeat, arm, request)
                    measurements[key] = output.summary()
                    generated[key] = output.tokens
                    if progress:
                        progress({"repeat": repeat, "arm": arm, "request": request,
                                  "adapter_version": learner.version if learner else 0,
                                  **measurements[key]})
                if learner is not None:
                    after = adapter_state(model)
                    adapter_changes.append(any(not torch.equal(v, after[n]) for n, v in initial.items()))
                    learner.clear_replay()
                del learner
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if base_fingerprint(model) != frozen:
            raise RuntimeError("benchmark online training changed base weights")
    finally:
        # A read-only evaluator must not publish its experimental online state.
        model.load_state_dict(initial, strict=False)
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(original_dtypes[name])
            parameter.grad = None
            parameter.requires_grad_(original_requires_grad[name])

    arms, repeats, comparisons = {}, [], []
    for arm in ("ar", "static", "online"):
        arms[arm] = aggregate([row for (_, label, _), row in measurements.items() if label == arm],
                              setup_seconds=sum(s for (_, label), s in setup.items() if label == arm))
        arms[arm]["speedup_vs_ar"] = arms[arm]["tps"] / arms["ar"]["tps"]
        arms[arm]["speedup_vs_ar_including_setup"] = (
            arms[arm]["tps_including_learner_setup"] / arms["ar"]["tps"])
    for repeat in range(config.repeats):
        repeats.append({arm: aggregate([measurements[(repeat, arm, i)] for i in range(len(prompts))],
                                        setup_seconds=setup[(repeat, arm)])
                        for arm in arms})
        for request in range(len(prompts)):
            for arm in ("static", "online"):
                comparisons.append({"repeat": repeat, "request": request, "arm": arm,
                                    **compare_tokens(generated[(repeat, "ar", request)],
                                                     generated[(repeat, arm, request)])})
    prompt_hashes = [hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in prompts]
    return {"config": asdict(config), "online_config": asdict(online_config), "arms": arms,
            "repeats": repeats, "comparisons": comparisons,
            "greedy_identical": all(c["identical"] for c in comparisons),
            "online_adapter_changed_per_stream": adapter_changes,
            "online_trainable_parameters": trainable,
            "base_unchanged": True, "adapter_restored": True, "peak_allocated_bytes": peak,
            "prompt_sha256": prompt_hashes,
            "scope": "balanced development continuation streams; no test-set or statistical-significance claim"}

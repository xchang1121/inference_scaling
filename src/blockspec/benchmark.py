"""Balanced request streams with cumulative throughput and online learning state.

Each online stream starts at the same offline checkpoint; learned weights and
Adam state persist across its requests. Measurements are returned to the caller.
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


_TOTAL_FIELDS = ("tokens", "seconds", "decode_forwards", "rounds", "accepted", "proposed",
                 "updates", "update_seconds", "feedback_blocks", "fully_covered_rounds", "coverage_skips")


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
    execution: str = "eager"

    def __post_init__(self):
        if min(self.tokens, self.warmup_tokens, self.top_k, self.prefix_budget) < 1:
            raise ValueError("positive token and tree budgets required")
        if self.block_size < 2 or self.repeats < 2 or self.repeats % 2:
            raise ValueError("block >=2 and a positive even repeat count required")
        if self.sampler not in ("linear", "tree"):
            raise ValueError("unknown sampler")
        if self.execution not in ("eager", "cuda_graph"):
            raise ValueError("unknown inference execution")


def continuation_prompts(sequences, *, count, length):
    """First N eligible records, first L tokens, in their original file order."""
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


def aggregate(rows, *, setup_seconds=0.0, engine_setup_seconds=0.0):
    """Total output tokens divided by total measured time and specified setup."""
    if not rows or any(not math.isfinite(t) or t < 0 for t in (setup_seconds, engine_setup_seconds)):
        raise ValueError("nonempty measurements and nonnegative setup time required")
    result = {key: sum(row[key] for row in rows) for key in _TOTAL_FIELDS}
    if result["seconds"] <= 0 or result["tokens"] < 0:
        raise ValueError("invalid measured tokens or elapsed time")
    result.update(requests=len(rows), setup_seconds=setup_seconds,
                  engine_setup_seconds=engine_setup_seconds,
                  tps=result["tokens"] / result["seconds"],
                  tps_including_learner_setup=result["tokens"] / (result["seconds"] + setup_seconds),
                  tps_including_all_setup=result["tokens"] / (result["seconds"] + setup_seconds + engine_setup_seconds))
    result["tokens_per_round"] = result["tokens"] / result["rounds"] if result["rounds"] else 0.0
    return result


def stream_trajectory(rows, *, setup_seconds=0.0, engine_setup_seconds=0.0):
    """Per-request counters and cumulative generation/learner/engine costs."""
    if not rows:
        raise ValueError("a request stream is required")
    totals = dict.fromkeys(_TOTAL_FIELDS, 0)
    trajectory = []
    for request, row in enumerate(rows):
        aggregate([row])
        for key in _TOTAL_FIELDS:
            totals[key] += row[key]
        cumulative = aggregate([totals], setup_seconds=setup_seconds, engine_setup_seconds=engine_setup_seconds)
        cumulative["requests"] = request + 1
        trajectory.append({"request": request, **row, "cumulative": cumulative})
    return trajectory


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
    trainable, optimizer_backend = 0, None
    executor, execution_info = None, {"kind": config.execution, "setup_seconds": 0.0, "signatures": 0}
    engine_cost = dict.fromkeys(("ar", "static", "online"), 0.0)
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
        if config.execution == "cuda_graph":
            from .execution import FixedShapeExecutor
            # Fixed addresses survive in-place adapter restoration and AdamW.
            model.train_adapters_only()
            maximum = max(config.block_size, config.prefix_budget if config.sampler == "tree" else 1)
            executor = FixedShapeExecutor(model, capacity=max(p.shape[1] for p in prompts) +
                                           max(config.tokens, config.warmup_tokens, 4),
                                           max_query=maximum)
            clean = [(i, False, None) for i in range(1, maximum + 1)]
            static_draft = [(i, True, None) for i in range(2, config.block_size + 1)]
            capture = (None if online_config.train_last_layers is None else
                       model.config.num_hidden_layers - online_config.train_last_layers)
            online_draft = [(i, True, capture) for i in range(2, config.block_size + 1)]
            executor.prepare(clean + static_draft + online_draft)
            ar_keys = {(1, False, None)} | {(p.shape[1] - 1, False, None) for p in prompts
                                            if 1 <= p.shape[1] - 1 <= maximum}
            online_keys = clean + online_draft
            if (online_config.feedback_execution == "windowed"
                    and online_config.replay_blocks < online_config.stride):
                online_keys += static_draft
            needed = {"ar": ar_keys, "static": clean + static_draft, "online": online_keys}
            # Charge each arm for the signatures required by its standalone deployment.
            engine_cost = {arm: sum(executor.signature_seconds[k] for k in set(keys)) for arm, keys in needed.items()}
            execution_info.update(setup_seconds=executor.setup_seconds, signatures=len(executor.slots),
                                  capacity=executor.capacity, setup_seconds_by_arm=engine_cost)
            options["executor"] = executor
        # Restore the offline weights after kernel warmup. Each measured learner
        # initializes Adam states during its first timed update.
        generate_ar(model, prompts[0], config.warmup_tokens, eos_id=config.eos_id, executor=executor)
        spec(model, prompts[0], config.warmup_tokens, **options, eos_id=config.eos_id,
             generator=rng(config.seed))
        # Periodic warmup exercises training kernels even for fully covered drafts.
        warm = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=1, update_policy="periodic",
                                                 learning_rate=online_config.learning_rate,
                                                 loss=online_config.loss,
                                                 clip_norm=online_config.clip_norm,
                                                 train_last_layers=online_config.train_last_layers,
                                                 optimizer=online_config.optimizer,
                                                 feedback_execution=online_config.feedback_execution))
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
                    optimizer_backend = learner.optimizer_backend
                    synchronize(model)
                    setup[(repeat, arm)] = time.perf_counter() - start
                for request, prompt in enumerate(prompts):
                    generator = rng(config.seed + repeat * len(prompts) + request)
                    version_start = learner.version if learner is not None else 0
                    if arm == "ar":
                        output = generate_ar(model, prompt, config.tokens, eos_id=config.eos_id,
                                             generator=generator, executor=executor)
                    else:
                        output = spec(model, prompt, config.tokens, **options, learner=learner,
                                      eos_id=config.eos_id, generator=generator)
                    key = (repeat, arm, request)
                    measurements[key] = output.summary()
                    measurements[key].update(adapter_version_start=version_start,
                                             adapter_version=learner.version if learner is not None else 0,
                                             last_update_loss=learner.last_loss if learner is not None else None)
                    generated[key] = output.tokens
                    if progress:
                        progress({"repeat": repeat, "arm": arm, "request": request,
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
        # Restore the caller's adapter weights, dtypes, gradients and training flags.
        model.load_state_dict(initial, strict=False)
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(original_dtypes[name])
            parameter.grad = None
            parameter.requires_grad_(original_requires_grad[name])

    arms, repeats, comparisons, trajectories = {}, [], [], []
    for arm in ("ar", "static", "online"):
        arms[arm] = aggregate([row for (_, label, _), row in measurements.items() if label == arm],
                              setup_seconds=sum(s for (_, label), s in setup.items() if label == arm),
                              engine_setup_seconds=engine_cost[arm])
        arms[arm]["speedup_vs_ar"] = arms[arm]["tps"] / arms["ar"]["tps"]
        arms[arm]["speedup_vs_ar_including_setup"] = (
            arms[arm]["tps_including_learner_setup"] / arms["ar"]["tps"])
        arms[arm]["speedup_vs_ar_including_all_setup"] = (
            arms[arm]["tps_including_all_setup"] / arms["ar"]["tps_including_all_setup"])
    for repeat in range(config.repeats):
        # Prepared graphs persist across repeats; charge setup in the first one.
        repeat_engine = engine_cost if repeat == 0 else dict.fromkeys(arms, 0.0)
        repeats.append({arm: aggregate([measurements[(repeat, arm, i)] for i in range(len(prompts))],
                                        setup_seconds=setup[(repeat, arm)], engine_setup_seconds=repeat_engine[arm])
                        for arm in arms})
        trajectories.append({"repeat": repeat, "arms": {
            arm: stream_trajectory([measurements[(repeat, arm, i)] for i in range(len(prompts))],
                                   setup_seconds=setup[(repeat, arm)],
                                   engine_setup_seconds=repeat_engine[arm]) for arm in arms}})
        for request in range(len(prompts)):
            for arm in ("static", "online"):
                comparisons.append({"repeat": repeat, "request": request, "arm": arm,
                                    **compare_tokens(generated[(repeat, "ar", request)],
                                                     generated[(repeat, arm, request)])})
    prompt_hashes = [hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in prompts]
    return {"config": asdict(config), "online_config": asdict(online_config), "arms": arms,
            "repeats": repeats, "comparisons": comparisons, "trajectories": trajectories,
            "greedy_identical": all(c["identical"] for c in comparisons),
            "online_adapter_changed_per_stream": adapter_changes,
            "online_trainable_parameters": trainable,
            "online_optimizer": optimizer_backend,
            "execution": execution_info,
            "base_unchanged": True, "adapter_restored": True, "peak_allocated_bytes": peak,
            "prompt_sha256": prompt_hashes,
            "scope": "balanced continuation streams with paired AR/static/online measurements"}

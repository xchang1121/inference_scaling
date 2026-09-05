"""R7D: forced common-history probes; NOT generation or a speed benchmark.

Standalone Python 3.10 entry point using the pinned native Uno installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys
import traceback

from wsl_official_baseline import ADAPTER_SHA, BASE_SHA, UNO_COMMIT, command, sha256


BASELINE_SHA = "c10f062b909e90ad96be802ee5221933e098578e0a1ee3ee0be2cb22ab785518"
WIDTHS = (1, 4, 8, 16)
MODES = ("off", "zero", "noise")
EXECUTIONS = ("graph", "eager")
EXPECTED_CONTEXTS = (("english", 94), ("chinese", 60), ("chinese", 94),
                     ("code", 25), ("code", 87), ("math", 85))


def select_contexts(payload):
    if not payload.get("completed") or payload.get("error"):
        raise ValueError("completed baseline required")
    contexts = []
    for name, prompt in payload["design"]["workloads"]:
        rows = [row for row in payload["records"] if row["workload"] == name]
        seed = min(row["seed"] for row in rows)
        rows = [row for row in rows if row["seed"] == seed]
        if sorted(row["block_size"] for row in rows) != list(WIDTHS):
            raise ValueError("one complete baseline width set required per workload")
        reference = next(row for row in rows if row["block_size"] == 1)["output"]["token_ids"]
        positions = set()
        for row in rows:
            other = row["output"]["token_ids"]
            if len(reference) != len(other):
                raise ValueError("baseline output lengths differ")
            difference = next((i for i, (a, b) in enumerate(zip(reference, other)) if a != b), None)
            if difference is not None:
                positions.add(difference)
        for position in sorted(positions):
            contexts.append({"workload": name, "prompt": prompt, "generation_index": position,
                             "ar_prefix": reference[:position], "ar_next_token": reference[position]})
    return contexts


def tensor_digest(value):
    # Canonical float32 CPU bytes; shape is separately recorded/checked.
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def distance(left, right):
    import torch

    if left.shape != right.shape:
        raise ValueError("comparison shapes must match")
    left, right = left.float(), right.float()
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("nonfinite diagnostic tensor")
    difference = left - right
    return {"exact": bool(torch.equal(left, right)),
            "changed_elements": int(torch.count_nonzero(difference)), "elements": left.numel(),
            "max_abs": float(difference.abs().max()),
            "rms": float(difference.square().mean().sqrt())}


def logit_summary(logits):
    import torch

    logits = logits.float().flatten()
    values, indices = logits.topk(5)
    return {"argmax": int(torch.argmax(logits)), "margin": float(values[0] - values[1]),
            "max_ties": int((logits == values[0]).sum()),
            "top5_ids": indices.tolist(), "top5_logits": values.tolist(),
            "sha256_float32": tensor_digest(logits)}


def logit_distance(left, right):
    result = distance(left, right)
    left, right = left.float(), right.float()
    result.update(argmax_equal=int(left.argmax()) == int(right.argmax()),
                  softmax_tv_temperature1=float((left.softmax(-1) - right.softmax(-1)).abs().sum() / 2),
                  perturbation_range=float((left - right).max() - (left - right).min()))
    # Only a sufficient certificate, using observed differences, not a global error bound.
    margin = float(left.topk(2).values[0] - left.topk(2).values[1])
    result["reference_margin_gt_2_max_abs"] = margin > 2 * result["max_abs"]
    if result["reference_margin_gt_2_max_abs"] and not result["argmax_equal"]:
        raise AssertionError("argmax perturbation certificate violated")
    return result


def pair_summary(left, right):
    return {"native_logits": logit_distance(left["logits"], right["logits"]),
            "hidden": distance(left["hidden"], right["hidden"]),
            "seed_kv": distance(left["seed_kv"], right["seed_kv"]),
            "fp32_single_head": logit_distance(left["fp32"], right["fp32"])}


def key(execution, width, mode, future=0, repeat=0):
    return f"{execution}/B{width}/{mode}/f{future}/r{repeat}"


def expected_keys():
    return {key(*args) for args in itertools.product(EXECUTIONS, WIDTHS, MODES, (0, 1), (0, 1))}


def validate(payload):
    if not payload.get("completed") or payload.get("error") or payload.get("stage") != "complete":
        raise ValueError("diagnostic is incomplete")
    if not payload["parameters_frozen_after"] or not payload["environment"]["tracked_source_clean"]:
        raise ValueError("model or source invariant failed")
    rows = payload["contexts"]
    if tuple((r["workload"], r["generation_index"]) for r in rows) != EXPECTED_CONTEXTS:
        raise ValueError("frozen contexts changed")
    for row in rows:
        if set(row["probes"]) != expected_keys():
            raise ValueError("incomplete diagnostic matrix")
        for probe_key, probe in row["probes"].items():
            if not probe["prefix_kv_unchanged"]:
                raise ValueError("prefix KV was modified")
            expected_graph_hits = int(probe_key.startswith("graph/"))
            if probe["graph_hits"] != expected_graph_hits or probe["graph_misses"] != 1 - expected_graph_hits:
                raise ValueError("unexpected execution path")
        expected_pairs = {"width": 18, "future": 24, "repeat": 48, "graph_eager": 12, "mask": 16}
        if {name: len(items) for name, items in row["comparisons"].items()} != expected_pairs:
            raise ValueError("incomplete pair matrix")
    return sum(len(row["probes"]) for row in rows)


def compare_probes(probes):
    comparisons = {name: [] for name in ("width", "future", "repeat", "graph_eager", "mask")}

    def add(group, left, right):
        comparisons[group].append({"left": left, "right": right, **pair_summary(probes[left], probes[right])})

    for execution, width, mode in itertools.product(EXECUTIONS, WIDTHS, MODES):
        if width != 1:
            add("width", key(execution, 1, mode), key(execution, width, mode))
        add("future", key(execution, width, mode), key(execution, width, mode, 1))
        for future in (0, 1):
            add("repeat", key(execution, width, mode, future), key(execution, width, mode, future, 1))
        if execution == "graph":
            add("graph_eager", key(execution, width, mode), key("eager", width, mode))
        if mode != "off":
            add("mask", key(execution, width, "off"), key(execution, width, mode))
    return comparisons


def layer_hook_names(model):
    return {"model.embed_tokens", "model.norm"} | {
        f"model.layers.{index}" for index in range(len(model.model.layers))
    } | {"model.layers.0." + name for name in (
        "input_layernorm", "self_attn.qkv_proj", "self_attn.rotary_emb", "self_attn.attn",
        "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_up_proj", "mlp.act_fn", "mlp.down_proj",
    )}


def probe_context(engine, item, context_index, fp32_weight):
    import torch
    import torch.nn.functional as functional
    from generation import format_chat_prompt
    from nano_vllm_uno.engine.sequence import Sequence
    from nano_vllm_uno.utils.context import reset_context

    runner = engine.model_runner
    prompt_ids = format_chat_prompt(engine.tokenizer, [{"role": "user", "content": item["prompt"]}])[0]
    history = prompt_ids + item["ar_prefix"]
    frontier = len(history) - 1
    if frontier < 1 or frontier + max(WIDTHS) > engine.config.kvcache_block_size:
        raise ValueError("R7D requires a single private KV page")
    seq = Sequence(history[:-1])
    manager = engine.scheduler.block_manager
    manager.allocate(seq, num_cached_blocks=0)
    item.update(history_token_ids=history, frontier=frontier, probes={})
    graph_runner = runner.block_graph_runner
    original_logits = runner.model.compute_logits
    captured_hidden = []
    layers = {}
    hooks = []
    capture_layers = False
    active_layer_capture = {}

    def capture_logits(hidden):
        captured_hidden.append(hidden[0].detach().clone())
        return original_logits(hidden)

    def make_hook(name):
        def hook(module, inputs, output):
            if capture_layers:
                values = output if isinstance(output, tuple) else (output,)
                active_layer_capture[name] = tuple(x[0].detach().clone() for x in values if torch.is_tensor(x))
        return hook

    try:
        input_ids, positions = runner.stage_prefill([seq])
        runner.model(input_ids, positions)
        reset_context()
        seq.num_cached_tokens = frontier
        seq.extend_tokens([history[-1]])
        manager.reserve_blocks_for_forward(seq, max(WIDTHS))
        if len(seq.block_table) != 1:
            raise AssertionError("unexpected extra KV page")
        page = seq.block_table[0]
        prefix_view = runner.kv_cache[:, :, page, :frontier]
        prefix = prefix_view.clone()
        runner.model.compute_logits = capture_logits
        for name, module in runner.model.named_modules():
            if name in layer_hook_names(runner.model):
                hooks.append(module.register_forward_hook(make_hook(name)))
        futures = []
        for future in (0, 1):
            generator = torch.Generator(device="cpu").manual_seed(20270205 + context_index * 100 + future)
            futures.append(torch.randint(0, engine.config.hf_config.vocab_size, (15,), generator=generator).tolist())
        item["future_token_ids"] = futures
        probes = {}
        for execution, width, mode, future, repeat in itertools.product(EXECUTIONS, WIDTHS, MODES, (0, 1), (0, 1)):
            runner.block_graph_runner = graph_runner if execution == "graph" else None
            seq.rollback_kv_to(frontier)
            if not torch.equal(prefix_view, prefix):
                raise AssertionError("committed prefix changed before probe")
            tokens = torch.tensor([[history[-1]] + futures[future][:width - 1]], dtype=torch.long, device="cuda")
            mask = None if mode == "off" else torch.zeros((1, width), dtype=torch.float32, device="cuda")
            if mode == "noise":
                mask[:, 1:] = 1.0
            capture_layers = execution == "eager" and mode == "off" and future == repeat == 0
            active_layer_capture = {}
            captured_hidden.clear()
            graph_before = (runner.cuda_graph_hits, runner.cuda_graph_misses)
            logits = runner._run_block([seq], tokens, lora_mask_batch=mask)[0, 0].detach().clone()
            if len(captured_hidden) != 1:
                raise AssertionError("expected exactly one LM-head call")
            hidden = captured_hidden[0]
            seed_kv = runner.kv_cache[:, :, page, frontier].detach().clone()
            if not torch.equal(prefix_view, prefix):
                raise AssertionError("committed prefix changed during probe")
            single_bf16 = functional.linear(hidden.unsqueeze(0), runner.model.lm_head.weight)[0]
            single_fp32 = functional.linear(hidden.float().unsqueeze(0), fp32_weight)[0]
            probe_key = key(execution, width, mode, future, repeat)
            probes[probe_key] = {"logits": logits.float().cpu(), "hidden": hidden.float().cpu(),
                                 "seed_kv": seed_kv.float().cpu(), "fp32": single_fp32.cpu()}
            item["probes"][probe_key] = {
                "prefix_kv_unchanged": True, "native_logits": logit_summary(logits),
                "graph_hits": runner.cuda_graph_hits - graph_before[0],
                "graph_misses": runner.cuda_graph_misses - graph_before[1],
                "single_bf16_head": logit_summary(single_bf16), "single_fp32_head": logit_summary(single_fp32),
                "native_vs_single_bf16_head": logit_distance(logits.cpu(), single_bf16.cpu()),
                "native_vs_single_fp32_head": logit_distance(logits.cpu(), single_fp32.cpu()),
                "hidden_sha256_float32": tensor_digest(hidden), "seed_kv_sha256_float32": tensor_digest(seed_kv),
            }
            if capture_layers:
                layers[width] = {name: tuple(x.float().cpu() for x in values)
                                 for name, values in active_layer_capture.items()}
        item["comparisons"] = compare_probes(probes)
        item["layer_comparisons_eager_base"] = []
        for width in WIDTHS[1:]:
            for name, reference in layers[1].items():
                candidate = layers[width][name]
                item["layer_comparisons_eager_base"].append({
                    "width": width, "name": name, "outputs": [distance(a, b) for a, b in zip(reference, candidate)]})
        item["prefill_prefix_sha256_float32"] = tensor_digest(prefix)
    finally:
        for hook in hooks:
            hook.remove()
        runner.model.compute_logits = original_logits
        runner.block_graph_runner = graph_runner
        reset_context()
        manager.deallocate(seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "base", "adapter", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a diagnostic record")
    payload = {"schema_version": 1, "scope": "R7D forced-context numerical diagnostic, not performance/quality evaluation",
               "completed": False, "stage": "preflight", "error": None, "contexts": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    engine = None
    try:
        if sha256(args.baseline) != BASELINE_SHA:
            raise ValueError("baseline raw-byte SHA differs from preregistration")
        revision = command(["git", "-C", str(args.source), "rev-parse", "HEAD"])
        dirty = command(["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"])
        if revision != UNO_COMMIT or dirty:
            raise ValueError("unchanged pinned upstream required")
        if sha256(args.base / "model-00000-of-00001.safetensors") != BASE_SHA or sha256(args.adapter / "adapter_model.safetensors") != ADAPTER_SHA:
            raise ValueError("checkpoint SHA mismatch")
        items = select_contexts(json.loads(args.baseline.read_text(encoding="utf-8")))
        if tuple((r["workload"], r["generation_index"]) for r in items) != EXPECTED_CONTEXTS:
            raise ValueError("frozen diagnostic contexts changed")
        sys.path.insert(0, str(args.source))
        import torch
        from nano_vllm_uno import LLM

        torch.set_num_threads(1)
        # FP32 is used ONLY by offline reference head; native BF16 remains unchanged.
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        payload["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                  "triton": importlib.metadata.version("triton"),
                                  "flash_attn": importlib.metadata.version("flash-attn"),
                                  "gpu": torch.cuda.get_device_name(0), "python": sys.version,
                                  "source_revision": revision, "tracked_source_clean": not dirty,
                                  "base_sha256": BASE_SHA, "adapter_sha256": ADAPTER_SHA,
                                  "baseline_sha256": BASELINE_SHA,
                                  "fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                                  "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        payload["config"] = dict(attention_backend="fa2", max_num_seqs=1, max_model_len=2048,
                                 max_num_batched_tokens=2048, gpu_memory_utilization=0.5,
                                 max_diffusion_block_size=16, cuda_graph_block_sizes=list(WIDTHS),
                                 cuda_graph_batch_sizes=[1], fail_on_preemption=True, torch_compile=False,
                                 hf_local_files_only=True, gated_lora_path=str(args.adapter))
        payload["stage"] = "initializing"
        save()
        engine = LLM(model=str(args.base), **payload["config"])
        for parameter in engine.model_runner.model.parameters():
            parameter.requires_grad_(False)
        payload["model_dtype"] = str(engine.config.dtype)
        payload["stage"] = "probing"
        with torch.inference_mode():
            fp32_weight = engine.model_runner.model.lm_head.weight.float()
            for index, item in enumerate(items):
                payload["contexts"].append(item)
                probe_context(engine, item, index, fp32_weight)
                save()
                print(f"{item['workload']} position={item['generation_index']} probes={len(item['probes'])}", flush=True)
        payload["parameters_frozen_after"] = all(not p.requires_grad for p in engine.model_runner.model.parameters())
        payload["stage"], payload["completed"] = "complete", True
        validate(payload)
    except Exception:
        payload["completed"] = False
        payload["error"] = traceback.format_exc()
        print(payload["error"], file=sys.stderr, flush=True)
    finally:
        if engine is not None:
            engine.exit()
        save()
    return 0 if payload["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

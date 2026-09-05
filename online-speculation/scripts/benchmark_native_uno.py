"""Compare fixed-width and online Uno in the pinned native WSL GPU engine.

Python 3.10 standalone entry point; the online wrapper leaves upstream unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


UNO_COMMIT = "ed2ee36bb7a3aea8732ebc635b3f09490a032ea3"
BASE_SHA = "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"
ADAPTER_SHA = "5a499229d19ef4a69eb0b21884819d1b67cd983ba02b7ee2031ba8567dedfe4e"
WORKLOADS = (
    ("english", "Explain how a hash table handles collisions. Compare separate chaining with linear probing, and give one example of each."),
    ("chinese", "解释为什么数据库需要事务。用银行转账的具体例子说明原子性、一致性、隔离性和持久性，并解释常见的并发问题。"),
    ("code", "Write a Python function that merges overlapping intervals. Include type hints, explain its time complexity, and give three unit tests."),
    ("math", "Find the sum of the integers from 1 to 100. Derive the formula for the first n positive integers in two different ways and check small cases."),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def gpu_snapshot():
    return command(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu", "--format=csv,noheader"])


def select_workloads(prompt_file=None, selected=None):
    suite = json.loads(prompt_file.read_text(encoding="utf-8")) if prompt_file else WORKLOADS
    if (not isinstance(suite, (list, tuple)) or not suite
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   or any(not isinstance(v, str) or not v.strip() for v in row) for row in suite)
            or len({row[0] for row in suite}) != len(suite)):
        raise ValueError("prompt suite requires unique nonempty [name, prompt] pairs")
    if selected is None:
        return suite
    names = selected.split(",")
    workloads = [row for row in suite if row[0] in names]
    if len(names) != len(set(names)) or set(names) != {name for name, _ in workloads}:
        raise ValueError("select known workloads without empty or repeated entries")
    return workloads


def paired_method_order(methods, prompt_index, repetition):
    """Each adjacent repetition pair has complementary positions for every arm."""
    offset = (prompt_index + repetition // 2) % len(methods)
    order = list(methods[offset:]) + list(methods[:offset])
    return list(reversed(order)) if repetition % 2 else order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", default="1,4,8,16")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20270005)
    parser.add_argument("--online", action="store_true", help="include request-local width policy (no gradient)")
    parser.add_argument("--shadow", action="store_true", help="fixed B=8 online wrapper control, no adaptive widths")
    parser.add_argument("--fused-norm", action="store_true", help="fuse XLLM grouped RMSNorm; validate numerical differences")
    parser.add_argument("--fast-weights", action="store_true", help="add real last-MLP online LoRA at fixed B=8")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--update-stride", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--replay-blocks", type=int, default=4, help="last R blocks per update, 1 <= R <= stride")
    parser.add_argument("--audit-fast", action="store_true", help="extra replay/change checks, included in TPS cost")
    parser.add_argument("--profile-update", action="store_true", help="profile one warmed online update; diagnostic run only")
    parser.add_argument("--training-backend", choices=("eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--workloads", help="comma-separated names, default all prompts in the chosen suite")
    parser.add_argument("--prompt-file", type=Path, help="fixed JSON list of [name, prompt] pairs")
    args = parser.parse_args()
    if not 1 <= args.replay_blocks <= args.update_stride:
        raise ValueError("1 <= replay blocks <= update stride required")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a previous baseline record")
    blocks = [int(n) for n in args.blocks.split(",")]
    methods = blocks + (["online"] if args.online else []) + (["shadow8"] if args.shadow else []) + (["plain8", "fast8"] if args.fast_weights else [])
    workloads = select_workloads(args.prompt_file, args.workloads)
    if len(set(blocks)) != len(blocks) or 1 not in blocks or min(blocks) < 1 or max(blocks) > 16:
        raise ValueError("distinct widths within 1..16 including AR width 1 required")
    if args.repetitions < 1 or min(args.max_new_tokens, args.warmup_tokens) < 2:
        raise ValueError("positive repetitions and at least two output tokens required")
    if args.online and not {4, 8, 16} <= set(blocks):
        raise ValueError("R7 requires captured and measured static widths 4,8,16")
    if args.shadow and 8 not in blocks:
        raise ValueError("shadow control requires measured static width 8")
    if args.fast_weights and (8 not in blocks or args.online or args.shadow):
        raise ValueError("fast LoRA requires B=8 and is evaluated separately from width policies")
    payload = {
        "schema_version": 1, "scope": "official-runtime engineering baseline; not confirmatory",
        "backend": "pinned Nano-vLLM Uno / WSL2 / FA2 / CUDA graphs; optional scoped runtime extensions",
        "design": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "workloads": workloads, "blocks": blocks, "methods": methods, "temperature": 0.0, "batch_size": 1,
                   "noise_mode": "random_uniform", "ignore_eos": True,
                   "method_roles": {"1": "AR reference", "8": "zero fast-branch control" if args.fast_weights else "static Uno",
                                    "plain8": "static Uno without fast matmuls/addition/feature caching",
                                    "fast8": "Online Uno including all branch, feedback, reset and update costs"},
                   "e2e_scope": "full official generate call including prefill and detokenization; excludes model initialization, shared prompt encoding, GPU snapshots and JSON I/O",
                   "order": "prompt-local reversed pairs; rotated between prompts and repetition pairs",
                   "order_pairing": "reverse_adjacent_repetitions"},
        "completed": False, "stage": "preflight", "records": [], "error": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    engine = None
    try:
        revision = command(["git", "-C", str(args.source), "rev-parse", "HEAD"])
        dirty = command(["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"])
        if revision != UNO_COMMIT or dirty:
            raise RuntimeError("Official baseline requires the unchanged pinned source")
        hashes = {"base": sha256(args.base / "model-00000-of-00001.safetensors"),
                  "adapter": sha256(args.adapter / "adapter_model.safetensors")}
        if hashes != {"base": BASE_SHA, "adapter": ADAPTER_SHA}:
            raise RuntimeError("Checkpoint hash mismatch")
        sys.path.insert(0, str(args.source))
        import torch
        from generation import format_chat_prompt
        from nano_vllm_uno import LLM, SamplingParams
        from native_fast_weights import extended_runner, frozen_digest, generate_fast, plain_uno
        from native_online_policy import NativeWidthPolicy, generate_online

        payload["environment"] = {
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "flash_attn": importlib.metadata.version("flash-attn"),
            "gpu": torch.cuda.get_device_name(0), "uno_commit": revision,
            "tracked_source_clean": not dirty, "checkpoint_sha256": hashes,
        }
        payload["implementation_sha256"] = {
            name: sha256(Path(__file__).parent / name) for name in
            ("benchmark_native_uno.py", "native_fast_weights.py", "native_norm.py", "native_update_graph.py")
        }
        if args.prompt_file:
            payload["prompt_file_sha256"] = sha256(args.prompt_file)
        config = dict(
            attention_backend="fa2", max_num_seqs=1, max_model_len=2048,
            max_num_batched_tokens=2048, gpu_memory_utilization=0.5,
            num_kvcache_blocks=32,
            max_diffusion_block_size=max(blocks), cuda_graph_block_sizes=sorted(blocks),
            cuda_graph_batch_sizes=[1], fail_on_preemption=True, torch_compile=False,
            hf_local_files_only=True, gated_lora_path=str(args.adapter),
        )
        payload["engine_config"] = config
        payload["stage"] = "initializing_engine"
        save()
        start = time.perf_counter()
        with extended_runner(fused_norm=args.fused_norm, fast_weights=args.fast_weights,
                             rank=args.rank, stride=args.update_stride, lr=args.learning_rate,
                             replay_blocks=args.replay_blocks, training_backend=args.training_backend,
                             capture_plain=args.fast_weights):
            engine = LLM(model=str(args.base), **config)
        torch.cuda.synchronize()
        payload["model_initialization_seconds"] = time.perf_counter() - start
        payload["model_dtype"] = str(engine.config.dtype)
        plain_graphs = getattr(engine.model_runner, "plain_block_graph_runner", None)
        payload["plain_control_graphs"] = len(plain_graphs.graphs) if plain_graphs is not None else 0
        if args.profile_update:
            if not args.fast_weights:
                raise ValueError("--profile-update requires --fast-weights")
            state = engine.model_runner.fast_weights
            original_update = state.update

            def profile_update(*update_args, **update_kwargs):
                if state.version != 1 or "update_profile" in payload:
                    return original_update(*update_args, **update_kwargs)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    result = original_update(*update_args, **update_kwargs)
                payload["update_profile"] = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
                print(payload["update_profile"], flush=True)
                return result

            state.update = profile_update
            payload["scope"] = "instrumented update profile; not throughput evidence"
        # Metadata-only freeze, no value changes or replacement of model code.
        for parameter in engine.model_runner.model.parameters():
            parameter.requires_grad_(False)
        payload["parameters_frozen"] = all(not p.requires_grad for p in engine.model_runner.model.parameters())
        payload["fused_norm_count"] = engine.model_runner.fused_norm_count
        payload["frozen_weights_before"] = frozen_digest(engine.model_runner.model)
        encoded = [(name, format_chat_prompt(engine.tokenizer, [{"role": "user", "content": prompt}])[0]) for name, prompt in workloads]
        if any(len(ids) >= engine.config.kvcache_block_size for _, ids in encoded):
            raise RuntimeError("This short-prompt baseline requires no reusable full prefix-cache pages")
        if any(len(ids) + max(args.max_new_tokens, args.warmup_tokens) + max(blocks) > config["max_model_len"] for _, ids in encoded):
            raise ValueError("prompt and output budget exceed the configured context window")

        def generate(ids, width, budget, seed):
            torch.manual_seed(seed)
            params = SamplingParams(
                temperature=0.0, top_k=32, top_p=0.95, max_tokens=budget,
                ignore_eos=True, stop_token_ids=[64019, 1], mask_token_id=64256,
                noise_mode="random_uniform", diffusion_block_size=8 if width in ("online", "shadow8", "plain8", "fast8") else width,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            diagnostics = None
            if width in ("online", "shadow8"):
                policy = NativeWidthPolicy(widths=(8,)) if width == "shadow8" else NativeWidthPolicy()
                output, diagnostics = generate_online(engine, ids, params, budget, policy)
            elif width == "fast8":
                output, diagnostics = generate_fast(engine, ids, params, budget, audit=args.audit_fast)
            elif width == "plain8":
                with plain_uno(engine):
                    output = engine.generate([ids], params, use_tqdm=False, request_max_tokens=[budget])[0]
            else:
                output = engine.generate([ids], params, use_tqdm=False, request_max_tokens=[budget])[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            return output, elapsed, diagnostics

        payload["stage"] = "warmup"
        save()
        for width in methods:
            generate(encoded[0][1], width, args.warmup_tokens, args.seed - 1)
        payload["stage"] = "measuring"
        save()
        for prompt_index, (name, ids) in enumerate(encoded):
            for rep in range(args.repetitions):
                seed = args.seed + prompt_index * 100 + rep
                order = paired_method_order(methods, prompt_index, rep)
                for width in order:
                    before = gpu_snapshot()
                    graph_before = (engine.model_runner.cuda_graph_hits, engine.model_runner.cuda_graph_misses)
                    output, elapsed, diagnostics = generate(ids, width, args.max_new_tokens, seed)
                    record = {"workload": name, "seed": seed, "block_size": width, "order": order,
                              "prompt_tokens": len(ids), "output": output, "end_to_end_seconds": elapsed,
                              "output_tokens": len(output["token_ids"]),
                              "e2e_tps": len(output["token_ids"]) / elapsed,
                              "gpu_before": before, "gpu_after": gpu_snapshot(), "online": diagnostics,
                              "cuda_graph_hits": engine.model_runner.cuda_graph_hits - graph_before[0],
                              "cuda_graph_misses": engine.model_runner.cuda_graph_misses - graph_before[1]}
                    payload["records"].append(record)
                    save()
                    print(f"{name} seed={seed} B={width} TPS={record['e2e_tps']:.2f} stats={output['stats']}", flush=True)
        payload["engine_stats"] = engine.get_stats()
        payload["parameters_frozen_after"] = all(not p.requires_grad for p in engine.model_runner.model.parameters())
        payload["frozen_weights_after"] = frozen_digest(engine.model_runner.model)
        if payload["frozen_weights_before"] != payload["frozen_weights_after"]:
            raise RuntimeError("Frozen teacher/offline Uno tensor bytes changed")
        payload["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        payload["stage"] = "complete"
        payload["completed"] = True
    except Exception:
        payload["error"] = traceback.format_exc()
        print(payload["error"], file=sys.stderr, flush=True)
    finally:
        if engine is not None:
            engine.exit()
        save()
    return 0 if payload["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

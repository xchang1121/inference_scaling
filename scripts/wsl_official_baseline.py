"""Measure the pinned, unmodified Uno engine in a persistent WSL GPU process.

Python 3.10 standalone entry point; does not install the Windows HF prototype.
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a previous baseline record")
    blocks = [int(n) for n in args.blocks.split(",")]
    if len(set(blocks)) != len(blocks) or 1 not in blocks or min(blocks) < 1 or max(blocks) > 16:
        raise ValueError("distinct widths within 1..16 including AR width 1 required")
    if args.repetitions < 1 or min(args.max_new_tokens, args.warmup_tokens) < 2:
        raise ValueError("positive repetitions and at least two output tokens required")
    payload = {
        "schema_version": 1, "scope": "official-runtime engineering baseline; not confirmatory",
        "backend": "unmodified pinned Nano-vLLM Uno / WSL2 / FA2 / CUDA graphs",
        "design": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "workloads": WORKLOADS, "blocks": blocks, "temperature": 0.0, "batch_size": 1,
                   "noise_mode": "random_uniform", "ignore_eos": True,
                   "e2e_scope": "full official generate call including prefill and detokenization; excludes model initialization, shared prompt encoding, GPU snapshots and JSON I/O",
                   "order": "rotated and alternately reversed within prompt/repetition"},
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

        payload["environment"] = {
            "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "flash_attn": importlib.metadata.version("flash-attn"),
            "gpu": torch.cuda.get_device_name(0), "uno_commit": revision,
            "tracked_source_clean": not dirty, "checkpoint_sha256": hashes,
        }
        config = dict(
            attention_backend="fa2", max_num_seqs=1, max_model_len=2048,
            max_num_batched_tokens=2048, gpu_memory_utilization=0.5,
            max_diffusion_block_size=max(blocks), cuda_graph_block_sizes=sorted(blocks),
            cuda_graph_batch_sizes=[1], fail_on_preemption=True, torch_compile=False,
            hf_local_files_only=True, gated_lora_path=str(args.adapter),
        )
        payload["engine_config"] = config
        payload["stage"] = "initializing_engine"
        save()
        start = time.perf_counter()
        engine = LLM(model=str(args.base), **config)
        torch.cuda.synchronize()
        payload["model_initialization_seconds"] = time.perf_counter() - start
        payload["model_dtype"] = str(engine.config.dtype)
        # Metadata-only freeze, no value changes or replacement of model code.
        for parameter in engine.model_runner.model.parameters():
            parameter.requires_grad_(False)
        payload["parameters_frozen"] = all(not p.requires_grad for p in engine.model_runner.model.parameters())
        encoded = [(name, format_chat_prompt(engine.tokenizer, [{"role": "user", "content": prompt}])[0]) for name, prompt in WORKLOADS]
        if any(len(ids) >= engine.config.kvcache_block_size for _, ids in encoded):
            raise RuntimeError("This short-prompt baseline requires no reusable full prefix-cache pages")

        def generate(ids, width, budget, seed):
            torch.manual_seed(seed)
            params = SamplingParams(
                temperature=0.0, top_k=32, top_p=0.95, max_tokens=budget,
                ignore_eos=True, stop_token_ids=[64019, 1], mask_token_id=64256,
                noise_mode="random_uniform", diffusion_block_size=width,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = engine.generate([ids], params, use_tqdm=False, request_max_tokens=[budget])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            return outputs[0], elapsed

        payload["stage"] = "warmup"
        save()
        for width in blocks:
            generate(encoded[0][1], width, args.warmup_tokens, args.seed - 1)
        payload["stage"] = "measuring"
        save()
        for prompt_index, (name, ids) in enumerate(encoded):
            for rep in range(args.repetitions):
                seed = args.seed + prompt_index * 100 + rep
                offset = (prompt_index + rep) % len(blocks)
                order = blocks[offset:] + blocks[:offset]
                if (prompt_index + rep) % 2:
                    order = list(reversed(order))
                for width in order:
                    before = gpu_snapshot()
                    output, elapsed = generate(ids, width, args.max_new_tokens, seed)
                    record = {"workload": name, "seed": seed, "block_size": width, "order": order,
                              "prompt_tokens": len(ids), "output": output, "end_to_end_seconds": elapsed,
                              "output_tokens": len(output["token_ids"]),
                              "e2e_tps": len(output["token_ids"]) / elapsed,
                              "gpu_before": before, "gpu_after": gpu_snapshot()}
                    payload["records"].append(record)
                    save()
                    print(f"{name} seed={seed} B={width} TPS={record['e2e_tps']:.2f} stats={output['stats']}", flush=True)
        payload["engine_stats"] = engine.get_stats()
        payload["parameters_frozen_after"] = all(not p.requires_grad for p in engine.model_runner.model.parameters())
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

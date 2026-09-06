"""Pinned public-weight audit and paired dual-view throughput measurement."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from blockspec.checkpoint import implementation_fingerprint
from blockspec.parallel import MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.weights import file_sha256, load_public
from blockspec.sampling import SamplingConfig
from blockspec.state import cache_length


MODEL_REVISION = "7dc9735acb89bb17dd2f3c5689928707c1fb0868"
WEIGHT_SHA = "1c1e0d155c298095b9fe7b8a84a7e6340ccf31009fdae62539f213d037103e7c"
SOURCE_SHA = "7e8f4e01c4c8c469866839976acd6fa45dec207af40d939e0a46b8918e413334"
PROMPTS = (
    "Explain how binary search works. Give a Python implementation and discuss its time complexity.",
    "Solve step by step: A train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed?",
    "Write a Python function that merges two sorted lists. Explain the invariant used in its loop.",
    "Explain the difference between a process and a thread, including memory sharing and synchronization.",
    "Find the sum of the first 100 positive integers and derive the general formula for the first n integers.",
    "Describe the water cycle, including evaporation, condensation, precipitation, and collection.",
    "Write a Python function to check whether brackets in a string are balanced, with examples and an explanation.",
    "Compare breadth-first search with depth-first search, including their data structures and typical uses.",
)


def load_reference(folder, dtype, backend):
    source = folder / "modeling_orthrus.py"
    if file_sha256(source) != SOURCE_SHA:
        raise ValueError("reference source SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("blockspec_pinned_orthrus", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OrthrusLM.from_pretrained(folder, dtype=dtype, attn_implementation=backend,
                                            local_files_only=True).cuda().eval().requires_grad_(False)


def prompt_ids(folder, count):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    result = []
    for text in PROMPTS[:count]:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        result.append(tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.cuda())
    return result


def compare_logits(actual, reference):
    a, b = actual.float(), reference.float()
    tv = 0.5 * (a.softmax(-1) - b.softmax(-1)).abs().sum(-1)
    return {"max_logit_error": float((a - b).abs().max()),
            "mean_logit_error": float((a - b).abs().mean()),
            "max_tv": float(tv.max()), "mean_tv": float(tv.mean()),
            "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}


def compare_cache(actual, reference):
    error, scaled_error = 0.0, 0.0
    for (key, value), layer in zip(actual, reference.layers, strict=True):
        if key.shape != layer.keys.shape or value.shape != layer.values.shape:
            raise AssertionError("reference cache shapes differ")
        tolerance = 0.0002 if key.dtype == torch.float32 else 0.05
        for actual_tensor, expected_tensor in ((key, layer.keys), (value, layer.values)):
            delta = (actual_tensor.float() - expected_tensor.float()).abs()
            error = max(error, float(delta.max()))
            scale = tolerance * (1 + expected_tensor.float().abs())
            scaled_error = max(scaled_error, float((delta / scale).max()))
    return {"max_kv_error": error, "max_kv_tolerance_ratio": scaled_error,
            "kv_atol_rtol": tolerance}


@torch.inference_mode()
def audit(args, own, reference):
    records = []
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    sequence = torch.randint(100, own.config.vocab_size - 1000, (1, 80), device="cuda", generator=rng)
    for prefix_length in (7, 29):
        prefix = sequence[:, :prefix_length]
        for block in args.blocks:
            ar = own(prefix)
            oracle = reference(input_ids=prefix, use_cache=True)
            records.append({"stage": "ar_prefill", "prefix": prefix_length, "block": block,
                            **compare_logits(ar.logits, oracle.logits),
                            **compare_cache(ar.cache, oracle.past_key_values)})
            draft_ids = prefix.new_full((1, block), own.config.mask_token_id)
            draft_ids[:, 0] = sequence[:, prefix_length]
            positions = torch.arange(prefix_length, prefix_length + block, device="cuda")[None]
            before = [(layer.keys.clone(), layer.values.clone()) for layer in oracle.past_key_values.layers]
            draft = own(draft_ids, view="draft", cache=ar.cache, positions=positions)
            oracle_draft = reference(input_ids=draft_ids, past_key_values=oracle.past_key_values,
                                     position_ids=positions, use_cache=False,
                                     is_diffusion_pass=True, ar_seq_len=prefix_length)
            for (key, value), layer in zip(before, oracle.past_key_values.layers, strict=True):
                if not torch.equal(key, layer.keys) or not torch.equal(value, layer.values):
                    raise AssertionError("reference draft mutated the persistent cache")
            records.append({"stage": "draft", "prefix": prefix_length, "block": block,
                            **compare_logits(draft.logits, oracle_draft.logits)})
            candidates = torch.cat((draft_ids[:, :1], draft.logits[:, :-1].argmax(-1)), 1)
            verifier = own(candidates, cache=ar.cache)
            oracle_verifier = reference(input_ids=candidates, past_key_values=oracle.past_key_values,
                                        position_ids=positions, use_cache=True)
            records.append({"stage": "ar_verify", "prefix": prefix_length, "block": block,
                            **compare_logits(verifier.logits, oracle_verifier.logits),
                            **compare_cache(verifier.cache, oracle_verifier.past_key_values)})
    limit = 0.0005 if args.dtype == "float32" else 0.5
    tv_limit = 0.0001 if args.dtype == "float32" else 0.02
    numerical_pass = all(row["max_logit_error"] <= limit and row["max_tv"] <= tv_limit
                         and row.get("max_kv_tolerance_ratio", 0) <= 1 for row in records)
    prompt = prompt_ids(args.model, 1)[0]
    own_ids = generate(MaskedAttentionBranch(own), prompt, args.tokens, block_size=args.blocks[-1],
                       eos_id=own.config.eos_token_id).tokens
    reference.config.block_size = args.blocks[-1]
    reference_ids = reference.generate(input_ids=prompt, max_new_tokens=args.tokens,
                                       temperature=0, top_k=0, top_p=1,
                                       eos_token_id=own.config.eos_token_id)[0, prompt.shape[1]:].tolist()
    ar_ids = generate_ar(MaskedAttentionBranch(own), prompt, args.tokens,
                         eos_id=own.config.eos_token_id).tokens
    greedy_pass = own_ids == reference_ids == ar_ids
    return {"records": records, "thresholds": {"max_logit_error": limit, "max_tv": tv_limit},
            "numerical_pass": numerical_pass, "greedy_pass": greedy_pass,
            "greedy_tokens": len(own_ids), "own_tokens": own_ids,
            "reference_tokens": reference_ids, "ar_tokens": ar_ids,
            "pass": numerical_pass and greedy_pass}


def reference_generate(model, prompt, tokens, block, sampling, seed):
    counts = {"prefill": 0, "ar": 0, "draft": 0}

    def count(module, positional, kwargs):
        if sum(counts.values()) == 0:
            counts["prefill"] += 1
        else:
            counts["draft" if kwargs.get("is_diffusion_pass", False) else "ar"] += 1

    hook = model.register_forward_pre_hook(count, with_kwargs=True)
    model.config.block_size = max(block, 2)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        output = model.generate(input_ids=prompt, max_new_tokens=tokens,
                                temperature=sampling.temperature, top_k=sampling.top_k,
                                top_p=sampling.top_p, use_diffusion_mode=block > 1,
                                do_sample=sampling.temperature > 0,
                                eos_token_id=model.config.eos_token_id)
        torch.cuda.synchronize()
    finally:
        hook.remove()
    elapsed = time.perf_counter() - start
    token_ids = output[0, prompt.shape[1]:].tolist()
    return {"tokens": len(token_ids), "seconds": elapsed,
            "prefill_forwards": counts["prefill"], "prefill_output_tokens": 1,
            "decode_forwards": counts["ar"] + counts["draft"], "token_ids": token_ids}


@torch.inference_mode()
def trace_decision(args, own):
    """Compare one greedy decision, including a same-cache one-row/block replay."""
    prompt = prompt_ids(args.model, args.request_index + 1)[args.request_index]
    branch, captures, output_ids = MaskedAttentionBranch(own), {}, {}
    for mode in ("ar", "speculative"):
        selected = {}

        def capture(module, positional, kwargs, output, captured=selected):
            if kwargs.get("view", "ar") == "draft":
                return
            first = cache_length(kwargs.get("cache")) + positional[0].shape[1] - output.logits.shape[1] + 1
            row = prompt.shape[1] + args.token_index - first
            if 0 <= row < output.logits.shape[1]:
                captured.update(logits=output.logits[0, row].float().clone(), cache=kwargs.get("cache"),
                                inputs=positional[0].clone(), row=row)

        hook = own.register_forward_hook(capture, with_kwargs=True)
        try:
            output = (generate_ar(branch, prompt, args.tokens) if mode == "ar" else
                      generate(branch, prompt, args.tokens, block_size=args.blocks[-1]))
        finally:
            hook.remove()
        output_ids[mode] = output.tokens
        captures[mode] = selected
    if not all(captures.values()):
        raise ValueError("decision was outside the generated token range")
    first_difference = next((index for index, (left, right) in enumerate(zip(
        output_ids["ar"], output_ids["speculative"])) if left != right), None)
    ar, spec = captures["ar"]["logits"], captures["speculative"]["logits"]
    ar_id, spec_id = int(ar.argmax()), int(spec.argmax())
    saved = captures["ar"]
    future = prompt.new_full((1, args.blocks[-1]), own.config.mask_token_id)
    future[:, :1] = saved["inputs"][:, -1:]
    single = own(future[:, :1], cache=saved["cache"]).logits[0, 0]
    block = own(future, cache=saved["cache"]).logits[0, 0]
    return {"request_index": args.request_index, "token_index": args.token_index,
            "first_difference": first_difference, "prefix_identical":
                output_ids["ar"][:args.token_index] == output_ids["speculative"][:args.token_index],
            "actual_path": compare_logits(ar[None], spec[None]),
            "ar_choice": ar_id, "speculative_choice": spec_id,
            "ar_gap_between_choices": float(ar[ar_id] - ar[spec_id]),
            "block_gap_between_choices": float(spec[ar_id] - spec[spec_id]),
            "same_cache_single_vs_block": compare_logits(single[None], block[None]),
            "same_cache_single_choice": int(single.argmax()), "same_cache_block_choice": int(block.argmax())}


@torch.inference_mode()
def benchmark(args, own, reference):
    prompts = prompt_ids(args.model, args.requests)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    methods = [(implementation, block) for implementation, model in (("own", own), ("reference", reference))
               if model is not None for block in [1] + args.blocks]
    branch = None if own is None else MaskedAttentionBranch(own)

    def run(implementation, block, prompt, count, seed):
        if implementation == "reference":
            return reference_generate(reference, prompt, count, block, sampling, seed)
        options = {"sampling": sampling, "eos_id": own.config.eos_token_id,
                   "generator": torch.Generator(device="cuda").manual_seed(seed)}
        result = (generate_ar(branch, prompt, count, **options) if block == 1 else
                  generate(branch, prompt, count, block_size=block, **options))
        return result.summary() | {"token_ids": result.tokens}

    for implementation, block in methods:
        run(implementation, block, prompts[0], min(32, args.tokens), args.seed)
    records = []
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    for repeat in range(args.repeats):
        for index, prompt in enumerate(prompts):
            ordered = methods if (repeat + index) % 2 == 0 else list(reversed(methods))
            for implementation, block in ordered:
                row = run(implementation, block, prompt, args.tokens, args.seed + repeat * 1000 + index)
                row.update(implementation=implementation, block=block, request=index, repeat=repeat,
                           input_tokens=prompt.numel())
                records.append(row)
    aggregate = {}
    for implementation, block in methods:
        key = f"{implementation}_k{block}"
        rows = [row for row in records if (row["implementation"], row["block"]) == (implementation, block)]
        total, seconds = sum(row["tokens"] for row in rows), sum(row["seconds"] for row in rows)
        forwards = sum(row["decode_forwards"] for row in rows)
        aggregate[key] = {"tokens": total, "seconds": seconds, "tps": total / seconds,
                          "decode_tpf": (total - len(rows)) / forwards if forwards else None}
    random = np.random.default_rng(args.seed)
    for implementation, block in methods:
        key, base = f"{implementation}_k{block}", f"{implementation}_k1"
        aggregate[key]["speedup_vs_matched_ar"] = aggregate[key]["tps"] / aggregate[base]["tps"]
        if block == 1:
            continue
        grouped = {}
        identical = True
        for index in range(len(prompts)):
            values = []
            for size in (1, block):
                rows = [row for row in records if row["implementation"] == implementation
                        and row["block"] == size and row["request"] == index]
                values.extend((sum(row["tokens"] for row in rows), sum(row["seconds"] for row in rows)))
            grouped[index] = values
            if sampling.temperature == 0:
                for repeat in range(args.repeats):
                    rows = [row for row in records if row["implementation"] == implementation
                            and row["block"] in (1, block) and row["request"] == index and row["repeat"] == repeat]
                    identical &= rows[0]["token_ids"] == rows[1]["token_ids"]
        values = np.asarray(list(grouped.values()))
        selected = random.integers(0, len(values), size=(2000, len(values)))
        summed = values[selected].sum(1)
        ratios = (summed[:, 2] / summed[:, 3]) / (summed[:, 0] / summed[:, 1])
        aggregate[key]["paired_request_ci95"] = np.quantile(ratios, [.025, .975]).tolist()
        aggregate[key]["greedy_identical"] = identical if sampling.temperature == 0 else None
    return {"sampling": asdict(sampling), "workload": "eight fixed smoke prompts; chat template with thinking disabled",
            "prompts": list(PROMPTS[:args.requests]), "records": records, "aggregate": aggregate,
            "resident_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "benchmark", "trace"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--backend", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--implementation", choices=("own", "reference", "both"), default="own")
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--request-index", type=int, default=6)
    parser.add_argument("--token-index", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.tokens < 1 or not 1 <= args.requests <= len(PROMPTS) or args.repeats < 1
            or any(block < 2 for block in args.blocks) or len(set(args.blocks)) != len(args.blocks)):
        parser.error("positive budgets, unique blocks >= 2, and 1..8 requests required")
    if args.output is not None and args.output.exists():
        parser.error("choose a new output file")
    if args.mode == "trace" and (args.implementation == "reference" or not 0 <= args.request_index < len(PROMPTS)
                                  or not 0 <= args.token_index < args.tokens):
        parser.error("trace requires the independent model and valid request/token indices")
    torch.set_num_threads(1)
    own = reference = None
    dtype = getattr(torch, args.dtype)
    if args.mode == "audit" or args.implementation in ("own", "both"):
        own = load_public(args.model, device="cuda", dtype=dtype, expected_sha256=WEIGHT_SHA).set_backend(args.backend)
    if args.mode == "audit" or args.implementation in ("reference", "both"):
        if own is None and file_sha256(args.model / "model.safetensors") != WEIGHT_SHA:
            raise ValueError("reference weight SHA256 mismatch")
        reference = load_reference(args.model, dtype, args.backend)
    result = {"mode": args.mode, "model_revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA,
              "reference_source_sha256": SOURCE_SHA, "implementation": implementation_fingerprint(),
              "script_sha256": file_sha256(__file__), "dtype": args.dtype, "backend": args.backend,
              "python": platform.python_version(), "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "seed": args.seed}
    if args.mode == "audit":
        result.update(audit(args, own, reference))
    elif args.mode == "trace":
        result.update(trace_decision(args, own))
    else:
        result.update(benchmark(args, own, reference))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result if args.mode == "audit" else {key: value for key, value in result.items()
                                                        if key != "records"}, indent=2))
    if args.mode == "audit" and not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

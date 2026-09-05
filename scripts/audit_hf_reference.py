"""Optional, pinned full-base oracle. Never used by training or generation.

This explicitly executes the reviewed model author's local Python, but only
after verifying its source, config and weight hashes. No download or outputs
are written by this script. Transformers can create its ordinary import cache.
Use an isolated dependency path for this oracle; do not upgrade the training
environment just to import the model's newer reference implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from blockspec.checkpoint import implementation_fingerprint, load_hf_base


LOCK = Path(__file__).resolve().parents[1] / "references" / "upstream.lock.json"


def checked_reference(directory):
    spec = json.loads(LOCK.read_text(encoding="utf-8"))["models"]["k2_1b_base"]
    for name, expected in spec["reference_lf_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"reference source/config differs from reviewed pin: {name}")
    digest = hashlib.sha256()
    with (directory / spec["weight_filename"]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["weight_sha256"]:
        raise ValueError("reference weights differ from reviewed pin")
    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if set(index["weight_map"].values()) != {spec["weight_filename"]}:
        raise ValueError("reference index points outside the checked weight shard")
    return spec


def error_summary(actual, expected):
    a, e = actual.float(), expected.float()
    if a.shape != e.shape or not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise ValueError("invalid oracle logit tensors")
    delta = (a - e).abs()
    tv = 0.5 * (a.softmax(-1) - e.softmax(-1)).abs().sum(-1)
    return {"positions": a.numel() // a.shape[-1], "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(), "max_tv": tv.max().item(), "mean_tv": tv.mean().item(),
            "argmax_mismatches": int((a.argmax(-1) != e.argmax(-1)).sum())}


def summarize(rows):
    count = sum(r["positions"] for r in rows)
    return {"positions": count, "argmax_mismatches": sum(r["argmax_mismatches"] for r in rows),
            **{k: max(r[k] for r in rows) for k in ("max_abs", "max_tv")},
            **{k: sum(r[k] * r["positions"] for r in rows) / count for k in ("mean_abs", "mean_tv")}}


@torch.no_grad()
def audit(ours, oracle, prompts, *, tokens, executor=None):
    forward = ours if executor is None else executor
    prefill, incremental, shifted = [], [], []
    for prompt in prompts:
        actual, cache = forward(prompt, return_cache=True)
        expected = oracle(prompt, use_cache=True)
        prefill.append(error_summary(actual, expected.logits))
        ref_cache = expected.past_key_values
        # Both see the reference continuation. Zero mismatches implies identical
        # independent greedy continuations on this finite probe, not all prompts.
        seed = expected.logits[:, -1:].argmax(-1)
        for _ in range(tokens):
            actual, cache = forward(seed, cache=cache, return_cache=True)
            expected = oracle(seed, past_key_values=ref_cache, use_cache=True)
            ref_cache = expected.past_key_values
            incremental.append(error_summary(actual, expected.logits))
            seed = expected.logits.argmax(-1)
        for offset in (8192, 65536):
            ids = prompt[:, :min(32, prompt.shape[1])]
            positions = torch.arange(ids.shape[1], device=ids.device)[None] + offset
            actual = forward(ids, positions=positions)
            expected = oracle(ids, position_ids=positions, use_cache=False).logits
            shifted.append(error_summary(actual, expected))
    return {"prefill": summarize(prefill), "incremental": summarize(incremental),
            "shifted_position_short_windows": summarize(shifted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Optional existing development token JSONL; never the sealed test set")
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--execution", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--max-logit-error", type=float, default=5e-4)
    parser.add_argument("--max-tv", type=float, default=1e-4)
    parser.add_argument("--require-same-argmax", action="store_true")
    args = parser.parse_args()
    if min(args.prompts, args.prompt_length, args.tokens) < 1:
        parser.error("positive prompt count, prompt length and continuation length required")
    if not (0 < args.max_logit_error < float("inf") and 0 < args.max_tv <= 1):
        parser.error("finite positive logit tolerance and TV tolerance in (0,1] required")
    implementation_sha = implementation_fingerprint()
    spec = checked_reference(args.base)
    import transformers
    if transformers.__version__ != spec["reference_transformers"]:
        raise RuntimeError(f"oracle requires isolated Transformers {spec['reference_transformers']}; "
                           f"found {transformers.__version__}")
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(4)
    torch.manual_seed(271828)
    ours = load_hf_base(args.base, rank=0, device=args.device, dtype=dtype).eval()
    oracle = transformers.AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, trust_remote_code=True, dtype=dtype,
        attn_implementation=args.attention,
    ).to(args.device).eval()
    state = oracle.state_dict()
    own = ours.state_dict()
    if own.keys() != state.keys() or any(not torch.equal(own[n], state[n]) for n in own):
        raise RuntimeError("independent and reference base tensors differ")
    if args.data:
        from blockspec.benchmark import continuation_prompts
        from blockspec.data import load_sequences
        sequences = load_sequences(args.data, vocab_size=ours.config.vocab_size)
        prompts = continuation_prompts(sequences, count=args.prompts, length=args.prompt_length)
        data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    else:
        rng = torch.Generator().manual_seed(271828)
        prompts = [torch.randint(ours.config.vocab_size, (1, args.prompt_length), generator=rng)
                   for _ in range(args.prompts)]
        data_hash = None
    prompts = [p.to(args.device) for p in prompts]
    executor = None
    if args.execution == "cuda_graph":
        from blockspec.execution import FixedShapeExecutor
        executor = FixedShapeExecutor(ours, capacity=args.prompt_length + args.tokens, max_query=32)
        executor.prepare([(1, False, None), (min(32, args.prompt_length), False, None)])
        if args.prompt_length <= 32:
            executor.prepare([(args.prompt_length, False, None)])
    result = audit(ours, oracle, prompts, tokens=args.tokens, executor=executor)
    passed = all(row["max_abs"] <= args.max_logit_error and row["max_tv"] <= args.max_tv
                 and (not args.require_same_argmax or row["argmax_mismatches"] == 0) for row in result.values())
    print(json.dumps({"reference": spec, "transformers": transformers.__version__,
                      "torch": torch.__version__, "device": str(next(ours.parameters()).device),
                      "dtype": args.dtype, "reference_attention": args.attention,
                      "identical_base_tensors": True, "data_sha256": data_hash,
                      "prompt_length": args.prompt_length, "prompts": args.prompts,
                      "incremental_steps_per_prompt": args.tokens,
                      "implementation_sha256_at_start": implementation_sha, "errors": result,
                      "execution": args.execution, "numeric_gate_passed": passed,
                      "gate": {"max_logit_error": args.max_logit_error, "max_tv": args.max_tv,
                               "require_same_argmax": args.require_same_argmax},
                      "scope": "full external base numerical audit; short windows at shifted positions "
                               "are not a long-context quality evaluation; no speed measurement"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

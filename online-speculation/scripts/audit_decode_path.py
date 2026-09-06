"""Inspect target logits on the common history at a chosen generation position."""

import argparse
import json

import torch

from blockspec.adapter_io import load_peft_adapter, peft_config
from blockspec.benchmark import compare_tokens, continuation_prompts
from blockspec.checkpoint import implementation_fingerprint, load_hf_base
from blockspec.data import load_sequences
from blockspec.decoding import generate_ar
from blockspec.execution import FixedShapeExecutor
from blockspec.model import cache_length
from blockspec.tree import generate_tree


class TargetTrace:
    """Wrap an executor and retain base rows at one absolute logit position."""

    def __init__(self, engine, position):
        self.engine, self.position, self.rows = engine, position, []

    def validate(self, model):
        self.engine.validate(model)

    def _forward(self, tokens, **kwargs):
        result = self.engine._forward(tokens, **kwargs)
        prefix = cache_length(kwargs.get("cache"))
        positions = kwargs.get("positions")
        if positions is None:
            positions = torch.arange(prefix, prefix + tokens.shape[1], device=tokens.device)[None]
        mask, allowed = kwargs.get("adapter_mask"), kwargs.get("allowed")
        for row in (positions[0] == self.position).nonzero().flatten().tolist():
            if mask is not None and bool(mask[0, row]):
                continue
            visible = (torch.arange(tokens.shape[1], device=tokens.device) <= row if allowed is None else
                       allowed[0, 0, row, prefix:])
            self.rows.append({"prefix": prefix, "path": tokens[0, visible].tolist(),
                              "kind": "draft_root" if mask is not None else "base",
                              "logits": result[0][0, row].detach().cpu().clone()})
        return result


def compare_target_rows(reference, actual):
    p, q = reference.softmax(-1), actual.softmax(-1)
    top = reference.topk(2)
    chosen = actual.topk(2)
    return {"max_logit_error": float((reference - actual).abs().max()),
            "tv": float((p - q).abs().sum() / 2),
            "ar_argmax": int(reference.argmax()), "spec_argmax": int(actual.argmax()),
            "ar_top_ids": top.indices.tolist(), "ar_top_logits": top.values.tolist(),
            "ar_margin": float(top.values[0] - top.values[1]),
            "spec_top_ids": chosen.indices.tolist(), "spec_top_logits": chosen.values.tolist()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "reference-sha256", "data"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--request", type=int, required=True)
    parser.add_argument("--token-index", type=int, required=True, help="zero-based generated token index")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prefix-budget", type=int, default=16)
    parser.add_argument("--seed", type=int, default=271844)
    args = parser.parse_args()
    if args.request < 0 or not 0 <= args.token_index < args.tokens:
        parser.error("valid request and generation index required")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    config = peft_config(args.adapter)
    model = load_hf_base(args.base, rank=config["r"], alpha=config["lora_alpha"], device="cuda")
    load_peft_adapter(args.adapter, model, expected_sha256=args.reference_sha256)
    model.set_attention_backend("grouped")
    prompt = continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                   count=args.request + 1, length=args.prompt_length)[args.request].cuda()
    maximum = max(args.block_size, args.prefix_budget)
    engine = FixedShapeExecutor(model, capacity=args.prompt_length + args.tokens, max_query=maximum)
    engine.prepare([(n, False, None) for n in range(1, maximum + 1)] +
                   [(n, True, None) for n in range(2, args.block_size + 1)])
    position = args.prompt_length - 1 + args.token_index
    ar_trace, spec_trace = TargetTrace(engine, position), TargetTrace(engine, position)
    ar = generate_ar(model, prompt, args.tokens, executor=ar_trace)
    spec = generate_tree(model, prompt, args.tokens, executor=spec_trace, block_size=args.block_size,
                         top_k=args.top_k, prefix_budget=args.prefix_budget,
                         generator=torch.Generator(device="cuda").manual_seed(args.seed))
    comparison = compare_tokens(ar.tokens, spec.tokens)
    if comparison["common_prefix"] < args.token_index:
        raise ValueError("choose a token index on the common history or its first differing prediction")
    history = prompt[0].tolist() + ar.tokens
    rows = [{"prefix": row["prefix"], "path": row["path"], "kind": row["kind"],
             **compare_target_rows(ar_trace.rows[0]["logits"], row["logits"])} for row in spec_trace.rows
            if row["path"] == history[row["prefix"]:position + 1]]
    if not rows:
        raise RuntimeError("the trace contains no matching ancestral target row")
    print(json.dumps({"implementation_sha256": implementation_fingerprint(), "request": args.request,
                      "token_index": args.token_index, "comparison": comparison, "target_rows": rows,
                      "scope": "instrumented numerical audit; elapsed time is excluded from TPS evidence"}))


if __name__ == "__main__":
    main()

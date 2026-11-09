"""Check fixed conditional states after a failed seeded cache-continuation test.

This post-hoc diagnostic never replaces the original test. Reuse its exact
parent, references, seeds, and the first recorded full eight-token cached path.
At each fixed-history position compare one greedy token and its log probability
with cached versus reset prefill. No example or parameter search is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.backends.vllm_backend import _generated_logprobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing diagnostics; select a new output path")
    original = json.loads(args.validation.read_text(encoding="utf-8"))
    parent = tuple(original["parent_ids"])
    path_branch = next(b for b in original["branches"]
                       if len(b["cached"]["token_ids"]) == original["sample_tokens"])
    history = tuple(path_branch["cached"]["token_ids"])
    settings = load_settings(args.settings)
    if not settings["ar"]["engine"]["vllm"]["enable_prefix_caching"]:
        raise ValueError("requires prefix caching")
    result = {
        "kind": "posthoc_fixed_history_conditional_cache_diagnostic",
        "original_validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        "original_validation_passed": original["passed"],
        "source_question": original["source_question"], "source_arm": original["source_arm"],
        "parent_ids": parent, "parent_sha256": original["parent_sha256"],
        "history_branch": path_branch["branch"], "fixed_history_ids": history,
        "seeds": [original["seed"] + i for i in range(len(original["branches"]))],
        "policy": "one greedy token per fixed-history position; no sampled history feedback",
        "absolute_logprob_tolerance": original["absolute_logprob_tolerance"],
        "conditional_checks_passed": False,
    }
    raw = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = load_backend(settings["ar"]["model"], settings["ar"]["engine"], seed=original["seed"], logprobs=1)
        result["model_id"] = raw.model_id

        def reset():
            async def perform():
                if await raw._engine.reset_prefix_cache() is False:
                    raise RuntimeError("prefix cache reset failed")
            raw._runner.run(perform())

        def sample(prefix, branch, label):
            # SamplingConfig describes positive-temperature sampling. Use the
            # same native factory as direct_generate for this greedy probe,
            # retaining sampled-token log probabilities and native cache counts.
            params = raw._sampling_params_factory(
                max_tokens=1, temperature=0.0, top_p=1.0, top_k=0,
                seed=original["seed"] + branch, logprobs=0, flat_logprobs=raw._flat_logprobs,
                logit_bias=raw._logit_bias(), ignore_eos=True,
                stop_token_ids=[raw.tokenizer.eos_token_id], detokenize=False,
                skip_special_tokens=False, spaces_between_special_tokens=False)
            outputs = raw._generate([raw._prompt(prefix)], params)
            if len(outputs) != 1:
                raise RuntimeError("one native completion required")
            value = raw._completion(outputs[0])
            tokens = tuple(int(t) for t in value.token_ids)
            probabilities, _ = _generated_logprobs(value.logprobs, tokens, None)
            cached = min(len(prefix), int(getattr(outputs[0], "num_cached_tokens", 0) or 0))
            return {"request_label":label,"token_ids": tokens, "token_logprobs": probabilities,
                    "cached_prefill_tokens": cached, "fresh_prefill_tokens":len(prefix) - cached}

        reset()
        result["warmup"] = sample(parent, 9, "conditional-warm-parent")
        contexts = []
        for branch in original["branches"]:
            reference = tuple(raw.encode(branch["reference"], add_special_tokens=False))
            for step in range(len(history)):
                contexts.append((branch["branch"], step, parent + reference + history[:step]))
        cached = [sample(p, b, f"conditional-cached-{b}-{i}") for b, i, p in contexts]
        fresh = []
        for b, i, p in contexts:
            reset()
            fresh.append(sample(p, b, f"conditional-fresh-{b}-{i}"))
        checks = []
        for (branch, step, prefix), a, b in zip(contexts, cached, fresh, strict=True):
            equal = a["token_ids"] == b["token_ids"]
            difference = max((abs(x - y) for x, y in zip(a["token_logprobs"], b["token_logprobs"], strict=True)), default=0) if equal else None
            checks.append({"branch": branch, "step": step,
                           "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                           "cached": a, "fresh": b, "tokens_equal": equal,
                           "max_abs_logprob_difference": difference,
                           "passed": equal and difference <= original["absolute_logprob_tolerance"]
                           and a["cached_prefill_tokens"] > 0 and b["cached_prefill_tokens"] == 0})
        result["checks"] = checks
        result["conditional_checks_passed"] = all(c["passed"] for c in checks)
    except BaseException:
        result["exception"] = traceback.format_exc()
        raise
    finally:
        try:
            close_backend(raw)
        finally:
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditional_checks_passed": result["conditional_checks_passed"],
                      "checks": len(checks), "original_validation_passed": original["passed"]}))
    if not result["conditional_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Compare cached sibling continuations against fresh prefill on the target model.

Run after formal generation. This is a separate correctness diagnostic and never
enters the quality or latency tables. Protocol: first sorted recorded context of
at least 1600 tokens, two fixed references, eight sampled tokens per branch, seed
20261109, exact sampled-token agreement and absolute log-probability tolerance
0.05 nats for BF16 execution. No tuning or retries with different examples/seeds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import traceback

from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",type=Path,required=True)
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing validation; select a new output path")
    rows = [r for line in args.records.read_text(encoding="utf-8").splitlines()
            if (r:=json.loads(line)).get("kind") == "measurement" and len(r["context_ids"]) >= 1600]
    row = min(rows,key=lambda r:(r["question_id"],r["arm"]))
    parent = tuple(row["context_ids"][:1600])
    settings = load_settings(args.settings)
    if not settings["ar"]["engine"]["vllm"]["enable_prefix_caching"]:
        raise ValueError("requires enabled automatic prefix caching")
    raw = None
    result = {"kind":"separate_cache_correctness_diagnostic","source_question":row["question_id"],
        "source_arm":row["arm"],"source_trajectory_sha256":row["trajectory_sha256"],
        "parent_ids":parent,"parent_sha256":hashlib.sha256(json.dumps(parent).encode()).hexdigest(),
        "seed":20261109,"sample_tokens":8,"absolute_logprob_tolerance":.05,"passed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        raw = load_backend(settings["ar"]["model"],settings["ar"]["engine"],seed=20261109,logprobs=1)
        result["model_id"] = raw.model_id
        policy = SamplingConfig(temperature=1,top_p=1,eos_token_id=raw.tokenizer.eos_token_id)

        def reset():
            async def perform():
                if await raw._engine.reset_prefix_cache() is False:
                    raise RuntimeError("prefix cache reset failed")
            raw._runner.run(perform())

        def sample(prefix,index,label,count=8):
            before = asdict(raw.snapshot())
            s = raw.sample_batch([GenerationRequest(prefix,count,policy,20261109+index,"cache-check:"+label)])[0]
            after = asdict(raw.snapshot())
            return {"token_ids":s.token_ids,"token_logprobs":s.token_logprobs,
                "finish_reason":s.finish_reason,"prefix_ids":prefix,
                "cached_prefill_tokens":after["shared_prefill_tokens_saved"]-before["shared_prefill_tokens_saved"],
                "fresh_prefill_tokens":after["prefill_tokens"]-before["prefill_tokens"]}

        references = ["\n[Validation reference A: the example setting alpha has value 17.]\n",
                      "\n[Validation reference B: the example setting beta has value 29.]\n"]
        prefixes = [parent+tuple(raw.encode(text,add_special_tokens=False)) for text in references]
        reset()
        result["warmup"] = sample(parent,9,"warm-parent",1)
        cached = [sample(p,i,f"cached-{i}") for i,p in enumerate(prefixes)]
        fresh = []
        for i,p in enumerate(prefixes):
            reset()
            fresh.append(sample(p,i,f"fresh-{i}"))
        pairs = []
        for i,(a,b) in enumerate(zip(cached,fresh,strict=True)):
            same = a["token_ids"] == b["token_ids"]
            difference = max((abs(x-y) for x,y in zip(a["token_logprobs"],b["token_logprobs"],strict=True)),default=0) if same else None
            pairs.append({"branch":i,"reference":references[i],"cached":a,"fresh":b,
                "tokens_equal":same,"max_abs_logprob_difference":difference,
                "passed":same and difference <= .05 and a["cached_prefill_tokens"] > 0 and b["cached_prefill_tokens"] == 0})
        result["branches"] = pairs
        result["passed"] = all(p["passed"] for p in pairs)
    except BaseException:
        result["exception"] = traceback.format_exc()
        raise
    finally:
        try:
            close_backend(raw)
        finally:
            args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    comparable = [p["max_abs_logprob_difference"] for p in pairs if p["tokens_equal"]]
    print(json.dumps({"passed":result["passed"],"branches_tokens_equal":[p["tokens_equal"] for p in pairs],
                      "max_logprob_difference_on_equal_paths":max(comparable,default=None)}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

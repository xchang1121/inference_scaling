"""No-effort-prompt, matched-content follow-up to the online-injection pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.retrieval import Document, Hit, OnlineInjectionController, Retriever
from inference_scaling.retrieval.ablation import SYSTEM, FrozenRetriever, ReferenceVariant, arms, reference_tokens, static_prompt
from inference_scaling.retrieval.corpus import build_query
from inference_scaling.shared.model.prompting import render_prompt
from inference_scaling.shared.rng import SeedStream


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def neutral_config(ar):
    # In this pinned tokenizer, the medium template branch emits no effort
    # instruction. Omitting the argument would silently insert xhigh text.
    return {**ar,"prompt":{**ar["prompt"],"chat_template_kwargs":{
        "enable_thinking":True,"reasoning_effort":"medium"}},
        "output":{**ar["output"],"thinking_mode":"enabled"}}


def prepare(args, settings):
    from transformers import AutoTokenizer
    options = settings["ar"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(options["path"],local_files_only=True,trust_remote_code=False)

    class Encoder:
        encode = staticmethod(lambda text,add_special_tokens=False:tuple(tokenizer.encode(text,add_special_tokens=add_special_tokens)))
        decode = staticmethod(lambda ids,skip_special_tokens=False:tokenizer.decode(list(ids),skip_special_tokens=skip_special_tokens))
    raw = Encoder()
    noise = tuple(raw.encode(args.noise.read_text(encoding="utf-8")))
    if len(noise) < 1100:
        raise ValueError("unrelated prose must supply at least 1100 tokens without repetition")
    questions = read_jsonl(args.data/"questions.jsonl")
    if len(questions) != 8 or any(set(q)-{"id","dataset","category","question"} for q in questions):
        raise ValueError("the frozen follow-up requires the original eight public TechQA questions")
    index = Retriever.from_jsonl(args.data/"corpus.jsonl",kind="bm25")
    configs = neutral_config(settings["ar"])
    cases = []
    for question in questions:
        hits = index.search(build_query(question["question"],"",kind="question"),k=2)
        rendered = render_prompt(tokenizer,[{"role":"system","content":SYSTEM},
            {"role":"user","content":question["question"]}],configs)
        if "Reasoning effort is set to" in rendered or not rendered.endswith("<think>\n"):
            raise ValueError("expected an open thinking prompt without any effort instruction")
        base = raw.encode(rendered)
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        separator = raw.encode("\n\n")
        variants = {}
        for content,wrapper in (("relevant","boxed"),("unrelated","boxed"),("blank","boxed"),
                                ("relevant","neutral"),("unrelated","neutral")):
            proxy = ReferenceVariant(raw,hits,noise,content=content,wrapper=wrapper)
            payload = reference_tokens(proxy,hits)
            if not payload or any(t in payload for t in (end_id,tokenizer.convert_tokens_to_ids("</think>"))):
                raise ValueError("invalid reference payload")
            variants[content+"."+wrapper] = {"ids":payload,"text":raw.decode(payload),
                "sha256":sha(json.dumps(payload).encode()),
                "static_ids":static_prompt(base,payload,end_id,separator)}
        if len({len(v["ids"]) for v in variants.values()}) != 1:
            raise ValueError("all reference variants must have exactly equal token lengths")
        if len({len(v["static_ids"]) for v in variants.values()}) != 1:
            raise ValueError("all static reference variants must have exactly equal prompt lengths")
        cases.append({"id":question["id"],"question":question["question"],"dataset":question["dataset"],
            "prompt_ids":base,"prompt_text":rendered,"prompt_sha256":sha(json.dumps(base).encode()),
            "hits":[{"document":asdict(h.document),"score":h.score} for h in hits],"variants":variants})
    prepared = {"schema":1,"cases":cases,"noise_ids":noise,"corpus_sha256":index.fingerprint,
        "input_hashes":{p.name:sha(p.read_bytes()) for p in (args.data/"questions.jsonl",args.data/"corpus.jsonl",args.noise)},
        "tokenizer_sha256":sha((Path(options["path"])/"tokenizer.json").read_bytes()),
        "tokenizer_config_sha256":sha((Path(options["path"])/"tokenizer_config.json").read_bytes()),
        "prompt_policy":"thinking enabled; medium template branch emits no effort instruction; no resolve_thinking call",
        "content_policy":"same question-only BM25 top-2; unrelated body tokens length-matched; source labels retained",
        "wrapper_policy":"neutral replaces outer header/footer with equal token counts of newlines; document labels and bodies unchanged"}
    args.prepared.write_text(json.dumps(prepared,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"prepared_cases":len(cases),"variants":5,"noise_tokens":len(noise),
                      "no_effort_prompts":True,"all_payload_lengths_matched":True}),flush=True)


def sampled(row):
    return [t for s in row["segments"] if s["kind"] == "generated" for t in s["token_ids"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",type=Path,required=True)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--data",type=Path)
    parser.add_argument("--noise",type=Path)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--budgets",nargs="+",type=int,default=[256,1024,4096])
    parser.add_argument("--seed",type=int,default=20261109)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.prepare_only:
        if args.prepared.exists():parser.error("preserve existing prepared input")
        prepare(args,settings)
        return
    if args.output is None:parser.error("--output required for generation")
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    package = Path(__file__).resolve().parents[1]/"src/inference_scaling"
    identity_data = {"settings_sha256":sha(args.settings.read_bytes()),"prepared_sha256":sha(args.prepared.read_bytes()),
        "budgets":args.budgets,"seed":args.seed,"runner_sha256":sha(Path(__file__).read_bytes()),
        "source":{p.relative_to(package).as_posix():sha(p.read_bytes()) for p in package.rglob("*.py")}}
    identity = sha(json.dumps(identity_data,sort_keys=True).encode())
    expected = len(prepared["cases"])*sum(len(arms(b)) for b in args.budgets)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old = read_jsonl(args.output) if args.output.exists() else []
        if old and old[0].get("identity") != identity:raise ValueError("resume identity changed")
        done = {(r["budget"],r["question_id"],r["arm"]):r for r in old if r.get("kind") == "measurement"}
        if len(done) != sum(r.get("kind") == "measurement" for r in old):raise ValueError("duplicate output")
        raw = graph = None
        with args.output.open("a",encoding="utf-8") as stream:
            def write(row):
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush();os.fsync(stream.fileno())
            if not old:write({"kind":"metadata","identity":identity,"identity_data":identity_data,
                "expected":expected,"budgets":args.budgets,"arms":list(arms(args.budgets[0])),
                "protocol":"independent full trajectories, identical request chunking; no effort instruction; exact length-matched controls"})
            try:
                raw = load_backend(settings["ar"]["model"],settings["ar"]["engine"],seed=args.seed,logprobs=1)
                from benchmark_is_gpu import GraphMeasurements,engine_settings
                graph = GraphMeasurements(raw)
                formats = thinking_format_from_backend(raw,output_settings_from_config(settings["ar"])).formats
                if len(formats) != 1:raise ValueError("one explicit thinking format required")
                fmt = formats[0]
                write({"kind":"engine","model_id":raw.model_id,"parameter_count":raw.parameter_count,
                       "config":engine_settings(raw)})
                from inference_scaling.arllm.config import SamplingConfig
                from inference_scaling.arllm.types import GenerationRequest
                raw.sample_batch([GenerationRequest(tuple(prepared["cases"][0]["prompt_ids"]),8,
                    SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id),1,"ablation-warmup")])
                for bi,budget in enumerate(args.budgets):
                    for qi,case in enumerate(prepared["cases"]):
                        methods = list(arms(budget));alternatives = methods[1:]
                        shift = (qi+bi)%len(alternatives)
                        methods = ["closed",*alternatives[shift:],*alternatives[:shift]]
                        hits = [Hit(Document(**h["document"]),h["score"]) for h in case["hits"]]
                        for arm in methods:
                            key = budget,case["id"],arm
                            if key in done:continue
                            async def reset():
                                if await raw._engine.reset_prefix_cache() is False:raise RuntimeError("prefix cache reset failed")
                            raw._runner.run(reset())
                            content,wrapper,cfg = arms(budget)[arm]
                            payload = case["variants"][content+"."+wrapper]
                            backend = ReferenceVariant(raw,hits,prepared["noise_ids"],content=content,wrapper=wrapper)
                            def prompt(question,evidence,static=tuple(payload["static_ids"]),base=tuple(case["prompt_ids"])):
                                return static if evidence else base
                            write({"kind":"attempt_start","budget":budget,"question_id":case["id"],"arm":arm})
                            print(json.dumps({"start":key,"done":len(done),"expected":expected}),flush=True)
                            before = graph.snapshot()
                            try:
                                row = OnlineInjectionController(backend,FrozenRetriever(hits,prepared["corpus_sha256"]),cfg,
                                    thinking_format=fmt,prompt=prompt).run(case["question"],
                                    seed=SeedStream(args.seed).derive("question",case["id"]),request_id=f"{budget}:{case['id']}:{arm}")
                                row.update(graph.difference(before))
                                injections = [e for e in row["events"] if e.get("injected")]
                                actual = [t for s in row["segments"] if s["kind"] in {"evidence","prompt_evidence"} for t in s["token_ids"]]
                                if injections and actual != payload["ids"]:raise RuntimeError("payload differs from frozen prepared IDs")
                                if row["answer_generated_tokens"] > 512:raise RuntimeError("answer budget exceeded")
                                for s in row["segments"]:
                                    if s.get("token_statistics"):
                                        import math
                                        s["token_statistics"] = [v if math.isfinite(v) else None for v in s["token_statistics"]]
                                coupling = None
                                if arm != "closed" and cfg.position != "user":
                                    base = done[budget,case["id"],"closed"]
                                    offset = injections[0]["actual_generated_offset"] if injections else min(len(sampled(base)),len(sampled(row)))
                                    coupling = {"checked_tokens":offset,"matches_baseline_before_injection":sampled(base)[:offset] == sampled(row)[:offset],
                                        "same_initial_prompt":base["prompt_ids"] == row["prompt_ids"],
                                        "same_complete_context_if_no_injection":None if injections else base["context_ids"] == row["context_ids"]}
                                first_after = []
                                for i,s in enumerate(row["segments"]):
                                    if s["kind"] == "evidence":
                                        first_after = next((t["token_ids"] for t in row["segments"][i+1:] if t["kind"] == "generated"),[])
                                        break
                                value = {"kind":"measurement","budget":budget,"question_id":case["id"],"dataset":case["dataset"],
                                    "arm":arm,"content_condition":content,"wrapper_condition":wrapper,
                                    "prepared_payload_sha256":payload["sha256"],"prepared_payload_tokens":len(payload["ids"]),
                                    "first_post_injection_ids":first_after,"immediate_thinking_end":
                                        bool(first_after) and tuple(first_after[:len(fmt.end_token_ids)]) == fmt.end_token_ids,
                                    "prefix_coupling":coupling,**row}
                                write(value);done[key] = value
                                print(json.dumps({"complete":key,"done":len(done),"seconds":row["seconds"]}),flush=True)
                            except BaseException as error:
                                write({"kind":"failed","budget":budget,"question_id":case["id"],"arm":arm,"error":repr(error)})
                                raise
                write({"kind":"complete","count":len(done),"expected":expected})
            finally:
                if graph is not None:graph.close()
                close_backend(raw)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

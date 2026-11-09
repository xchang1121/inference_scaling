"""Matched-cap single / SC-BoN / complete-pool block-Met IS retrieval pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import time

from benchmark_flare_gap import clean, prepare, render, sha
from benchmark_is_gpu import GraphMeasurements, engine_settings
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Retriever
from inference_scaling.retrieval.scaled import ProbeController, ScalingConfig, scaled_condition
from inference_scaling.retrieval.stepwise import StepConfig, StepState, WindowEvidence
from inference_scaling.shared.rng import SeedStream

METHODS = ("single", "sc_bon", "block_met_sc")
CONTEXTS = ("closed", "static", "dynamic")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("settings", "prepared", "corpus", "questions", "noise", "output"):
        p.add_argument("--"+n, type=Path, required=n in {"settings", "prepared", "corpus"})
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=CONTEXTS)
    a = p.parse_args()
    settings, cfg = load_settings(a.settings), ScalingConfig()
    if a.prepare_only:
        if a.prepared.exists():
            raise ValueError("preserve the previous prepared input")
        prepare(a, settings)
        prepared = json.loads(a.prepared.read_text(encoding="utf-8"))
        prepared["scaling_config"] = asdict(cfg)
        prepared["config"] = asdict(replace(StepConfig(), work_tokens=cfg.work_tokens,
            answer_tokens=cfg.answer_tokens, context_tokens=cfg.context_tokens))
        a.prepared.write_text(json.dumps(prepared, ensure_ascii=False)+"\n", encoding="utf-8")
        return
    if a.output is None or a.output.exists():
        raise ValueError("a fresh output path is required")
    prepared = json.loads(a.prepared.read_text(encoding="utf-8"))
    cfg = ScalingConfig(**prepared["scaling_config"])
    step_cfg = StepConfig(**prepared["config"])
    import fcntl
    with a.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = graph = None
        with a.output.open("w", encoding="utf-8") as stream:
            def write(row):
                stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush(); os.fsync(stream.fileno())
            root = Path(__file__).resolve().parents[1]
            source = {x.relative_to(root).as_posix(): sha(x.read_bytes())
                for x in [*(root/"src").rglob("*.py"), Path(__file__)]}
            expected = len(prepared["cases"])*len(METHODS)*sum(1 if c == "closed" else 2 for c in a.contexts)
            write({"kind": "metadata", "config": asdict(cfg), "step_config": asdict(step_cfg),
                "seed": a.seed, "contexts": a.contexts, "methods": METHODS,
                "expected_conditions": expected, "source_hashes": source,
                "prepared_sha256": sha(a.prepared.read_bytes()), "settings_sha256": sha(a.settings.read_bytes()),
                "protocol": "one frozen context per suffix search; common probe charged to every standalone view; no native KV fork; complete candidate weights only"})
            try:
                raw = load_backend(settings["ar"]["model"], settings["ar"]["engine"], seed=a.seed, logprobs=1)
                graph = GraphMeasurements(raw)
                fmt, = thinking_format_from_backend(raw, output_settings_from_config(settings["ar"])).formats
                index = Retriever.from_jsonl(a.corpus, kind="bm25")
                if index.fingerprint != prepared["corpus_sha256"]:
                    raise ValueError("corpus changed")
                evidence = WindowEvidence(raw, index, prepared["noise_ids"], step_cfg)
                ctrl = ProbeController(raw, evidence, fmt,
                    lambda system, user: render(raw.tokenizer, settings["ar"], system, user, False), cfg=step_cfg)
                ctrl.scan_tokens = cfg.probe_tokens
                write({"kind": "engine", "model_id": raw.model_id, "parameter_count": raw.parameter_count,
                    "config": engine_settings(raw)})

                def reset():
                    async def clear():
                        if await raw._engine.reset_prefix_cache() is False:
                            raise RuntimeError("cache reset failed")
                    raw._runner.run(clear())

                def unit(uid, action):
                    reset()
                    before, graphs, started = _snapshot(raw), graph.snapshot(), time.perf_counter()
                    try:
                        state, detail = action()
                    except BaseException as error:
                        write({"kind": "failed_unit", "unit_id": uid, "error": repr(error),
                            "seconds": time.perf_counter()-started, "backend_snapshot": _snapshot(raw),
                            **graph.difference(graphs)})
                        raise
                    after = _snapshot(raw)
                    cost = {k: after[k]-before[k] for k in before.keys() & after.keys()
                        if type(before[k]) in (int, float) and type(after[k]) in (int, float)}
                    result = ctrl.result(state)
                    write({"kind": "execution_unit", "unit_id": uid, "calls": state.calls,
                        "detail": detail, "result": result, "seconds": time.perf_counter()-started,
                        "cost": cost, **graph.difference(graphs)})
                    return state, detail

                # A separately recorded warmup keeps compilation out of case 1.
                from inference_scaling.arllm.config import SamplingConfig
                from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
                warm_sampling = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                before, started, graphs = _snapshot(raw), time.perf_counter(), graph.snapshot()
                warm, = raw.sample_batch([GenerationRequest(tuple(prepared["cases"][0]["prompt_ids"]), 16,
                    warm_sampling, 1, "warmup", stop_sequences=(fmt.end_token_ids,),
                    statistic=TokenStatistic(warm_sampling), intermediate_events=False)])
                write({"kind": "warmup", "token_ids": warm.token_ids, "seconds": time.perf_counter()-started,
                    "before": before, "after": _snapshot(raw), **graph.difference(graphs)})
                for qi, case in enumerate(prepared["cases"]):
                    seed = SeedStream(a.seed).derive("question", case["id"])
                    contexts = list(a.contexts[qi % len(a.contexts):])+list(a.contexts[:qi % len(a.contexts)])
                    for context in contexts:
                        print(json.dumps({"question": case["id"], "context": context}), flush=True)
                        common = StepState(tuple(case["prompt_ids"]), seed, case["question"])
                        packet, common_units = None, []
                        if context == "dynamic":
                            uid = case["id"]+":probe"
                            common, packet = unit(uid, partial(ctrl.probe, case["question"], case["prompt_ids"], seed))
                            common_units = [uid]
                        elif context == "static":
                            packet = case["static_packet"]
                            common.retrievals = 1
                        cached = {}
                        methods = list(METHODS[qi % len(METHODS):])+list(METHODS[:qi % len(METHODS)])
                        for method in methods:
                            contents = ("none",) if context == "closed" else ("relevant", "unrelated")
                            for content in contents:
                                key = ("terminal" if common.phase != "thinking" or common.finish
                                    else method if not packet else method+":"+content)
                                if key not in cached:
                                    if key == "terminal":
                                        cached[key] = (common, {}, [])
                                    else:
                                        uid = case["id"]+":"+context+":"+method+":"+content
                                        state, audit = unit(uid, partial(scaled_condition, ctrl, common, packet, content, method, cfg))
                                        cached[key] = (state, audit, [uid])
                                state, audit, units = cached[key]
                                result = ctrl.result(state)
                                if result["work_tokens"] > cfg.work_tokens or result["answer_tokens"] > cfg.answer_tokens:
                                    raise RuntimeError("condition cap exceeded")
                                write({"kind": "measurement", "question_id": case["id"], "cohort": case["cohort"],
                                    "context": context, "method": method, "content": content,
                                    "arm": context+"."+content+"/"+method,
                                    "execution_units": common_units+units, "shared_no_intervention": not bool(packet),
                                    "search_bypassed_natural_end": key == "terminal", "result": result,
                                    "search": audit})
                write({"kind": "completed", "conditions": expected})
            finally:
                if graph is not None:
                    graph.close()
                if raw is not None:
                    close_backend(raw)


if __name__ == "__main__":
    main()

"""Paired legacy, anchored-SC and answer-verified IS on isolated TP1 workers."""
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
from inference_scaling.retrieval.grounded import AnchoredController, AnchoredEvidence, GroundedConfig, grounded_condition
from inference_scaling.retrieval.scaled import ProbeController, ScalingConfig, scaled_condition
from inference_scaling.retrieval.stepwise import StepConfig, StepState, WindowEvidence
from inference_scaling.shared.rng import SeedStream

ARMS = {
    "static_legacy_sc": ("static", "legacy_sc", "relevant"),
    "dynamic_legacy_sc": ("dynamic", "legacy_sc", "relevant"),
    "static_anchor_sc": ("static", "anchor_sc", "relevant"),
    "dynamic_anchor_sc": ("dynamic", "anchor_sc", "relevant"),
    "static_verified_is": ("static", "verified_is", "relevant"),
    "dynamic_verified_is": ("dynamic", "verified_is", "relevant"),
    "static_verified_noise": ("static", "verified_is", "unrelated"),
    "dynamic_verified_noise": ("dynamic", "verified_is", "unrelated"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("settings", "prepared", "corpus", "questions", "noise", "output"):
        p.add_argument("--"+name, type=Path, required=name in {"settings", "prepared", "corpus"})
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[20261110, 20261111])
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    p.add_argument("--worker", default="unassigned")
    a = p.parse_args()
    settings, cfg, legacy_cfg = load_settings(a.settings), GroundedConfig(), ScalingConfig()
    if a.prepare_only:
        if a.prepared.exists():
            raise ValueError("preserve existing prepared data")
        prepare(a, settings)
        data = json.loads(a.prepared.read_text(encoding="utf-8"))
        data["grounded_config"], data["scaling_config"] = asdict(cfg), asdict(legacy_cfg)
        data["config"] = asdict(replace(StepConfig(), work_tokens=cfg.work_tokens))
        a.prepared.write_text(json.dumps(data, ensure_ascii=False)+"\n", encoding="utf-8")
        return
    if a.output is None or a.output.exists():
        raise ValueError("a fresh output path is required")
    prepared = json.loads(a.prepared.read_text(encoding="utf-8"))
    cfg = GroundedConfig(**prepared["grounded_config"])
    legacy_cfg = ScalingConfig(**prepared["scaling_config"])
    step_cfg = StepConfig(**prepared["config"])
    cases = [x for x in prepared["cases"] if not a.case_ids or x["id"] in a.case_ids]
    if a.case_ids and set(a.case_ids) != {x["id"] for x in cases}:
        raise ValueError("unknown case IDs")
    if not cases or len(set(a.arms)) != len(a.arms) or len(set(a.seeds)) != len(a.seeds):
        raise ValueError("empty cases or duplicate arms/seeds")
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
            expected = len(cases)*len(a.seeds)*len(a.arms)
            write({"kind": "metadata", "grounded_config": asdict(cfg), "scaling_config": asdict(legacy_cfg),
                "step_config": asdict(step_cfg), "seeds": a.seeds, "worker": a.worker, "arms": a.arms,
                "cases": [x["id"] for x in cases], "expected_conditions": expected,
                "source_hashes": source, "prepared_sha256": sha(a.prepared.read_bytes()),
                "settings_sha256": sha(a.settings.read_bytes()),
                "protocol": "immutable evidence per IS epoch; completed candidate pools; gold never supplied; answer-verification cost in work cap"})
            try:
                raw = load_backend(settings["ar"]["model"], settings["ar"]["engine"], seed=a.seeds[0], logprobs=1)
                graph = GraphMeasurements(raw)
                fmt, = thinking_format_from_backend(raw, output_settings_from_config(settings["ar"])).formats
                index = Retriever.from_jsonl(a.corpus, kind="bm25")
                if index.fingerprint != prepared["corpus_sha256"]:
                    raise ValueError("corpus changed")
                aux_prompt = lambda system, user: render(raw.tokenizer, settings["ar"], system, user, False)
                legacy = ProbeController(raw, WindowEvidence(raw, index, prepared["noise_ids"], step_cfg), fmt, aux_prompt, cfg=step_cfg)
                ctrl = AnchoredController(raw, AnchoredEvidence(raw, index, prepared["noise_ids"], step_cfg), fmt, aux_prompt, cfg=step_cfg, grounded_cfg=cfg)
                write({"kind": "engine", "model_id": raw.model_id, "parameter_count": raw.parameter_count, "config": engine_settings(raw)})

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
                            "seconds": time.perf_counter()-started, "backend_snapshot": _snapshot(raw), **graph.difference(graphs)})
                        raise
                    after = _snapshot(raw)
                    cost = {k: after[k]-before[k] for k in before.keys() & after.keys()
                        if type(before[k]) in (int, float) and type(after[k]) in (int, float)}
                    write({"kind": "execution_unit", "unit_id": uid, "calls": state.calls,
                        "detail": detail, "result": ctrl.result(state), "seconds": time.perf_counter()-started,
                        "cost": cost, **graph.difference(graphs)})
                    return state, detail

                from inference_scaling.arllm.config import SamplingConfig
                from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
                sampling = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                before, graphs, started = _snapshot(raw), graph.snapshot(), time.perf_counter()
                warm, = raw.sample_batch([GenerationRequest(tuple(cases[0]["prompt_ids"]), 16, sampling, 1,
                    "warmup", stop_sequences=(fmt.end_token_ids,), statistic=TokenStatistic(sampling), intermediate_events=False)])
                write({"kind": "warmup", "token_ids": warm.token_ids, "seconds": time.perf_counter()-started,
                    "before": before, "after": _snapshot(raw), **graph.difference(graphs)})
                measured = 0
                for qi, case in enumerate(cases):
                    for run_seed in a.seeds:
                        seed = SeedStream(run_seed).derive("question", case["id"])
                        stem = case["id"]+":"+str(run_seed)
                        common_cache, result_cache = {}, {}
                        arms = a.arms[qi % len(a.arms):]+a.arms[:qi % len(a.arms)]
                        for arm in arms:
                            context, method, content = ARMS[arm]
                            print(json.dumps({"question": case["id"], "seed": run_seed, "arm": arm, "done": measured}), flush=True)
                            is_legacy = method == "legacy_sc"
                            common_key = context+("_legacy" if is_legacy else "_anchor")
                            if common_key not in common_cache:
                                controller = legacy if is_legacy else ctrl
                                if context == "static" and is_legacy:
                                    common = StepState(tuple(case["prompt_ids"]), seed, case["question"])
                                    common.retrievals = 1
                                    packet, units = case["static_packet"], []
                                else:
                                    uid = stem+":"+common_key+":probe"
                                    action = controller.static if context == "static" else controller.probe
                                    common, packet = unit(uid, partial(action, case["question"], case["prompt_ids"], seed))
                                    units = [uid]
                                common_cache[common_key] = (common, packet, units, controller)
                            common, packet, common_units, controller = common_cache[common_key]
                            terminal = common.phase != "thinking" or bool(common.finish)
                            key = (common_key, "terminal" if terminal else method, content if packet else "none")
                            if key not in result_cache:
                                if terminal:
                                    result_cache[key] = (common, {"bypassed_natural_end": True}, [])
                                else:
                                    uid = stem+":"+arm
                                    action = (partial(scaled_condition, controller, common, packet, content, "block_met_sc", legacy_cfg)
                                              if is_legacy else partial(grounded_condition, controller, common, packet, content, method == "verified_is", cfg))
                                    state, audit = unit(uid, action)
                                    result_cache[key] = (state, audit, [uid])
                            state, audit, units = result_cache[key]
                            result = ctrl.result(state)
                            if result["work_tokens"] > cfg.work_tokens or result["answer_tokens"] > cfg.answer_tokens:
                                raise RuntimeError("condition budget exceeded")
                            measured += 1
                            write({"kind": "measurement", "question_id": case["id"], "cohort": case["cohort"],
                                "seed": run_seed, "worker": a.worker, "arm": arm, "context": context, "method": method,
                                "content": content, "execution_units": common_units+units,
                                "shared_no_intervention": not bool(packet), "search_bypassed_natural_end": terminal,
                                "result": result, "search": audit})
                write({"kind": "completed", "conditions": measured})
            finally:
                if graph is not None:
                    graph.close()
                if raw is not None:
                    close_backend(raw)


if __name__ == "__main__":
    main()

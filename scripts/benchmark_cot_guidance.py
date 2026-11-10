"""Paired static/inline short evidence with BoN and conditional IS selectors."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from benchmark_flare_gap import clean, prepare, render
from benchmark_is_gpu import GraphMeasurements, engine_settings
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.retrieval.ablation import static_prompt
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Retriever
from inference_scaling.retrieval.cot_guidance import (
    DualStatistics, GuidanceConfig, GuidanceLedger, completed_boundary, finish_answer, search,
)
from inference_scaling.retrieval.grounded import AnchoredController, AnchoredEvidence, GroundedConfig
from inference_scaling.retrieval.stepwise import StepConfig, StepState, token_hash
from inference_scaling.shared.rng import SeedStream

METHODS = ("bon", "is_sir_sc", "is_met_sc", "is_block_sc", "is_block_consilience")
CONTEXTS = ("static_relevant", "inline_relevant", "static_unrelated", "inline_unrelated")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("settings", "prepared", "corpus", "questions", "noise", "output"):
        p.add_argument("--"+name, type=Path, required=name in {"settings", "prepared", "corpus"})
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=[20261112, 20261113])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=CONTEXTS)
    p.add_argument("--worker", default="unassigned")
    a = p.parse_args()
    settings, cfg = load_settings(a.settings), GuidanceConfig()
    if a.prepare_only:
        if a.prepared.exists():
            raise ValueError("prepared data already exists")
        prepare(a, settings)
        return
    if a.output is None or a.output.exists():
        raise ValueError("fresh output required")
    prepared = json.loads(a.prepared.read_text(encoding="utf-8"))
    cases = [c for c in prepared["cases"] if not a.case_ids or c["id"] in a.case_ids]
    if not cases or a.case_ids and {c["id"] for c in cases} != set(a.case_ids):
        raise ValueError("invalid cases")
    import fcntl
    with a.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with a.output.open("w", encoding="utf-8") as stream:
            def write(row):
                stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
            source = Path(__file__).resolve().parents[1]
            hashes = {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [*source.joinpath("src").rglob("*.py"), Path(__file__)]}
            write({"kind": "metadata", "config": asdict(cfg), "worker": a.worker, "source": hashes,
                "cases": [c["id"] for c in cases], "seeds": a.seeds, "methods": a.methods, "contexts": a.contexts,
                "prepared_sha256": hashlib.sha256(a.prepared.read_bytes()).hexdigest(),
                "settings_sha256": hashlib.sha256(a.settings.read_bytes()).hexdigest()})
            raw = graph = tap = None
            try:
                # Top-5 plus the existing tagged full-vocabulary SC transport slot.
                raw = load_backend(settings["ar"]["model"], settings["ar"]["engine"], seed=a.seeds[0], logprobs=6)
                graph, tap = GraphMeasurements(raw), DualStatistics(raw)
                fmt, = thinking_format_from_backend(raw, output_settings_from_config(settings["ar"])).formats
                close = tuple(fmt.end_token_ids)
                index = Retriever.from_jsonl(a.corpus, kind="bm25")
                assert index.fingerprint == prepared["corpus_sha256"]
                step_cfg = StepConfig(work_tokens=cfg.search_tokens, context_tokens=cfg.context_tokens,
                    answer_tokens=cfg.answer_tokens, evidence_tokens=cfg.evidence_tokens)
                ev = AnchoredEvidence(raw, index, prepared["noise_ids"], step_cfg)
                ctrl = AnchoredController(raw, ev, fmt,
                    lambda s,u: render(raw.tokenizer, settings["ar"], s,u,False), cfg=step_cfg,
                    grounded_cfg=GroundedConfig(context_tokens=cfg.context_tokens, extract_tokens=512))
                write({"kind": "engine", "model_id": raw.model_id, "config": engine_settings(raw)})
                completed = 0
                for qi, case in enumerate(cases):
                    evidence_seed = SeedStream(20261112).derive("evidence", case["id"])
                    state = StepState(tuple(case["prompt_ids"]), evidence_seed, case["question"])
                    started = time.perf_counter()
                    packet = ctrl.extract(state)
                    write({"kind": "evidence", "question_id": case["id"], "packet": packet,
                        "events": state.events, "calls": state.calls, "work": state.work,
                        "seconds": time.perf_counter()-started})
                    for run_seed in a.seeds:
                        seed = SeedStream(run_seed).derive("question", case["id"])
                        base = tuple(case["prompt_ids"])
                        probe = GuidanceLedger(raw, tap, base, close, SeedStream(seed).derive("probe"), cfg, cap=cfg.probe_tokens)
                        draft, boundary, terminal = [], None, False
                        while len(draft) < cfg.probe_tokens:
                            sample, = probe.sample_batch([probe.request(min(128,cfg.probe_tokens-len(draft)), len(probe.calls), draft)])
                            draft.extend(sample.token_ids)
                            boundary = completed_boundary(raw, draft, cfg.probe_minimum)
                            stop = next((i for i,t in enumerate(draft) if t in {raw.tokenizer.eos_token_id, close[0]}), len(draft))
                            if boundary is not None and boundary <= stop:
                                break
                            if stop < len(draft):
                                terminal = True
                                break
                        if boundary is not None:
                            draft = draft[:boundary]
                        elif not terminal:
                            # No complete step: skip the intervention; do not invent a boundary at token 512.
                            draft = []
                        carried = probe.statistics(draft)
                        write({"kind": "probe", "question_id": case["id"], "seed": run_seed,
                            "calls": probe.calls, "work": probe.work, "reason_ids": draft,
                            "boundary": boundary, "terminal": terminal, "statistics": carried})
                        contexts = list(a.contexts[qi%len(a.contexts):])+list(a.contexts[:qi%len(a.contexts)])
                        reuse = {}
                        for context in contexts:
                            position, content = context.split("_",1)
                            inline = position == "inline"
                            injected = bool(packet) and (not inline or boundary is not None)
                            payload = tuple(ev.variant(packet, content)) if injected else ()
                            root = (base+tuple(draft)+payload) if inline else (static_prompt(base,payload,ctrl.end_id,()) if payload else base)
                            common = carried if inline else None
                            for method in a.methods:
                                identity = (tuple(root), method, inline, terminal if inline else False)
                                if identity in reuse:
                                    old = reuse[identity]
                                    write({**old, "context": context, "reused_from": old["unit_id"]})
                                    completed += 1
                                    continue
                                uid = f"{case['id']}:{run_seed}:{context}:{method}"
                                print(json.dumps({"unit": uid, "completed": completed}), flush=True)
                                before, g0, started = _snapshot(raw), graph.snapshot(), time.perf_counter()
                                ledger = GuidanceLedger(raw,tap,root,close,SeedStream(seed).derive("search"),cfg,
                                    carried=common,cap=cfg.search_tokens-(probe.work if inline else 0))
                                if inline and terminal and boundary is None:
                                    candidates = [()]
                                    audit = {"selection": {m:0 for m in (["single","bon_sc","bon_consilience","bon_logprob"] if method=="bon" else [method])}, "natural_end_before_injection": True}
                                else:
                                    candidates, audit = search(ledger,method)
                                answers = []
                                for i, candidate in enumerate(candidates):
                                    answer_seed = SeedStream(seed).derive("answer_suffix",token_hash(candidate))
                                    if inline and terminal and boundary is None:
                                        result = finish_answer(raw,base,draft,close,cfg,answer_seed)
                                    else:
                                        result = finish_answer(raw,root,candidate,close,cfg,answer_seed)
                                    answers.append(result)
                                after = _snapshot(raw)
                                row = {"kind":"measurement", "unit_id":uid, "question_id":case["id"],
                                    "seed":run_seed, "worker":a.worker, "context":context, "method":method,
                                    "injected":injected, "payload_ids":payload, "root_ids":root,
                                    "root_sha256":token_hash(root), "carried_reason_ids":draft if inline else [],
                                    "calls":ledger.calls, "search_work":ledger.work, "probe_work":probe.work if inline else 0,
                                    "evidence_work":state.work, "candidates":candidates, "answers":answers, "audit":audit,
                                    "seconds":time.perf_counter()-started,
                                    "cost":{k:after[k]-before[k] for k in before.keys()&after.keys() if type(before[k]) in (int,float) and type(after[k]) in (int,float)},
                                    **graph.difference(g0)}
                                write(row)
                                if any(x["truncated"] for x in answers):
                                    raise RuntimeError("answer reached 16384 tokens: stop study; do not score truncated answers")
                                reuse[identity] = row
                                completed += 1
                write({"kind":"completed", "units":completed})
            except BaseException as error:
                write({"kind":"failed", "error":repr(error)})
                raise
            finally:
                if tap is not None: tap.close()
                if graph is not None: graph.close()
                if raw is not None: close_backend(raw)


if __name__ == "__main__":
    main()

"""Step-boundary FLARE/gap ablation; exact paired stems, cold continuations."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time

from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.retrieval.ablation import SYSTEM
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Document, Retriever
from inference_scaling.retrieval.stepwise import StepConfig, StepController, StepState, WindowEvidence, token_hash
from inference_scaling.shared.model.prompting import render_prompt
from inference_scaling.shared.rng import SeedStream


POLICIES = ("closed", "static", "legacy_sc", "step_sc", "flare_gap")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_config(ar, thinking):
    return {**ar, "prompt": {**ar["prompt"], "chat_template_kwargs": {
        "enable_thinking": thinking, "reasoning_effort": "medium"}},
        "output": {**ar["output"], "thinking_mode": "enabled" if thinking else "disabled"}}


def render(tokenizer, ar, system, user, thinking):
    text = render_prompt(tokenizer, [{"role": "system", "content": system},
        {"role": "user", "content": user}], prompt_config(ar, thinking))
    if "Reasoning effort is set to" in text:
        raise ValueError("effort instruction unexpectedly rendered")
    if thinking and not text.endswith("<think>\n"):
        raise ValueError("main prompt must leave thinking open")
    if not thinking and not text.endswith("</think>\n\n"):
        raise ValueError("checker must start outside thinking")
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def prepare(args, settings):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings["ar"]["model"]["path"], local_files_only=True)
    class Codec:
        encode = staticmethod(lambda text, add_special_tokens=False: tuple(tokenizer.encode(text, add_special_tokens=add_special_tokens)))
        decode = staticmethod(lambda ids, skip_special_tokens=False: tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens))
    codec = Codec()
    cfg = StepConfig()
    index = Retriever.from_jsonl(args.corpus, kind="bm25")
    noise = codec.encode(args.noise.read_text(encoding="utf-8"))
    if len(noise) < 1024:
        raise ValueError("unrelated prose pool too short")
    evidence = WindowEvidence(codec, index, noise, cfg)
    cases = []
    for q in jsonl(args.questions):
        if set(q)-{"id", "dataset", "category", "question", "cohort"}:
            raise ValueError("runtime question file contains annotations")
        base = render(tokenizer, settings["ar"], SYSTEM, q["question"], True)
        packet = evidence.fetch(q["question"])
        if not packet["ids"] or len(packet["ids"]) > cfg.evidence_tokens:
            raise ValueError("static reference does not fit frozen packet cap")
        real, noise_variant = evidence.variant(packet, "relevant"), evidence.variant(packet, "unrelated")
        if len(real) != len(noise_variant) or real == noise_variant:
            raise ValueError("invalid same-length noise intervention")
        forbidden = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("</think>"), tokenizer.convert_tokens_to_ids("<think>")}
        if forbidden.intersection(real) or forbidden.intersection(noise_variant):
            raise ValueError("structural marker in reference")
        cases.append({**q, "prompt_ids": base, "prompt_sha256": token_hash(base), "static_packet": packet})
    # Template contract is checked without starting the model.
    aux = render(tokenizer, settings["ar"], "Return exactly JSON.", '{"question":"test"}', False)
    result = {"schema": 1, "config": asdict(cfg), "cases": cases, "noise_ids": noise,
        "corpus_sha256": index.fingerprint, "auxiliary_example_ids": aux,
        "hashes": {str(p.name): sha(p.read_bytes()) for p in (args.questions, args.corpus, args.noise)},
        "tokenizer_hashes": {n: sha((Path(settings["ar"]["model"]["path"])/n).read_bytes())
                            for n in ("tokenizer.json", "tokenizer_config.json")}}
    args.prepared.write_text(json.dumps(result, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"prepared": len(cases), "config": asdict(cfg), "no_effort_instruction": True}), flush=True)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("settings", "prepared", "questions", "corpus", "noise", "output"):
        parser.add_argument("--"+name, type=Path, required=name in {"settings", "prepared", "corpus"})
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--checker-contract-probe", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.prepare_only:
        if args.prepared.exists(): parser.error("preserve previous prepared file")
        prepare(args, settings)
        return
    if args.output is None: parser.error("--output required")
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    cfg = StepConfig(**prepared["config"])
    if len(set(args.policies)) != len(args.policies): parser.error("duplicate policy")
    conditions_per_question = sum(1 if p == "closed" else 4 if p == "flare_gap" else 2 for p in args.policies)
    root = Path(__file__).resolve().parents[1]
    identity_data = {"settings_sha256": sha(args.settings.read_bytes()), "prepared_sha256": sha(args.prepared.read_bytes()),
        "seed": args.seed, "policies": args.policies, "checker_contract_probe": args.checker_contract_probe,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "source": {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in (root/"src").rglob("*.py")}}
    identity = sha(json.dumps(identity_data, sort_keys=True).encode())
    import fcntl
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        # An interrupted condition is preserved. Explicitly launch a new output,
        # rather than silently rerunning it and hiding its cost in a resume.
        if args.output.exists(): raise ValueError("preserve existing run; explicit retry output required")
        raw = graph = None
        with args.output.open("w", encoding="utf-8") as stream:
            def write(row):
                stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush(); os.fsync(stream.fileno())
            write({"kind": "metadata", "identity": identity, "identity_data": identity_data,
                "config": asdict(cfg), "policies": args.policies, "expected_conditions": conditions_per_question*len(prepared["cases"]),
                "protocol": "exact shared preintervention units; each condition charged common+branch; cold branch prefill; no native KV fork",
                "noise_policy": "substitute body only after real-evidence support gate; no independent noise gate evaluation"})
            try:
                raw = load_backend(settings["ar"]["model"], settings["ar"]["engine"], seed=args.seed, logprobs=1)
                from benchmark_is_gpu import GraphMeasurements, engine_settings
                graph = GraphMeasurements(raw)
                fmt, = thinking_format_from_backend(raw, output_settings_from_config(settings["ar"])).formats
                index = Retriever.from_jsonl(args.corpus, kind="bm25")
                if index.fingerprint != prepared["corpus_sha256"]: raise ValueError("corpus changed")
                evidence = WindowEvidence(raw, index, prepared["noise_ids"], cfg)
                aux_prompt = lambda system, user: render(raw.tokenizer, settings["ar"], system, user, False)
                ctrl = StepController(raw, evidence, fmt, aux_prompt, cfg=cfg)
                write({"kind": "engine", "model_id": raw.model_id, "parameter_count": raw.parameter_count,
                    "config": engine_settings(raw)})
                from inference_scaling.arllm.config import SamplingConfig
                from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
                warm_prefix = render(raw.tokenizer, settings["ar"], SYSTEM, "State a short technical fact.", True)
                warm_sampling = SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                warm_before, warm_graph, warm_started = _snapshot(raw), graph.snapshot(), time.perf_counter()
                warm = raw.sample_batch([GenerationRequest(warm_prefix, 16, warm_sampling, 1, "warmup",
                    stop_sequences=(fmt.end_token_ids,), statistic=TokenStatistic(warm_sampling), intermediate_events=False)])[0]
                warm_after = _snapshot(raw)
                write({"kind": "warmup", "seconds": time.perf_counter()-warm_started,
                    "prefix_ids": warm_prefix, "token_ids": warm.token_ids,
                    "cost": {k: warm_after[k]-warm_before[k] for k in warm_before.keys() & warm_after.keys()
                             if type(warm_before[k]) in (int, float) and type(warm_after[k]) in (int, float)},
                    **graph.difference(warm_graph)})
                def reset():
                    async def clear():
                        if await raw._engine.reset_prefix_cache() is False: raise RuntimeError("prefix cache reset failed")
                    raw._runner.run(clear())
                if args.checker_contract_probe:
                    # A transparent synthetic positive contract, not a benchmark
                    # sample or sampled prehistory. It checks the supported path
                    # even when the two TRAIN retrievals are correctly rejected.
                    fixture_question = "What exact retry timeout should Mercury API v2 use?"
                    fixture_reason = "The required value depends on the version-specific retry policy. "*10
                    fixture_draft = "The Mercury API v2 retry timeout is 31 seconds."
                    fixture_source = "Mercury API v2 uses a retry timeout of exactly 27 seconds. Apply this timeout before retrying a failed request."
                    fixture_index = Retriever([Document("mercury-v2", "Mercury API v2 retry policy", fixture_source)], kind="bm25")
                    fixture_evidence = WindowEvidence(raw, fixture_index, prepared["noise_ids"], cfg)
                    fixture_ctrl = StepController(raw, fixture_evidence, fmt, aux_prompt, cfg=cfg)
                    state = StepState(render(raw.tokenizer, settings["ar"], SYSTEM, fixture_question, True), 77, fixture_question,
                        reason=list(raw.encode(fixture_reason, add_special_tokens=False)),
                        pending=list(raw.encode(fixture_draft, add_special_tokens=False)))
                    state.pending_lp = [-3.]*len(state.pending)
                    reset()
                    probe_before, probe_graph, probe_started = _snapshot(raw), graph.snapshot(), time.perf_counter()
                    packet = fixture_ctrl._gap(state, len(state.pending))
                    branches = []
                    if packet:
                        state.pending.clear(); state.pending_lp.clear()
                        short = replace(cfg, work_tokens=state.work+64, answer_tokens=32)
                        fixture_ctrl = StepController(raw, fixture_evidence, fmt, aux_prompt, cfg=short)
                        for position in ("flare_gap", "flare_gap_rebuild"):
                            reset()
                            branch = fixture_ctrl.branch(state, packet, "relevant", position)
                            branches.append({"policy": position, "calls": branch.calls, "result": fixture_ctrl.result(branch)})
                    probe_after = _snapshot(raw)
                    write({"kind": "checker_contract", "accepted": bool(packet), "seconds": time.perf_counter()-probe_started,
                        "fixture": {"question": fixture_question, "provided_reason": fixture_reason, "provided_draft": fixture_draft,
                                    "source": fixture_source, "note": "provided prefix is input, not sampled or charged as generation"},
                        "calls": state.calls, "events": state.events, "branches": branches,
                        "cost": {k: probe_after[k]-probe_before[k] for k in probe_before.keys() & probe_after.keys()
                                 if type(probe_before[k]) in (int, float) and type(probe_after[k]) in (int, float)},
                        **graph.difference(probe_graph)})
                    if not packet: raise RuntimeError("supported synthetic checker contract was not accepted; preserve failed smoke")
                def unit(uid, action):
                    reset()
                    before, graphs, started = _snapshot(raw), graph.snapshot(), time.perf_counter()
                    try:
                        value = action()
                    except BaseException as error:
                        after = _snapshot(raw)
                        write({"kind": "execution_failed", "unit_id": uid, "error": repr(error),
                            "seconds": time.perf_counter()-started,
                            "cost": {k: after[k]-before[k] for k in before.keys() & after.keys()
                                     if type(before[k]) in (int, float) and type(after[k]) in (int, float)},
                            **graph.difference(graphs)})
                        raise
                    after = _snapshot(raw)
                    cost = {k: after[k]-before[k] for k in before.keys() & after.keys()
                            if type(before[k]) in (int, float) and type(after[k]) in (int, float)}
                    state = value[0] if isinstance(value, tuple) else value
                    write({"kind": "execution_unit", "unit_id": uid, "seconds": time.perf_counter()-started,
                        "cost": cost, **graph.difference(graphs), "calls": state.calls,
                        "result": ctrl.result(state), "packet": value[1] if isinstance(value, tuple) else None})
                    return value
                for qi, case in enumerate(prepared["cases"]):
                    alternatives = [p for p in args.policies if p != "closed"]
                    shift = qi % len(alternatives) if alternatives else 0
                    policies = (["closed"] if "closed" in args.policies else [])+alternatives[shift:]+alternatives[:shift]
                    seed = SeedStream(args.seed).derive("question", case["id"])
                    for policy in policies:
                        uid = case["id"]+":"+policy+":common"
                        write({"kind": "attempt_start", "question_id": case["id"], "policy": policy})
                        print(json.dumps({"question": case["id"], "cohort": case.get("cohort", "smoke"), "policy": policy}), flush=True)
                        common, packet = unit(uid, partial(ctrl.stem, case["question"], case["prompt_ids"], seed, policy))
                        conditions = [(policy, "none")] if policy == "closed" else [(policy, c) for c in ("relevant", "unrelated")]
                        if policy == "flare_gap":
                            conditions += [("flare_gap_rebuild", c) for c in ("relevant", "unrelated")]
                        for condition_policy, content in conditions:
                            units = [uid]
                            result_state = common
                            if packet:
                                bid = case["id"]+":"+condition_policy+":"+content
                                result_state = unit(bid, partial(ctrl.branch, common, packet, content, condition_policy))
                                units.append(bid)
                            result = ctrl.result(result_state)
                            if result["work_tokens"] > cfg.work_tokens or result["answer_tokens"] > cfg.answer_tokens:
                                raise RuntimeError("generation budget exceeded")
                            write({"kind": "measurement", "question_id": case["id"], "cohort": case.get("cohort", "smoke"),
                                "policy": condition_policy, "stem_policy": policy, "content": content, "arm": condition_policy+"."+content,
                                "execution_units": units, "shared_no_intervention": not bool(packet) and policy != "closed",
                                "base_prompt_sha256": case["prompt_sha256"], "result": result})
                write({"kind": "completed", "conditions": conditions_per_question*len(prepared["cases"])})
            finally:
                if graph is not None: graph.close()
                if raw is not None: close_backend(raw)


if __name__ == "__main__":
    main()

"""Resumable reference-free reasoning comparisons on public math problems."""

from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.shared.artifacts import file_sha256, json_fingerprint
from experiments.shared.math_benchmark import load_math500, stratified_subset, MathJudge
from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides
from experiments.shared.statistics import wilson_interval
from experiments.arllm.reasoning_methods import (
    REWARDS, budget_plan, generation_cost, check_budget, compare_sir, compare_mh, majority_index, add_costs,
)
from inference_scaling.arllm.backends.loader import load_backend_from_config, close_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.generation import generation_config_for_prompt
from inference_scaling.shared.prompting import render_prompt
from inference_scaling.shared.rng import SeedStream


def model_prompt(backend, question: str, config: dict) -> tuple[int, ...]:
    text = render_prompt(backend.tokenizer, [{"role": "user", "content": question +
        "\nPlease reason step by step, and put your final answer within \\boxed{}."}], config)
    return backend.encode(text, add_special_tokens=False)


def visible_output(backend, prompt, tokens, config) -> dict:
    info = SamplingScope.from_config(backend, config, active=False).describe_output(backend, prompt, tokens)
    if config.get("output", {}).get("thinking_mode") == "enabled" and info["thinking_status"] != "complete":
        # A truncated thought is not a final answer in this evaluation protocol.
        info["content_text"] = ""
    return info


def measured_call(backend, callback):
    import torch
    cuda = str(getattr(backend, "device", "cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    before = asdict(backend.snapshot())
    start = time.perf_counter()
    value = callback()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    after = asdict(backend.snapshot())
    cost = {key: after[key] - before[key] for key in before}
    cost["seconds"] = elapsed
    return value, cost


def run_base(backend, judge, problem, config, seed, mode):
    current = deepcopy(config)
    current.setdefault("output", {})["thinking_mode"] = mode
    prompt = model_prompt(backend, problem.question, current)
    current, budget = generation_config_for_prompt(current, len(prompt), [backend])
    sample, cost = measured_call(backend, lambda: backend.sample_batch([GenerationRequest(
        prefix=prompt, max_new_tokens=budget["effective_max_new_tokens"],
        sampling=SamplingConfig(temperature=float(current["sampling"]["temperature"]),
                                eos_token_id=backend.tokenizer.eos_token_id),
        seed=seed, request_id=f"base:{problem.identifier}:{mode}:{seed}",
    )])[0])
    segments = visible_output(backend, prompt, sample.token_ids, current)
    grade = judge.grade(segments["content_text"], problem.answer)
    return {"token_ids": sample.token_ids, "token_logprobs": sample.token_logprobs,
            "prompt_tokens": len(prompt), "thinking_mode": mode,
            "thinking_tokens": len(segments["thinking_token_ids"]),
            "thinking_status": segments["thinking_status"], "content": segments["content_text"],
            "thinking": segments["thinking_text"], "cost": cost, "generation_budget": budget,
            "ended_by_eos": segments["ended_by_eos"], **grade}


def summarize(output: Path) -> dict:
    records = [json.loads(line) for line in (output / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["reward"], row["budget_forward_tokens"])].append(row)
    summary = []
    for (method, reward, budget), rows in sorted(grouped.items()):
        trials, correct = len(rows), sum(row["correct"] for row in rows)
        summary.append({"method": method, "reward": reward, "budget_forward_tokens": budget,
            "trials": trials, "correct": correct, "accuracy": correct / trials,
            "wilson_95": wilson_interval(correct, trials),
            "mean_used_forward_tokens": sum(row["used_forward_tokens"] for row in rows) / trials,
            "mean_generated_tokens": sum(row["cost"].get("generated_tokens", 0) for row in rows) / trials,
            "mean_pfLOPs": sum(row["cost"]["estimated_dense_forward_flops"] for row in rows) / trials / 1e15,
            "mean_selected_tokens": sum(row["selected_tokens"] for row in rows) / trials,
            "unparseable": sum(not row["parseable"] for row in rows),
            "incomplete_thinking": sum(row["thinking_status"] not in {"complete", "disabled"} for row in rows),
            "changed_mh_updates": sum(row.get("changed_updates", 0) for row in rows),
            "accepted_changed_mh_updates": sum(row.get("accepted_changed_updates", 0) for row in rows),
        })
    value = {"rows": summary, "records": len(records)}
    (output / "summary.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(f"{row['budget_forward_tokens']:>7} {row['method']:>14} {row['reward']:>24} "
              f"{row['correct']}/{row['trials']} ({row['accuracy']:.1%}) "
              f"used={row['mean_used_forward_tokens']:.0f} PFLOPs={row['mean_pfLOPs']:.4f}", flush=True)
    return value


def run_comparisons(backend, judge, problems, config, args, fingerprint):
    path = args.output / "comparisons.jsonl"
    previous = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {(row["problem_id"], row["draw"], row["method"], row["reward"], row["budget_forward_tokens"]) for row in previous}
    pool_directory = args.output / "pools"
    pool_directory.mkdir(exist_ok=True)
    current = deepcopy(config)
    current.setdefault("output", {})["thinking_mode"] = "enabled"
    current["output"]["sampling_scope"] = "full"
    with path.open("a", encoding="utf-8", buffering=1) as sink:
        for problem in problems:
            for draw in range(args.draws):
                key = json_fingerprint([problem.identifier, draw])[:20]
                pool_path = pool_directory / (key + ".json")
                pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else {"fingerprint": fingerprint, "samples": {}}
                if pool["fingerprint"] != fingerprint:
                    raise ValueError("candidate pool belongs to another experiment")

                def sample(kind, index, mode="enabled", *, pool=pool, problem=problem, draw=draw, pool_path=pool_path):
                    sample_key = f"{kind}:{index}:{mode}"
                    if sample_key not in pool["samples"]:
                        seed = SeedStream(args.seed).derive(problem.identifier, draw, kind, index, mode)
                        pool["samples"][sample_key] = run_base(backend, judge, problem, current, seed, mode)
                        pool_path.write_text(json.dumps(pool, ensure_ascii=False) + "\n", encoding="utf-8")
                        info = pool["samples"][sample_key]
                        print(f"pool {problem.identifier} draw={draw} {sample_key} tokens={len(info['token_ids'])} "
                              f"correct={info['correct']} seconds={info['cost']['seconds']:.1f}", flush=True)
                    return pool["samples"][sample_key]

                prompt = model_prompt(backend, problem.question, current)
                bounded, generation = generation_config_for_prompt(current, len(prompt), [backend])
                score_cache = {}
                for budget, candidates in zip(args.budgets, args.candidate_counts, strict=True):
                    plan = budget_plan(budget, candidates, len(prompt), generation["effective_max_new_tokens"])
                    for method in args.methods:
                        sources = ["none"] if method in {"base", "vote"} else args.rewards
                        for source in sources:
                            modes = ["disabled", "enabled"] if method == "base" else ["enabled"]
                            for mode in modes:
                                label = "base_" + mode if method == "base" else method
                                identity = (problem.identifier, draw, label, source, budget)
                                if identity in done:
                                    continue
                                if method == "base":
                                    item = sample("candidate", 0, mode)
                                    mode_config = deepcopy(current)
                                    mode_config["output"]["thinking_mode"] = mode
                                    mode_prompt = model_prompt(backend, problem.question, mode_config)
                                    maximum = min(plan["max_new_tokens"], budget - len(mode_prompt))
                                    tokens = tuple(item["token_ids"][:maximum])
                                    output = visible_output(backend, mode_prompt, tokens, mode_config)
                                    cost = generation_cost(len(mode_prompt), len(tokens), backend.parameter_count)
                                    result = {"method": label, "reward": source, "cost": cost,
                                        "content": output["content_text"], "thinking_status": output["thinking_status"],
                                        "selected_tokens": len(tokens), "budget_forward_tokens": budget,
                                        "used_forward_tokens": check_budget(cost, budget),
                                        **judge.grade(output["content_text"], problem.answer)}
                                elif method == "vote":
                                    sequences = [tuple(sample("candidate", i)["token_ids"][:plan["max_new_tokens"]]) for i in range(candidates)]
                                    outputs = [visible_output(backend, prompt, tokens, bounded) for tokens in sequences]
                                    selected = majority_index([item["content_text"] for item in outputs], judge)
                                    cost = add_costs(*(generation_cost(len(prompt), len(tokens), backend.parameter_count) for tokens in sequences))
                                    result = {"method": "vote", "reward": "none", "cost": cost,
                                        "content": outputs[selected]["content_text"], "thinking_status": outputs[selected]["thinking_status"],
                                        "selected_tokens": len(sequences[selected]), "budget_forward_tokens": budget,
                                        "used_forward_tokens": check_budget(cost, budget),
                                        **judge.grade(outputs[selected]["content_text"], problem.answer)}
                                else:
                                    pilots = [sample("pilot", i) for i in range(2)] if source == "self_consistency" else []
                                    common = dict(backend=backend, judge=judge, prompt=prompt, reference=problem.answer,
                                        config=bounded, plan=plan, pilots=pilots, source=source,
                                        seed=SeedStream(args.seed).derive(problem.identifier, draw, method),
                                        render_output=visible_output)
                                    if method == "is":
                                        result = compare_sir(**common, samples=[sample("candidate", i) for i in range(candidates)],
                                                             score_cache=score_cache)
                                    else:
                                        result = compare_mh(**common)
                                result.update(problem_id=problem.identifier, subject=problem.subject, level=problem.level,
                                              draw=draw, parameter_count=backend.parameter_count, manifest_fingerprint=fingerprint)
                                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                                done.add(identity)
                                print(f"result {problem.identifier} draw={draw} {label}/{source} B={budget} "
                                      f"correct={result['correct']} used={result['used_forward_tokens']}", flush=True)
    summarize(args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3_math.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/math500/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/qwen3_math"))
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--stage", choices=("base", "compare", "summarize"), default="base")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true", help="validate data, references and manifest without loading model weights")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--modes", nargs="+", choices=("enabled", "disabled"), default=["disabled", "enabled"])
    parser.add_argument("--methods", nargs="+", choices=("base", "vote", "is", "mh"), default=["base", "vote", "is", "mh"])
    parser.add_argument("--rewards", nargs="+", choices=REWARDS, default=list(REWARDS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[32768, 131072])
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[2, 4])
    add_model_output_arguments(parser)
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize(args.output)
        return
    if len(args.budgets) != len(args.candidate_counts) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("each distinct budget requires exactly one candidate count")
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    apply_model_output_overrides(config, args)
    if args.stage == "compare" and config.get("output", {}).get("sampling_scope", "full") != "full":
        raise ValueError("this reward comparison uses full-sequence IS/MH; select --sampling-scope full")
    selection = stratified_subset(load_math500(args.data, download=bool(args.allow_download)), **{
        ("seed" if key == "subset_seed" else key): value for key, value in config["benchmark"].items()})
    problems = selection[args.split]
    if args.limit is not None:
        problems = problems[:args.limit]
    manifest = {"config": config, "seed": args.seed, "data_sha256": file_sha256(args.data),
                "split": args.split, "subset": {key: [p.identifier for p in values] for key, values in selection.items()}}
    if args.stage == "compare":
        from experiments.arllm.runtime import validate_model_artifacts
        artifacts = validate_model_artifacts(config, ["base"])
        manifest.update(protocol="reasoning-comparison-v2", weight_sha256=artifacts["weight_sha256"],
                        metadata_sha256=artifacts["metadata_sha256"],
                        budgets=args.budgets, candidate_counts=args.candidate_counts)
    fingerprint = json_fingerprint(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json_fingerprint(json.loads(manifest_path.read_text(encoding="utf-8"))) != fingerprint:
        raise ValueError("existing experiment uses a different configuration; select a new output directory")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records_path = args.output / "base.jsonl"
    previous = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()] if records_path.exists() else []
    done = {(row["problem_id"], row["thinking_mode"], row["draw"]) for row in previous}
    backend = judge = None
    try:
        judge = MathJudge()
        for problem in problems:
            validation = judge.grade("\\boxed{" + problem.answer + "}", problem.answer)
            if not validation["correct"]:
                raise ValueError(f"benchmark reference is unsupported by the evaluator: {problem.identifier}")
        if args.validate_only:
            print(f"validated {len(problems)} {args.split} problems; model weights were not loaded", flush=True)
            return
        backend = load_backend_from_config(config["models"]["base"], config, role="base")
        # Warm-up is excluded from per-problem inference cost.
        warm = model_prompt(backend, "Compute 1 + 1.", config)
        backend.sample_batch([GenerationRequest(warm, 2, SamplingConfig(), args.seed, "warmup")])
        if args.stage == "compare":
            run_comparisons(backend, judge, problems, config, args, fingerprint)
            return
        with records_path.open("a", encoding="utf-8", buffering=1) as sink:
            for problem in problems:
                for mode in args.modes:
                    for draw in range(args.draws):
                        if (problem.identifier, mode, draw) in done:
                            continue
                        seed = SeedStream(args.seed).derive(problem.identifier, mode, draw)
                        result = run_base(backend, judge, problem, config, seed, mode)
                        result.update(problem_id=problem.identifier, subject=problem.subject,
                                      level=problem.level, draw=draw, seed=seed,
                                      parameter_count=backend.parameter_count, manifest_fingerprint=fingerprint)
                        sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                        print(f"{problem.identifier} mode={mode} draw={draw} correct={result['correct']} "
                              f"tokens={len(result['token_ids'])} thinking={result['thinking_status']} "
                              f"seconds={result['cost']['seconds']:.1f}", flush=True)
    finally:
        if judge is not None:
            judge.close()
        if backend is not None:
            close_backend(backend)


if __name__ == "__main__":
    main()

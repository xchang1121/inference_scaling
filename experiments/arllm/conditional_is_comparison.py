"""Conditional IS vs its archived block variant vs whole-sequence SIR on math problems.

No method is capped by a budget. Each record keeps the backend counters of what
it spent: forward-token slots charge every request for its whole prefix, while
generated tokens count only new tokens, which is closer to a run that reuses
prefix caches. Compare methods on accuracy against both views.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.arllm.assembly.reasoning_methods import compare_conditional, reward_temperature, sir_curve
from experiments.arllm.reasoning_benchmark import model_prompt, run_base, visible_output
from experiments.shared.artifacts import file_sha256, json_fingerprint, load_jsonl, write_json_atomic
from experiments.shared.math_benchmark import MathJudge, load_math500, stratified_subset
from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides
from inference_scaling.arllm.backends.loader import close_backend, load_backend_from_config
from inference_scaling.arllm.rewards.factory import MODEL_REWARD_SOURCES, model_reward_from_config
from inference_scaling.shared.model.generation import generation_config_for_prompt
from inference_scaling.shared.rng import SeedStream

METHODS = ("sir", "conditional_is", "block_conditional_is")


def record_key(row: dict) -> tuple[str, str, str, str]:
    return row["problem_id"], row["method"], row["reward"], row["setting"]


def pool_rewards(backend, config, prompt, samples, source):
    """Reward of each shared sample and the scoring cost it took."""
    reward = model_reward_from_config(backend, config, source=source)
    values, costs = [], []
    for sample in samples:
        if source == "sequence_log_probability":
            # Generation already returned the exact actual-policy log probabilities.
            values.append(reward.from_token_logprobs(prompt, tuple(sample["token_ids"]), sample["token_logprobs"]))
            costs.append({})
            continue
        before = asdict(backend.snapshot())
        values.append(reward(prompt, tuple(sample["token_ids"])))
        after = asdict(backend.snapshot())
        costs.append({key: after[key] - before[key] for key in before})
    return values, costs


def summarize(output: Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in load_jsonl(output / "records.jsonl"):
        grouped[(row["method"], row["reward"], row["setting"])].append(row)
    summary = []
    for (method, reward, setting), rows in sorted(grouped.items()):
        costs = [row["cost"] for row in rows]
        item = {"method": method, "reward": reward, "setting": setting, "problems": len(rows),
                "accuracy": fmean(float(row["correct"]) for row in rows),
                "mean_forward_token_slots": fmean(cost.get("generation_forward_token_slots", 0)
                                                  + cost.get("score_forward_token_slots", 0) for cost in costs),
                "mean_generated_tokens": fmean(cost.get("generated_tokens", 0) for cost in costs),
                "mean_scored_tokens": fmean(cost.get("scored_tokens", 0) for cost in costs),
                "mean_pflops": fmean(cost.get("estimated_dense_forward_flops", 0) for cost in costs) / 1e15,
                "incomplete_thinking": sum(row["thinking_status"] != "complete" for row in rows)}
        if method == "sir":
            item["expected_accuracy"] = fmean(row["expected_correct"] for row in rows)
        summary.append(item)
    write_json_atomic(output / "summary.json", summary, indent=2)
    for item in summary:
        print(f"{item['method']:>12} {item['reward']:>24} {item['setting']:>28} "
              f"{item['accuracy']:.1%} ({item['problems']}) slots={item['mean_forward_token_slots']:.0f} "
              f"generated={item['mean_generated_tokens']:.0f} PFLOPs={item['mean_pflops']:.4f}", flush=True)
    return summary


def run(args) -> None:
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    apply_model_output_overrides(config, args)
    config.setdefault("output", {}).update(thinking_mode="enabled", sampling_scope="full")
    for source in args.rewards:
        reward_temperature(source, config)
    selection = stratified_subset(load_math500(args.data, download=bool(args.allow_download)), **{
        ("seed" if key == "subset_seed" else key): value for key, value in config["benchmark"].items()})
    problems = selection[args.split][:args.limit]
    from experiments.arllm.assembly.runtime import validate_model_artifacts
    artifacts = validate_model_artifacts(config, ["base"])
    manifest = {"protocol": "conditional-is-comparison-v1", "config": config, "seed": args.seed,
                "data_sha256": file_sha256(args.data), "split": args.split,
                "weight_sha256": artifacts["weight_sha256"], "metadata_sha256": artifacts["metadata_sha256"]}
    fingerprint = json_fingerprint(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json_fingerprint(json.loads(manifest_path.read_text(encoding="utf-8"))) != fingerprint:
        raise ValueError("existing results use a different configuration; select a new output directory")
    write_json_atomic(manifest_path, manifest, indent=2)
    records_path = args.output / "records.jsonl"
    done = {record_key(row) for row in load_jsonl(records_path)}
    (args.output / "pools").mkdir(exist_ok=True)
    judge = backend = None
    try:
        judge = MathJudge()
        backend = load_backend_from_config(config["models"]["base"], config, role="base")
        with records_path.open("a", encoding="utf-8", buffering=1) as sink:
            for problem in problems:
                compare_problem(backend, judge, problem, config, args, fingerprint, done, sink)
    finally:
        if judge is not None:
            judge.close()
        if backend is not None:
            close_backend(backend)
    summarize(args.output)


def compare_problem(backend, judge, problem, config, args, fingerprint, done, sink) -> None:
    prompt = model_prompt(backend, problem.question, config)
    bounded, generation = generation_config_for_prompt(config, len(prompt), [backend])
    total_length = generation["effective_max_new_tokens"]
    pool_path = args.output / "pools" / (json_fingerprint(problem.identifier)[:20] + ".json")
    pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else []
    identity = {"problem_id": problem.identifier, "subject": problem.subject, "level": problem.level,
                "parameter_count": backend.parameter_count, "manifest_fingerprint": fingerprint}
    for source in args.rewards:
        temperature = reward_temperature(source, config)
        seed = SeedStream(args.seed).derive(problem.identifier, source)
        rows = []
        if "sir" in args.methods and any((problem.identifier, "sir", source, f"N={n}") not in done
                                         for n in args.sir_counts):
            while len(pool) < max(args.sir_counts):
                pool.append(run_base(backend, judge, problem, bounded,
                                     SeedStream(args.seed).derive(problem.identifier, "pool", len(pool)), "enabled"))
                write_json_atomic(pool_path, pool)
            rewards, reward_costs = pool_rewards(backend, bounded, prompt, pool, source)
            rows += sir_curve(samples=pool, rewards=rewards, reward_costs=reward_costs, source=source,
                              temperature=temperature, seed=seed, counts=args.sir_counts)
        setting = f"M={args.candidates},K={args.rollouts},B={min(args.block_size, total_length)}"
        for method in ("conditional_is", "block_conditional_is"):
            if method in args.methods and (problem.identifier, method, source, setting) not in done:
                rows.append(compare_conditional(backend=backend, judge=judge, prompt=prompt, reference=problem.answer,
                    config=bounded, source=source, temperature=temperature, seed=seed, render_output=visible_output,
                    candidates=args.candidates, rollouts=args.rollouts, block_size=args.block_size,
                    total_length=total_length, block=method == "block_conditional_is"))
        for row in rows:
            row.update(identity)
            if record_key(row) in done:
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add(record_key(row))
            print(f"result {problem.identifier} {row['method']}/{source} {row['setting']} "
                  f"correct={row['correct']} generated={row['cost'].get('generated_tokens', 0)}", flush=True)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3_math.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/math500/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/qwen3_conditional_is"))
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--stage", choices=("compare", "summarize"), default="compare")
    parser.add_argument("--limit", type=_positive_integer, help="use the first N problems in the fixed split order")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rewards", nargs="+", choices=MODEL_REWARD_SOURCES, default=["consilience"])
    parser.add_argument("--sir-counts", nargs="+", type=_positive_integer, default=[1, 2, 4, 8, 16],
                        help="SIR pool sizes; every size reuses the first samples of one shared pool")
    parser.add_argument("--candidates", type=_positive_integer, default=4, help="candidate blocks per step (M)")
    parser.add_argument("--rollouts", type=_positive_integer, default=1, help="completions per candidate (K)")
    parser.add_argument("--block-size", type=_positive_integer, default=4096, help="block length in tokens (B)")
    add_model_output_arguments(parser, exclude={"--proposal-model", "--mh-iterations", "--thinking-mode"})
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "summarize":
        summarize(args.output)
        return
    run(args)


if __name__ == "__main__":
    main()

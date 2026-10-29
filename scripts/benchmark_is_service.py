"""Measure IS service latency and throughput with fixed public-dataset requests.

TTFT counts nonempty reasoning/content deltas, not role events or keep-alives.
The wall-clock interval includes HTTP queuing, but excludes grading and warmup.
No accelerator is loaded by this client.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np

from inference_scaling.app.settings import load_settings
from inference_scaling.datasets.gsm8k import GSM8K
from inference_scaling.datasets.math500 import MATH500


def read_stream(lines, started, deadline, clock=time.perf_counter):
    thinking, content, chunks = [], [], 0
    first_thinking = first_content = None
    finish, usage, scaling, done = None, None, {}, False
    for line in lines:
        elapsed = clock() - started
        if elapsed > deadline:
            raise TimeoutError("client deadline exceeded; server inference may still be running")
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(str(event["error"]))
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if text := delta.get("reasoning_content"):
                if first_thinking is None:
                    first_thinking = elapsed
                thinking.append(text)
                chunks += 1
            if text := delta.get("content"):
                if first_content is None:
                    first_content = elapsed
                content.append(text)
            finish = choice.get("finish_reason") or finish
        usage = event.get("usage") or usage
        scaling = event.get("scaling") or scaling
    if not done or finish is None or usage is None:
        raise RuntimeError("incomplete SSE response or missing final usage")
    thought = "".join(thinking)
    return {
        "thinking_ttft_seconds": first_thinking, "content_ttft_seconds": first_content,
        "thinking_chunks": chunks, "finish_reason": finish, "usage": usage, "scaling": scaling,
        "thinking_sha256": hashlib.sha256(thought.encode()).hexdigest(), "content": "".join(content),
    }


def request_one(url, payload, deadline):
    started = time.perf_counter()
    try:
        request = Request(url.rstrip("/") + "/v1/chat/completions",
                          data=json.dumps({**payload, "stream": True}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=deadline) as response:
            row = read_stream(response, started, deadline)
        return {"status": "complete", "seconds": time.perf_counter() - started, **row}
    except Exception as error:
        return {"status": "failed", "seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}"}


def summary(rows, elapsed, cards):
    complete = [row for row in rows if row["status"] == "complete"]
    all_complete = len(complete) == len(rows)
    def quantiles(key):
        values = [row[key] for row in complete if row.get(key) is not None]
        return ({"count": len(values), "p50": float(np.percentile(values, 50)),
                 "p95": float(np.percentile(values, 95))} if values else {"count": 0, "p50": None, "p95": None})
    tokens = sum(row["usage"]["completion_tokens"] for row in complete)
    generated = [row["scaling"].get("generated_tokens") for row in complete]
    output_tps = tokens / elapsed if all_complete else None
    generation_tps = (sum(generated) / elapsed
                      if all_complete and all(value is not None for value in generated) else None)
    return {
        "kind": "summary", "requests": len(rows), "completed": len(complete), "failed": len(rows) - len(complete),
        "wall_seconds": elapsed, "cards": cards,
        "output_tokens": tokens, "output_tokens_per_second": output_tps,
        "output_tokens_per_second_per_card": None if output_tps is None else output_tps / cards,
        "generated_tokens_per_second": generation_tps,
        "latency_seconds": quantiles("seconds"), "thinking_ttft_seconds": quantiles("thinking_ttft_seconds"),
        "content_ttft_seconds": quantiles("content_ttft_seconds"),
        "length_limited": sum(row["finish_reason"] == "length" for row in complete),
        "incomplete_thinking": sum(row["scaling"].get("thinking_status") not in {None, "complete", "disabled"} for row in complete),
        "unknown_thinking_status": sum("thinking_status" not in row["scaling"] for row in complete),
        "correct": sum(row.get("correct", False) for row in rows), "accuracy": sum(row.get("correct", False) for row in rows) / len(rows),
        "ungraded": sum("correct" not in row for row in rows),
        "accounting_complete": all_complete and all(value is not None for value in generated),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8000"],
                        help="one URL per replica; requests are assigned round-robin")
    parser.add_argument("--settings", type=Path, default=Path("settings/inference.json"))
    parser.add_argument("--dataset", choices=("math500", "gsm8k"), default="math500")
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--selection-skip", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cards", type=int, required=True, help="actual total cards of the measured replicas")
    parser.add_argument("--deadline", type=float, default=1800, help="seconds per client request, including queuing")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--effort")
    group.add_argument("--thinking-budget", type=int)
    parser.add_argument("--max-tokens", type=int, help="explicit workload limit; omission inherits the service setting")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.questions, args.concurrency, args.repeats, args.cards, args.deadline) <= 0 or args.warmup < 0:
        parser.error("positive workload sizes and a nonnegative warmup are required")
    if args.output.exists():
        parser.error("output exists; choose a new path")
    if args.selection_skip is not None and (args.dataset != "math500" or args.selection_skip < 0):
        parser.error("--selection-skip requires MATH-500 and a nonnegative offset")
    if len(set(args.urls)) != len(args.urls):
        parser.error("replica URLs must be distinct")
    settings = load_settings(args.settings)
    config = settings["datasets"][args.dataset]
    config["selection"]["count"] = args.questions
    if args.selection_skip is not None:
        config["selection"]["skip"] = args.selection_skip
    dataset = (MATH500 if args.dataset == "math500" else GSM8K)(config)
    problems = dataset.problems
    if not problems:
        parser.error("empty workload")
    def payload(problem):
        body = {"model": settings["serve"]["server"]["served_model_name"],
                "messages": [{"role": "user", "content": dataset.prompt(problem)}]}
        for key, value in (("reasoning_effort", args.effort), ("thinking_budget", args.thinking_budget),
                           ("max_tokens", args.max_tokens)):
            if value is not None:
                body[key] = value
        return body
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as output:
            def write(row):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
            write({"kind": "metadata", "label": args.label,
                   "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   "dataset": dataset.describe(), "request": payload(problems[0]),
                   "accounting": "client wall time; nonempty deltas; all selected and unselected generation from request-local counters"})
            for url in args.urls:
                for _ in range(args.warmup):
                    row = request_one(url, payload(problems[0]), args.deadline)
                    if row["status"] != "complete":
                        write({"kind": "warmup_failure", **row})
                        raise RuntimeError(row["error"])
            jobs = [(repeat, problem) for repeat in range(args.repeats) for problem in problems]
            rows = []
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(request_one, args.urls[index % len(args.urls)], payload(problem), args.deadline):
                           (index, repeat, problem) for index, (repeat, problem) in enumerate(jobs)}
                for future in as_completed(futures):
                    index, repeat, problem = futures[future]
                    row = {"kind": "measurement", "index": index, "repeat": repeat, "problem_id": problem.id,
                           "replica": index % len(args.urls), **future.result()}
                    rows.append(row)
                    write(row)
                    print(json.dumps({key: row.get(key) for key in
                                     ("problem_id", "repeat", "status", "seconds", "thinking_ttft_seconds", "content_ttft_seconds")}), flush=True)
            elapsed = time.perf_counter() - started
            for row in rows:
                if row["status"] == "complete":
                    problem = next(problem for problem in problems if problem.id == row["problem_id"])
                    try:
                        grade = dataset.grade(row["content"], problem)
                        row["correct"] = grade.correct
                        write({"kind": "grade", "index": row["index"], "correct": grade.correct, "parseable": grade.parseable})
                    except Exception as error:
                        write({"kind": "grade_error", "index": row["index"], "error": str(error)})
            report = summary(rows, elapsed, args.cards)
            write(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()

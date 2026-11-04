"""Paired twelve-arm generation; hidden tests are never loaded in this process."""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from benchmark_is_gpu import (AdmissionMeasurements, GraphMeasurements, engine_settings, measured_solve, select_mode)
from coding_common import MODES, PublicCodingDataset, completed_keys, digest, extract_code, read_jsonl
from inference_scaling.app import ar as ar_app
from inference_scaling.app.ar import ARFamily
from inference_scaling.app.records import environment, git_state, source_sha256
from inference_scaling.app.settings import load_settings, validate_settings
from inference_scaling.datasets.base import file_sha256
from inference_scaling.shared.rng import SeedStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    settings["ar"]["engine"]["vllm"]["engine_kwargs"]["cudagraph_metrics"] = True
    settings["rewards"]["self_certainty"]["scope"] = "thinking"
    validate_settings(settings)
    dataset = PublicCodingDataset(args.manifest, shard=args.shard, shards=args.shards, limit=args.limit)
    if not dataset.problems:
        raise ValueError("empty shard")
    identity = digest({"settings": settings, "manifest": dataset.source_sha256, "modes": MODES,
                       "shard": args.shard, "shards": args.shards, "limit": args.limit,
                       "package": source_sha256(),
                       "scripts": {p.name: file_sha256(p) for p in Path(__file__).parent.glob('*coding*.py')},
                       "paired_helper": file_sha256(Path(__file__).with_name('benchmark_is_gpu.py'))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Kernel lock releases on process death; data remain append-only for audit.
    import fcntl
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_jsonl(args.output)
        done = completed_keys(old, identity)
        expected = len(dataset.problems) * len(MODES)
        if len(done) == expected:
            print(json.dumps({"kind": "already_complete", "count": expected}), flush=True)
            return
        family = graph = admission = None
        original_run = ar_app.run_conditional_is
        with args.output.open('a', encoding='utf-8') as sink:
            def write(row):
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
                sink.flush()
                os.fsync(sink.fileno())
            if not old:
                write({"kind": "metadata", "identity": identity, "settings": settings,
                       "manifest_sha256": dataset.source_sha256, "modes": MODES,
                       "shard": args.shard, "shards": args.shards,
                       "expected_measurements": expected, "problem_ids": [p.id for p in dataset.problems],
                       "git": git_state(), "environment": environment(),
                       "timing": "cold-prefix generation incl final content; excluding load, warmup, grading and writes",
                       "judge_access": "none; CPU grading after final submission; candidate correctness unavailable"})
            write({"kind": "session_start", "time": time.time(), "already_complete": len(done)})
            try:
                family = ARFamily(settings, SimpleNamespace(algorithm='is', reward='self_certainty'), dataset)
                family.load()
                if not family.raw.supports_streaming_generation:
                    raise RuntimeError('native streaming required')
                graph, admission = GraphMeasurements(family.raw), AdmissionMeasurements(family.raw)
                ar_app.run_conditional_is = partial(admission.measure_search, original_run)
                write({"kind": "engine", "config": engine_settings(family.raw),
                       "parameter_count": family.raw.parameter_count, "artifacts": family.artifacts()})
                maximum = family.max_new_tokens
                try:
                    family.max_new_tokens = 64
                    for mode in MODES:
                        select_mode(family, mode, settings, study='selection')
                        family.solve(dataset.problems[0], SeedStream(SeedStream(settings['run']['seed']).derive('coding-warmup', mode)))
                finally:
                    family.max_new_tokens = maximum
                for problem in dataset.problems:
                    meta = dataset.rows[problem.id]
                    offset = meta['global_index'] % len(MODES)
                    seed = SeedStream(settings['run']['seed']).derive('paired-benchmark', problem.id, 0)
                    for mode in MODES[offset:] + MODES[:offset]:
                        if (problem.id, mode) in done:
                            continue
                        select_mode(family, mode, settings, study='selection')
                        row = {"kind": "measurement", "problem_id": problem.id, "dataset": meta['dataset'],
                               "phase": meta['phase'], "repeat": 0, "mode": mode, "seed": seed,
                               "algorithm_config": family.config, "reward": family.choices.reward,
                               "reward_config": family.reward_settings}
                        write({"kind": "attempt_start", "time": time.time(), "problem_id": problem.id, "mode": mode})
                        print(json.dumps({"kind": "start", "problem_id": problem.id, "mode": mode}), flush=True)
                        try:
                            solved, seconds, counters, execution = measured_solve(family, problem, seed, graph, admission)
                        except BaseException as error:
                            write({**row, "status": "failed", "error": f"{type(error).__name__}: {error}",
                                   "cost_known": False})
                            raise
                        for candidate in solved['trace'].get('candidates', []):
                            candidate['correct'] = None
                        code = extract_code(solved['answer_text'])
                        row.update(status='complete', seconds=seconds, counters=counters, **execution,
                                   output_tokens=solved['output']['tokens'], prompt_tokens=solved['prompt_tokens'],
                                   output_text=solved['output']['text'], thinking_text=solved['output']['thinking'],
                                   answer_text=solved['answer_text'], submitted_code=code,
                                   submission_sha256=digest(code), parseable=bool(code), trace=solved['trace'],
                                   cost=solved['cost'], selected_reward=solved['reward'],
                                   thinking_status=solved['output']['thinking_status'],
                                   length_exhausted=solved['output']['length_exhausted'], fallbacks=solved['fallbacks'])
                        write(row)
                        done.add((problem.id, mode))
                        print(json.dumps({"kind": "complete", "problem_id": problem.id, "mode": mode,
                                          "seconds": seconds, "done": len(done), "expected": expected}), flush=True)
                write({"kind": "generation_complete", "expected": expected, "completed": len(done), "time": time.time()})
            finally:
                ar_app.run_conditional_is = original_run
                if graph is not None:
                    graph.close()
                if admission is not None:
                    admission.close()
                if family is not None:
                    family.close()


if __name__ == '__main__':
    main()

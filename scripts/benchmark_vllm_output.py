"""CPU-only output parsing/dispatch benchmark. Prints results; creates no data files.

The test engine has no model forward and no decode delay. These timings measure
Python output overhead, not LLM generation latency or GPU speedup.
Requires the repository's dev extra for the shared test fixtures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_scaling.arllm.backends.scheduling import SampleBuffer, SchedulerConfig
from inference_scaling.arllm.backends.vllm_backend import _generated_logprobs
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.shared.rewards.consilience import confidence_windows
from tests.test_streaming_vllm import DeltaEngine, backend_for
from tests.test_vllm_backend import FlatLogprobs, _Logprob


def parse_benchmark(length, repeats):
    # Sampled tokens alternate between inside/outside the top-five set.
    tokens = tuple(4 if index % 2 else 9 for index in range(length))
    rows = [([token, 3, 4, 5, 6, 7], [-2. if token == 4 else -7., -.5, -2., -3., -4., -5.]) for token in tokens]

    def run(flat):
        positions = (FlatLogprobs(rows) if flat else
                     [dict(zip(ids, map(_Logprob, logs), strict=True)) for ids, logs in rows])
        return _generated_logprobs(positions, tokens, 5)

    expected = run(False)
    result = {}
    for flat in (False, True):
        times = []
        for _ in range(repeats):
            start = perf_counter()
            actual = run(flat)
            times.append(perf_counter() - start)
            assert actual == expected
            assert confidence_windows(actual[1]) == confidence_windows(expected[1])
        tracemalloc.start()
        run(flat)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["flat" if flat else "dictionary"] = {"seconds": median(times), "peak_bytes": peak}
    result["speedup"] = result["dictionary"]["seconds"] / result["flat"]["seconds"]
    return result


class QuietTrace:
    def append(self, event):
        pass


def dispatch_once(length, optimized):
    engine = DeltaEngine(decode_delay=0, stream_interval=32 if optimized else 1)
    engine.trace = QuietTrace()
    backend = backend_for(engine, flat_logprobs=optimized)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=4, max_prefill_tokens=1))
    session = backend.open_generation_session()
    policy = SamplingConfig()
    # Four fixed requests with unequal lengths; the complete work is identical in both paths.
    requests = [GenerationRequest((1,), length // (2 ** index), policy, index + 10, str(index),
                                  statistic=TokenStatistic(policy, 2), intermediate_events=not optimized)
                for index in range(4)]
    buffers = {request.request_id: SampleBuffer() for request in requests}
    try:
        start = perf_counter()
        session.submit(requests, phase="prefix", checkpoint=None if optimized else 128)
        session.seal()
        finished = set()
        while len(finished) < len(requests):
            event = session.next_event()
            buffers[event.delta.request_id].append(event.delta)
            if event.finished:
                finished.add(event.delta.request_id)
        elapsed = perf_counter() - start
        session.close()
        counts = backend.snapshot()
        return [buffers[str(index)].sample() for index in range(4)], {
            "seconds": elapsed, "generated_tokens": counts.generated_tokens,
            "forward_token_slots": counts.generation_forward_token_slots,
            "engine_outputs": counts.stream_outputs, "algorithm_events": counts.stream_events,
        }
    finally:
        session.close()
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.tokens < 8 or args.repeats < 1:
        parser.error("tokens >= 8 and repeats >= 1 are required")
    # Warm up imports, threads, and code paths outside the measurements.
    dispatch_once(128, False)
    dispatch_once(128, True)
    result = {"tokens": args.tokens, "repeats": args.repeats, "gpu": False,
              "scope": "CPU output construction, parsing and dispatch; no model forward",
              "probability_processing": parse_benchmark(args.tokens, args.repeats)}
    timings = {"baseline": [], "compact": []}
    summaries = {}
    for repeat in range(args.repeats):
        paired = {}
        for optimized in ((False, True) if repeat % 2 == 0 else (True, False)):
            name = "compact" if optimized else "baseline"
            paired[name], counts = dispatch_once(args.tokens, optimized)
            timings[name].append(counts["seconds"])
            summaries[name] = counts
        assert paired["baseline"] == paired["compact"]
        for name in ("generated_tokens", "forward_token_slots"):
            assert summaries["baseline"][name] == summaries["compact"][name]
    for name in timings:
        summaries[name]["seconds"] = median(timings[name])
    summaries["speedup"] = summaries["baseline"]["seconds"] / summaries["compact"]["seconds"]
    result["dispatch"] = summaries
    result["samples_probabilities_statistics_costs_identical"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

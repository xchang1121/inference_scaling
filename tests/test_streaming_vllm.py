import asyncio
import threading
from math import log
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.backends.scheduling import SampleBuffer, SchedulerConfig
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.stopping import StoppedSequenceBackend
from inference_scaling.arllm.backends.vllm_backend import AsyncVLLMBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream, uniform_stream
from inference_scaling.shared.types import pointwise
from tests.test_vllm_backend import FlatLogprobs


class DeltaEngine:
    def __init__(self, decode_delay=.001, stream_interval=1):
        self.active = 0
        self.maximum_active = 0
        self.aborted = []
        self.trace = []
        self.fail = False
        self.decode_delay = decode_delay
        self.stream_interval = stream_interval
        self.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(stream_interval=stream_interval, async_scheduling=True),
            model_config=SimpleNamespace(enforce_eager=False),
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=[1, 2, 4]))

    async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
        params = sampling_params
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.trace.append(("start", request_id, params.seed, tuple(prompt["prompt_token_ids"])))
        tokens, positions, sent = [], [], 0
        context_sum = sum(prompt["prompt_token_ids"])
        try:
            for offset, uniform in enumerate(uniform_stream(params.seed, 0, params.max_tokens)):
                await asyncio.sleep(self.decode_delay * (1 + params.seed % 3))
                if self.fail:
                    raise RuntimeError("engine failed")
                probability = .2 + .1 * (context_sum % 7)
                token = 3 + int(uniform >= probability)
                distribution = {3: log(probability), 4: log(1 - probability)}
                tokens.append(token)
                context_sum += token
                tops = sorted(distribution, key=distribution.get, reverse=True)[:params.logprobs]
                positions.append({index: SimpleNamespace(logprob=distribution[index]) for index in [token, *tops]})
                finished = offset + 1 == params.max_tokens or token in params.stop_token_ids
                self.trace.append(("token", request_id, offset, token))
                delta = getattr(params, "output_kind", None) == 1
                if finished or (delta and (sent == 0 or len(tokens) - sent >= self.stream_interval)):
                    low = sent if delta else 0
                    returned = positions[low:]
                    if params.flat_logprobs:
                        # Native flat output stores sampled first, then the requested top-K (possibly duplicated).
                        rows = []
                        for token_id, position in zip(tokens[low:], returned, strict=True):
                            tops = sorted(position, key=lambda token_id: position[token_id].logprob, reverse=True)[:params.logprobs]
                            ids = [token_id, *tops]
                            rows.append((ids, [position[index].logprob for index in ids]))
                        returned = FlatLogprobs(rows)
                    yield SimpleNamespace(
                        request_id=request_id, num_cached_tokens=0, finished=finished,
                        outputs=[SimpleNamespace(token_ids=tokens[low:], logprobs=returned,
                                                 finish_reason=("stop" if token in params.stop_token_ids else "length")
                                                 if finished else None)])
                    sent = len(tokens)
                if finished:
                    break
        finally:
            self.active -= 1
            self.trace.append(("finish", request_id, params.seed, ()))

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def shutdown(self):
        pass


def backend_for(engine, **options):
    return AsyncVLLMBackend(engine, SimpleNamespace(bos_token_id=9, eos_token_id=2), model_id="fake",
                            parameter_count=100, sampling_params_factory=lambda **kwargs: SimpleNamespace(**kwargs), **options)


@pytest.mark.parametrize("block_first", [False, True])
@pytest.mark.parametrize("rollouts", [1, 2, 4])
@pytest.mark.parametrize("interval", [1, 4])
def test_delta_engine_preserves_native_request_stream_and_full_is_result(block_first, rollouts, interval):
    engine = DeltaEngine(stream_interval=interval)
    backend = backend_for(engine)
    config = ConditionalISConfig(4, rollouts, 4, 12, 1.0, block_first)
    reward = pointwise(lambda prompt, output: float(sum(output)) / 3)
    try:
        baseline = run_conditional_is(backend, (1,), config, reward, SeedStream(4))
        # A single prompt fills the prefill quota. The first-token signal must still release it.
        backend.configure_scheduler(SchedulerConfig(max_in_flight=5, max_prefill_tokens=1 if rollouts == 1 else 1000))
        engine.trace.clear()
        engine.maximum_active = 0
        before = backend.snapshot()
        result = run_conditional_is(backend, (1,), config, reward, SeedStream(4))
        assert result == baseline
        assert 1 < engine.maximum_active <= 5
        assert not engine.aborted
        assert backend.snapshot().generated_tokens > 0
        if rollouts == 1:
            assert backend.snapshot().stream_events - before.stream_events == len([
                event for event in engine.trace if event[0] == "start"])
        if not block_first:
            # Every root is one native request through its whole original horizon.
            roots = [event for event in engine.trace if event[0] == "start" and event[3] == (1,)]
            assert len(roots) == config.candidate_count
            assert all(sum(event[0] == "token" and event[1] == root[1] for event in engine.trace) == 12 for root in roots)
            if rollouts > 1:
                first_child = next(index for index, event in enumerate(engine.trace) if event[0] == "start" and len(event[3]) == 5)
                last_root_finish = max(index for index, event in enumerate(engine.trace)
                                       if event[0] == "finish" and event[1] in {root[1] for root in roots})
                assert first_child < last_root_finish
        assert backend.execution_config() == {
            "flat_logprobs": True, "stream_interval": interval, "async_scheduling": True,
            "enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4]}
    finally:
        backend.close()
    assert engine.active == 0


def test_shared_pool_bounds_independent_callers_and_keeps_logical_ids_isolated():
    engine = DeltaEngine()
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=2))
    barrier = threading.Barrier(2)

    def run(seed):
        session = backend.open_generation_session()
        try:
            barrier.wait()
            session.submit([GenerationRequest((1,), 3, SamplingConfig(), seed + index, f"same:{index}") for index in range(3)])
            buffers = {f"same:{index}": SampleBuffer() for index in range(3)}
            done = set()
            while len(done) < 3:
                event = session.next_event()
                buffers[event.delta.request_id].append(event.delta)
                if event.finished:
                    done.add(event.delta.request_id)
            return [buffer.sample() for buffer in buffers.values()]
        finally:
            session.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, (10, 40)))
        assert all(len(sample.token_ids) == 3 for samples in results for sample in samples)
        assert engine.maximum_active == 2
        assert backend.snapshot().maximum_in_flight_requests == 2
        assert len({event[1] for event in engine.trace if event[0] == "start"}) == 6
        assert not backend._admission.queue.pending and not backend._admission.queue.active
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ["engine", "reward"])
def test_failures_abort_and_release_every_request(failure):
    engine = DeltaEngine()
    engine.fail = failure == "engine"
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=2))

    def reward(prompt, output):
        raise RuntimeError("reward failed")

    try:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            run_conditional_is(backend, (1,), ConditionalISConfig(3, 3, 1, 6, 1.0, False), reward, SeedStream(5))
        assert engine.active == 0
        assert not backend._admission.queue.pending and not backend._admission.queue.active
        assert backend.snapshot().maximum_in_flight_requests <= 2
    finally:
        backend.close()


def test_explicit_scope_stop_aborts_only_its_request_and_releases_capacity():
    engine = DeltaEngine()
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=1))
    session = backend.open_generation_session()
    try:
        session.submit([GenerationRequest((1,), 8, SamplingConfig(), index, str(index)) for index in range(2)], checkpoint=1)
        first = session.next_event()
        session.stop(first.delta.request_id)
        other = "1" if first.delta.request_id == "0" else "0"
        while True:
            event = session.next_event()
            assert event.delta.request_id == other
            if event.finished:
                break
        assert engine.aborted
    finally:
        session.close()
        backend.close()
    assert engine.active == 0


def test_terminal_hint_keeps_deltas_for_a_multitoken_thinking_boundary():
    engine = DeltaEngine(decode_delay=.001, stream_interval=4)
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=1))
    parser = ThinkingParser((ThinkingFormat((3, 4), (9,)),))
    scoped = StoppedSequenceBackend(backend, thinking_parser=parser, thinking_prompt=(9,), eos_token_id=2)
    request = GenerationRequest((9,), 128, SamplingConfig(), 0, "scope", intermediate_events=False)
    session = scoped.open_generation_session()
    buffer = SampleBuffer()
    try:
        session.submit([request])
        while True:
            event = session.next_event()
            buffer.append(event.delta)
            if event.finished:
                break
        session.close()
        sample = buffer.sample()
        expected = scoped.sample_batch([request])[0]
        assert sample.token_ids == expected.token_ids and sample.token_logprobs == expected.token_logprobs
        assert sample.finish_reason == "stop" and len(sample.token_ids) < 128
        assert engine.aborted and sample.forward_token_slots is None
        assert not backend._admission.queue.active
    finally:
        session.close()
        backend.close()


@pytest.mark.parametrize("rollouts", [1, 4])
@pytest.mark.parametrize("marker", [(4,), (3, 4)])
def test_streamed_scope_rewards_use_parent_prefix_statistics_before_the_parent_finishes(rollouts, marker):
    engine = DeltaEngine(decode_delay=0, stream_interval=32)
    backend = backend_for(engine)
    statistic = TokenStatistic(SamplingConfig(), top_k=1)
    parser = ThinkingParser((ThinkingFormat(marker, (9,)),))
    config = ConditionalISConfig(4, rollouts, 1, 36, 1.0, False)

    def unavailable(*args):
        raise AssertionError("generated prefix statistics should already be recorded")

    backend.token_statistics = unavailable
    try:
        for seed in range(4):
            results = []
            for streaming in (False, True):
                backend.configure_scheduler(SchedulerConfig(enabled=streaming, max_in_flight=4))
                scoped = StoppedSequenceBackend(backend, thinking_parser=parser, thinking_prompt=(9,), eos_token_id=2)
                recorder = StatisticRecorder(scoped, statistic)
                reward = ConsilienceReward(recorder, statistic, parser)
                results.append(run_conditional_is(recorder, (9,), config, reward.batch, SeedStream(seed)))
            assert results[0] == results[1]
    finally:
        backend.close()


def test_reward_can_use_the_same_engine_while_long_generation_events_are_unread():
    # A bounded, blocking per-token event queue would deadlock here: the caller
    # waits for scoring admission, while generation waits for that caller to read.
    engine = DeltaEngine(decode_delay=0)
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=1))
    session = backend.open_generation_session()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        session.submit([GenerationRequest((1,), 16384, SamplingConfig(), 2, "long")])
        first = session.next_event()
        buffer = SampleBuffer()
        buffer.append(first.delta)
        future = executor.submit(backend.sample_batch, [GenerationRequest((1,), 1, SamplingConfig(), 3, "reward")])
        assert len(future.result(timeout=10)[0].token_ids) == 1
        while not first.finished:
            first = session.next_event()
            buffer.append(first.delta)
        assert len(buffer.sample().token_ids) == 16384
    finally:
        session.close()
        backend.close()
        executor.shutdown(wait=True)

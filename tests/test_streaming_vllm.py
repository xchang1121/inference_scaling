import asyncio
import threading
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
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, TokenStatistic
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream, uniform_stream
from inference_scaling.shared.types import pointwise


class DeltaEngine:
    def __init__(self, decode_delay=.001):
        self.active = 0
        self.maximum_active = 0
        self.aborted = []
        self.trace = []
        self.fail = False
        self.decode_delay = decode_delay

    async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
        params = sampling_params
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.trace.append(("start", request_id, params.seed, tuple(prompt["prompt_token_ids"])))
        tokens, positions = [], []
        try:
            for offset, uniform in enumerate(uniform_stream(params.seed, 0, params.max_tokens)):
                await asyncio.sleep(self.decode_delay * (1 + params.seed % 3))
                if self.fail:
                    raise RuntimeError("engine failed")
                token = 3 + int(uniform >= .5)
                tokens.append(token)
                positions.append({token: SimpleNamespace(logprob=-.6931471805599453)})
                finished = offset + 1 == params.max_tokens or token in params.stop_token_ids
                self.trace.append(("token", request_id, offset, token))
                if getattr(params, "output_kind", None) == 1 or finished:
                    delta = getattr(params, "output_kind", None) == 1
                    yield SimpleNamespace(
                        request_id=request_id, num_cached_tokens=0, finished=finished,
                        outputs=[SimpleNamespace(token_ids=tokens[-1:] if delta else tokens,
                                                 logprobs=positions[-1:] if delta else positions,
                                                 finish_reason=("stop" if token in params.stop_token_ids else "length")
                                                 if finished else None)])
                if finished:
                    break
        finally:
            self.active -= 1
            self.trace.append(("finish", request_id, params.seed, ()))

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def shutdown(self):
        pass


def backend_for(engine):
    return AsyncVLLMBackend(engine, SimpleNamespace(bos_token_id=9, eos_token_id=2), model_id="fake",
                            parameter_count=100, sampling_params_factory=lambda **kwargs: SimpleNamespace(**kwargs))


@pytest.mark.parametrize("block_first", [False, True])
def test_delta_engine_preserves_native_request_stream_and_full_is_result(block_first):
    engine = DeltaEngine()
    backend = backend_for(engine)
    config = ConditionalISConfig(3, 3, 2, 6, 1.0, block_first)
    reward = pointwise(lambda prompt, output: float(sum(output)) / 3)
    try:
        baseline = run_conditional_is(backend, (1,), config, reward, SeedStream(4))
        backend.configure_scheduler(SchedulerConfig(max_in_flight=4, max_prefill_tokens=20))
        engine.trace.clear()
        engine.maximum_active = 0
        result = run_conditional_is(backend, (1,), config, reward, SeedStream(4))
        assert result == baseline
        assert engine.maximum_active <= 4
        assert not engine.aborted
        assert backend.snapshot().generated_tokens > 0
        if not block_first:
            # Every root is one native request through its whole original horizon.
            roots = [event for event in engine.trace if event[0] == "start" and event[3] == (1,)]
            assert len(roots) == config.candidate_count
            assert all(sum(event[0] == "token" and event[1] == root[1] for event in engine.trace) == 6 for root in roots)
            first_child = next(index for index, event in enumerate(engine.trace) if event[0] == "start" and len(event[3]) == 3)
            first_root_finish = min(index for index, event in enumerate(engine.trace)
                                    if event[0] == "finish" and event[1] in {root[1] for root in roots})
            assert first_child < first_root_finish
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


def test_streamed_scope_rewards_use_parent_prefix_statistics_before_the_parent_finishes():
    engine = DeltaEngine(decode_delay=0)
    backend = backend_for(engine)
    statistic = TokenStatistic(SamplingConfig(), top_k=1)
    parser = ThinkingParser((ThinkingFormat((4,), (9,)),))
    config = ConditionalISConfig(3, 3, 1, 6, 1.0, False)

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

                def reward(prompt, outputs, recorder=recorder):
                    return [sum(values) for values in recorder.token_statistics([ScoreRequest(prompt, tuple(outputs))], statistic)]

                results.append(run_conditional_is(recorder, (9,), config, reward, SeedStream(seed)))
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

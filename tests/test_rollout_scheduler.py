from dataclasses import replace

import pytest

from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter, RetainedSequence, run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.backends.scheduling import Admission, AdmissionQueue, SampleBuffer, SchedulerConfig
from tests.simulation import SimulatedBackend
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.stopping import StoppedSequenceBackend
from tests.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, TokenStatistic
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import pointwise


def _model():
    return TabularAutoregressiveBackend({(): [.25, .25, .5], (0,): [.7, .25, .05]}, fallback=[.48, .48, .04])


def _duration(request, offset, sample):
    # Deliberately outcome-dependent service time: the scheduler must not select
    # fast completions preferentially even when runtime correlates with reward.
    return .001 * (1 + 6 * int(sample.token_ids[-1] == 1) + request.seed % 3)


@pytest.mark.parametrize("block_first", [False, True])
@pytest.mark.parametrize("capacity,policy", [(1, "fifo"), (2, "fair_lpt"), (8, "fair_lpt")])
def test_all_weights_choices_and_tokens_match_for_changed_execution_orders(block_first, capacity, policy):
    config = ConditionalISConfig(4, 3, 2, 7, .7, block_first)
    reward = pointwise(lambda prompt, tokens: .4 * sum(tokens) - .1 * len(tokens))
    policy_config = SamplingConfig(eos_token_id=2)
    for seed in range(3):
        plain = run_conditional_is(_model(), (), config, reward, SeedStream(seed), sampling=policy_config)
        simulator = SimulatedBackend(_model(), SchedulerConfig(max_in_flight=capacity, policy=policy), token_seconds=_duration)
        streamed = run_conditional_is(simulator, (), config, reward, SeedStream(seed), sampling=policy_config)
        assert streamed == plain
        assert simulator.queue.maximum_active <= capacity
        assert not simulator.queue.active and not simulator.queue.pending and not simulator.jobs


def test_retained_children_start_without_waiting_for_fresh_candidate_tails():
    config = ConditionalISConfig(3, 3, 1, 8, 1.0, False)
    simulator = SimulatedBackend(TabularAutoregressiveBackend({}, fallback=[.5, .5]),
                                 SchedulerConfig(max_in_flight=6), token_seconds=_duration)
    adapter = ConditionalISAdapter(backend=simulator, prompt=(), config=config, sampling=SamplingConfig(),
                                    reward=pointwise(lambda prompt, tokens: float(sum(tokens))))
    state = RetainedSequence((0, 1, 1, 0, 0, 1, 1, 0), 4.0)
    record, _ = adapter.step(state, 1, SeedStream(3))
    submitted = {event["id"]: event["time"] for event in simulator.trace if event["event"] == "submit"}
    completed = {event["id"]: event["time"] for event in simulator.trace if event["event"] == "finish"}
    assert submitted["conditional-is:step:1:candidate:0:rollout:1"] == 0
    assert submitted["conditional-is:step:1:candidate:1:rollout:1"] < completed["conditional-is:step:1:candidate:1"]
    assert record.candidates[0].rollouts[0].reward == 4.0
    assert record.rollout_evaluations_performed == 8


@pytest.mark.parametrize("marker", [(1,), (0, 1)])
@pytest.mark.parametrize("block_first", [False, True])
@pytest.mark.parametrize("sibling_replay", [False, True])
def test_streaming_scope_and_recorded_rewards_match_at_split_boundaries(marker, block_first, sibling_replay):
    parser = ThinkingParser((ThinkingFormat(marker, (3,)),))
    statistic = TokenStatistic(SamplingConfig(), top_k=2)
    config = ConditionalISConfig(3, 3, 2, 7, .7, block_first, sibling_replay=sibling_replay)
    for seed in range(3):
        results = []
        for streamed in (False, True):
            raw = SimulatedBackend(_model(), SchedulerConfig(max_in_flight=2), streaming=streamed, token_seconds=_duration)
            scoped = StoppedSequenceBackend(raw, thinking_parser=parser, thinking_prompt=(3,), eos_token_id=2)
            recorder = StatisticRecorder(scoped, statistic)

            def reward(prompt, outputs, recorder=recorder):
                return [sum(values) / len(values) for values in recorder.token_statistics(
                    [ScoreRequest(prompt, tuple(outputs))], statistic)]

            results.append(run_conditional_is(recorder, (3,), config, reward, SeedStream(seed), sampling=SamplingConfig()))
        assert results[0] == results[1]


def test_scope_resumes_a_native_empty_thinking_close_with_the_same_offset():
    class NativeStop(TabularAutoregressiveBackend):
        def sample_batch(self, requests):
            outputs = super().sample_batch(requests)
            result = []
            for request, sample in zip(requests, outputs, strict=True):
                end = next((index + 1 for index, token in enumerate(sample.token_ids)
                            if (token,) in request.stop_sequences), None)
                sample = replace(sample, forward_token_slots=len(request.prefix) + (end or len(sample.token_ids)) - 1)
                result.append(sample if end is None else replace(
                    sample, token_ids=sample.token_ids[:end], token_logprobs=sample.token_logprobs[:end],
                    reference_token_logprobs=sample.reference_token_logprobs[:end],
                    token_cdf_bounds=sample.token_cdf_bounds[:end], finish_reason="stop"))
            return result

    raw = NativeStop({(3,): [0, 1, 0], (3, 1): [1, 0, 0], (3, 1, 0): [0, 0, 1]})
    parser = ThinkingParser((ThinkingFormat((1,), (3,)),))
    request = GenerationRequest((3,), 6, SamplingConfig(), 1, "empty")
    plain = StoppedSequenceBackend(raw, thinking_parser=parser, thinking_prompt=(3,), eos_token_id=2)
    simulator = SimulatedBackend(raw, SchedulerConfig(max_in_flight=1))
    streamed = StoppedSequenceBackend(simulator, thinking_parser=parser, thinking_prompt=(3,), eos_token_id=2)
    session = streamed.open_generation_session()
    buffer = SampleBuffer()
    try:
        session.submit([request])
        while True:
            event = session.next_event()
            buffer.append(event.delta)
            if event.finished:
                break
    finally:
        session.close()
    assert buffer.sample() == plain.sample_batch([request])[0]
    # Empty close: one prefill slot; resumed request: two prefill plus one decode slot.
    assert buffer.sample().forward_token_slots == plain.sample_batch([request])[0].forward_token_slots == 4


def test_admission_cap_fairness_prefill_pressure_aging_and_tail_drain():
    queue = AdmissionQueue(SchedulerConfig(max_in_flight=2, max_prefill_tokens=10, aging_seconds=2, tail_prefill_tokens=4))
    queue.add(Admission("a", "q1", 8, 3), 0)
    queue.add(Admission("b", "q1", 8, 30), 0)
    assert queue.admit(0).key == "b"  # longest estimate first
    queue.add(Admission("c", "q2", 2, 2, "prefix"), 0)
    assert queue.admit(0).key == "c"  # another question gets the second slot
    assert queue.admit(0) is None
    queue.finish("c", 2)
    assert queue.admit(.1, kv_usage=.99) is None
    assert queue.admit(.1) is None  # active prefills would exceed ten tokens
    queue.progress("b")
    assert queue.admit(.1).key == "a"
    queue.finish("a", 3)
    queue.sealed.add("q1")
    queue.add(Admission("d", "q2", 20, 5), .2)
    assert queue.admit(.3) is None  # protect the remaining tail from a long prefill
    assert queue.admit(2.3).key == "d"  # aging overrides tail protection
    queue.finish("b", 30)
    queue.finish("d", 5)
    queue.add(Admission("e", "q3", 100, 1), 3)
    assert queue.admit(3, kv_usage=1).key == "e"  # idle liveness despite stale pressure


def test_old_large_prefill_cannot_be_starved_by_new_small_prefills():
    queue = AdmissionQueue(SchedulerConfig(max_in_flight=3, max_prefill_tokens=10, aging_seconds=1))
    queue.add(Admission("running", "a", 4, 2), 0)
    assert queue.admit(0).key == "running"
    queue.add(Admission("large", "b", 9, 2), 0)
    queue.add(Admission("small", "c", 1, 2), 1)
    assert queue.admit(2) is None
    queue.progress("running")
    assert queue.admit(2).key == "large"


def test_reward_failure_closes_all_queued_and_running_jobs():
    simulator = SimulatedBackend(_model(), SchedulerConfig(max_in_flight=2))

    def reward(prompt, tokens):
        raise ValueError("reward failed")

    with pytest.raises(ValueError, match="reward failed"):
        run_conditional_is(simulator, (), ConditionalISConfig(3, 4, 1, 7, 1, False), reward, SeedStream(2))
    assert not simulator.queue.active and not simulator.queue.pending and not simulator.jobs


def test_exhaustive_transition_matrix_and_target_invariance():
    from scripts.simulate_rollout_scheduling import exact_transition_check

    report = exact_transition_check()
    assert len(report["states"]) == 4
    assert all(case["maximum_matrix_difference"] == 0 for case in report["schedules"])
    assert all(case["target_invariance_max_error"] < 1e-14 for case in report["schedules"])

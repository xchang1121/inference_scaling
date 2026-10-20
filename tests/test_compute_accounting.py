from dataclasses import dataclass, replace
from types import SimpleNamespace

from inference_scaling.app.records import Meter, RequestMeter, pass_at_k, summarize_executions
from inference_scaling.arllm.types import ScoreRequest
from training.grpo import estimate_grpo_compute, estimate_grpo_compute_from_logs


def _cost(meter: Meter) -> dict | None:
    # The same closures ``ARFamily.solve`` uses, so a partial attributed delta cannot slip through.
    return meter.cost(lambda delta: delta.get("generation_forward_token_slots", 0)
                      + delta.get("score_forward_token_slots", 0),
                      lambda _role, delta: delta.get("estimated_dense_forward_flops", 0))


@dataclass(frozen=True)
class _Counters:
    generation_forward_token_slots: int = 0
    score_forward_token_slots: int = 0
    estimated_dense_forward_flops: int = 0
    cancelled_requests: int = 0
    preemptions: int = 0


class _Engine:
    """A backend whose counters cover all of its work, as a synchronous engine's do."""

    def __init__(self) -> None:
        self.counters = _Counters()

    def snapshot(self) -> _Counters:
        return self.counters


def test_meter_attributes_per_request_work_when_problems_run_concurrently():
    # No engine counters: concurrent problems share them, so the caller reports what each request cost.
    meter = Meter({})
    backend = SimpleNamespace(sample_batch=lambda requests: [SimpleNamespace(forward_token_slots=value) for value in requests],
                              score_batch=lambda requests: [])
    measured = RequestMeter(backend, meter, 10)
    with meter.phase("search"):
        measured.sample_batch([30])
    assert _cost(meter) == {"phases": {"search": {"base": {"generation_forward_token_slots": 30,
                                                            "score_forward_token_slots": 0,
                                                            "estimated_dense_forward_flops": 600}}},
                            "forward_token_slots": 30, "flops": 600}
    with meter.phase("score"):
        measured.score_batch([ScoreRequest((1,), ((2,),))])
    assert _cost(meter) is None
    # A phase that measured nothing does not invent a zero cost.
    empty = Meter({})
    with empty.phase("search"):
        pass
    assert _cost(empty) is None
    with empty.phase("finish"):
        empty.attribute({"generation_forward_token_slots": 7})
    assert _cost(empty)["forward_token_slots"] == 7
    with empty.phase("unmeasured_search"):
        RequestMeter(backend, empty, 10).sample_batch([None])
    assert _cost(empty) is None


def test_meter_ignores_attribution_when_the_counters_already_cover_it():
    engine = _Engine()
    meter = Meter({"base": engine})
    with meter.phase("search"):
        engine.counters = _Counters(12, 0, 240)
        meter.attribute({"generation_forward_token_slots": 12, "estimated_dense_forward_flops": 240})
        meter.attribute(None)
    # The engine delta is used as it is, so the same work is not counted twice.
    assert _cost(meter)["forward_token_slots"] == 12
    for counter in ("cancelled_requests", "preemptions"):
        partial = Meter({"base": engine})
        with partial.phase("search"):
            engine.counters = replace(engine.counters, **{counter: 1})
        assert _cost(partial) is None


def test_execution_summary_marks_missing_cost_after_a_crash():
    records = [{"problem_id": "a", "draw": 0}, {"problem_id": "b", "draw": 0}]
    executions = [{"status": "complete", "cost": {"forward_token_slots": 30, "flops": 600},
                   "wall_seconds": 2, "completed": records[:1], "output_tokens": 10},
                  {"status": "running"}]
    summary = summarize_executions(executions, records)
    assert summary["cost_total"] == {"forward_token_slots": 30, "flops": 600}
    assert summary["output_tokens_per_second"] == 5
    assert summary["unmetered_records"] == summary["incomplete_executions"] == 1
    assert not summary["complete_accounting"]
    # All outputs are present, but a cancelled stream left unobserved native work.
    executions[0]["cost"]["unmeasured_work_events"] = 1
    summary = summarize_executions(executions[:1], records[:1])
    assert summary["incomplete_executions"] == summary["unmetered_records"] == 0
    assert summary["unmeasured_work_events"] == 1 and not summary["complete_accounting"]


def test_grpo_compute_is_split_into_observed_forward_equivalents() -> None:
    compute = estimate_grpo_compute(
        model_sequence_tokens=1_000,
        generated_completions=10,
        total_parameters=100,
        trainable_parameters=5,
        optimizer_steps=2,
        gradient_checkpointing=True,
        reference_scoring=True,
    )

    assert compute.rollout_generation_forward_token_slots == 990
    assert compute.reference_scoring_forward_token_slots == 1_000
    assert compute.policy_forward_backward_equivalent_token_slots == 3_000
    assert compute.total_forward_equivalent_token_slots == 4_990
    assert compute.estimated_dense_model_flops == 998_000
    assert compute.estimated_optimizer_flops == 100
    assert compute.estimated_total_flops == 998_100


def test_grpo_logs_reconstruct_padded_generation_and_training_slots() -> None:
    compute = estimate_grpo_compute_from_logs(
        log_history=[
            {
                "step": 1,
                "num_tokens": 40,
                "completions/mean_length": 5,
                "completions/max_length": 6,
            },
            {
                "step": 2,
                "num_tokens": 90,
                "completions/mean_length": 6,
                "completions/max_length": 8,
            },
        ],
        sequences_per_optimizer_step=4,
        generated_completions=8,
        total_parameters=100,
        trainable_parameters=5,
        optimizer_steps=2,
        gradient_checkpointing=True,
        reference_scoring=True,
    )

    assert compute.rollout_generation_forward_token_slots == 94
    assert compute.reference_scoring_forward_token_slots == 102
    assert compute.policy_forward_backward_equivalent_token_slots == 306
    assert compute.total_forward_equivalent_token_slots == 502
    assert compute.estimated_dense_model_flops == 100_400
    assert compute.estimated_optimizer_flops == 100
    assert compute.estimated_total_flops == 100_500
    assert compute.accounting_basis.startswith("padded forward token slots")


def test_standard_pass_at_k_estimator() -> None:
    assert pass_at_k(correct=1, draws=4, k=1) == 0.25
    assert pass_at_k(correct=1, draws=4, k=2) == 0.5
    assert pass_at_k(correct=1, draws=4, k=4) == 1.0

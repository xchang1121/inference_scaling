from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from scripts.benchmark_is_gpu import comparison_summary, pair_summary
from inference_scaling.app.records import Meter, RequestMeter, pass_at_k, summarize_executions
from inference_scaling.arllm.types import ScoreRequest


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
    backend = SimpleNamespace(sample_batch=lambda requests: [SimpleNamespace(forward_token_slots=value, token_ids=(1, 2)) for value in requests],
                              score_batch=lambda requests: [])
    measured = RequestMeter(backend, meter, 10)
    with meter.phase("search"):
        measured.sample_batch([30])
    assert _cost(meter) == {"phases": {"search": {"base": {"generation_forward_token_slots": 30,
                                                            "score_forward_token_slots": 0, "generated_tokens": 2,
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

    # Paired benchmark failures must not turn a completed subset into a speedup.
    base = {"problem_id": "a", "repeat": 0, "mode": "base", "status": "complete", "seconds": 2.,
            "correct": True, "output_tokens": 3, "counters": {"generated_tokens": 3},
            "graph_execution": {"estimated_dense_padded_flops": 60},
            "length_exhausted": False, "thinking_status": "complete", "fallbacks": []}
    fast = {**base, "mode": "is", "seconds": 3.}
    paired = pair_summary([base, fast], 1)
    assert paired["latency_ratio_is_over_base"] == 1.5 and paired["sample_quality_gate"]
    missing = pair_summary([base], 1)
    assert missing["latency_ratio_is_over_base"] is None and missing["missing_measurements"] == 1
    failed = {"problem_id": "a", "repeat": 0, "mode": "is", "status": "failed"}
    assert pair_summary([base, failed], 1)["sample_quality_gate"] is None
    unknown = {key: value for key, value in fast.items() if key != "correct"}
    unknown["counters"] = {"generated_tokens": 3, "cancelled_requests": 1}
    incomplete = pair_summary([base, unknown], 1)
    assert incomplete["sample_quality_gate"] is None and incomplete["is"]["generated_tokens"] is None
    with pytest.raises(ValueError, match="duplicate"):
        pair_summary([base, base], 1)
    alternatives = [base, {**base, "mode": "sir-k2", "correct": False}, {**fast, "mode": "block-met-k2"}]
    matrix = comparison_summary(alternatives, 1, ("base", "sir-k2", "block-met-k2"))
    assert matrix["selection_pairs"][0]["wins"] == 1 and matrix["selection_pairs"][0]["losses"] == 0
    assert matrix["versus_base"]["sir-k2"]["losses"] == 1
    assert matrix["versus_base"]["block-met-k2"]["latency_ratio_is_over_base"] == 1.5
    supplement = comparison_summary([{**base, "mode": "bon-consilience"}, {**fast, "mode": "is-self-certainty"}],
                                    1, ("bon-consilience", "is-self-certainty"))
    assert supplement["versus_base"] == {}
    assert supplement["reward_pairs"][0]["complete_pairs"] == 1
    assert supplement["reward_pairs"][0]["latency_ratio_is_over_base"] == 1.5
    from types import SimpleNamespace
    from inference_scaling.app.settings import load_settings
    from scripts.benchmark_is_gpu import select_mode
    settings = load_settings()
    family = SimpleNamespace(ar=settings["ar"])
    for mode, algorithm, reward in (("bon-consilience", "best_of_n", "consilience"),
                                     ("is-self-certainty", "is", "self_certainty"), ("is", "is", "consilience")):
        select_mode(family, mode, settings)
        assert (family.choices.algorithm, family.choices.reward) == (algorithm, reward)
        assert family.reward_settings["scope"] == "thinking"
    assert family.config == settings["ar"]["algorithms"]["is"]
    assert settings["ar"]["algorithms"]["best_of_n"]["samples"] == 8  # Mode overrides remain local.


def test_standard_pass_at_k_estimator() -> None:
    assert pass_at_k(correct=1, draws=4, k=1) == 0.25
    assert pass_at_k(correct=1, draws=4, k=2) == 0.5
    assert pass_at_k(correct=1, draws=4, k=4) == 1.0

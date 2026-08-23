from experiments.arllm.run_qwen15b_draft_speculation_screen import summarize


def _record(arm: str, wall: float, *, proposed: int = 0, accepted: int = 0):
    return {
        "arm": arm,
        "correct": True,
        "prediction": "7",
        "wall_seconds": wall,
        "output_tokens": 8,
        "main_model_forward_token_slots": 10,
        "draft_model_forward_token_slots": proposed,
        "main_model_flops": 100,
        "draft_model_flops": proposed * 3,
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted,
        "verification_rounds": 2 if proposed else 0,
        "peak_allocated_mib": 100.0,
    }


def test_summary_selects_only_complete_wall_time_improvement() -> None:
    records = [
        _record("target_only", 10.0),
        _record("target_only", 10.0),
        _record("draft_model_k2", 8.0, proposed=6, accepted=4),
        _record("draft_model_k2", 9.0, proposed=6, accepted=5),
        _record("draft_model_k4", 11.0, proposed=8, accepted=6),
        _record("draft_model_k4", 10.0, proposed=8, accepted=5),
    ]

    summary, decision = summarize(
        records,
        ("target_only", "draft_model_k2", "draft_model_k4"),
        expected_records_per_arm=2,
    )

    assert decision["complete"] is True
    assert decision["status"] == "accepted"
    assert decision["selected_default"] == "draft_model_k2"
    assert summary[1]["wall_ratio_to_target_only"] == 0.85
    assert summary[1]["draft_acceptance_rate"] == 0.75
    assert summary[1]["total_flops_ratio_to_target_only"] > 1.0


def test_incomplete_summary_remains_running() -> None:
    _summary, decision = summarize(
        [_record("target_only", 10.0)],
        ("target_only", "draft_model_k2"),
        expected_records_per_arm=2,
    )

    assert decision["complete"] is False
    assert decision["status"] == "running"

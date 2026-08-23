from experiments.arllm.run_qwen15b_mh_stack import summarize


def _arm(name: str, wall: float, flops: int, *, cache_wall=0.0, cache_flops=0):
    replay = name.startswith("replay")
    return {
        "name": name,
        "cache_build": {
            "telemetry": {"wall_seconds": cache_wall},
            "main_model": {"estimated_dense_forward_flops": cache_flops},
        },
        "online": {
            "telemetry": {"wall_seconds": wall},
            "main_model": {"estimated_dense_forward_flops": flops},
            "acceptance_rate": 0.75,
            "proposal_sources": (
                {"base": 20, "history": 12} if replay else {"base": 32, "history": 0}
            ),
        },
    }


def _run(stack_wall: float):
    return {
        "seed": 3,
        "arms": [
            _arm("base_uniform", 10.0, 100),
            _arm("base_multiscale", 9.0, 90),
            _arm("replay_uniform", 8.5, 85, cache_wall=2.0, cache_flops=10),
            _arm(
                "replay_multiscale",
                stack_wall,
                80,
                cache_wall=2.0,
                cache_flops=10,
            ),
        ],
    }


def test_stack_summary_reports_interaction_and_break_even() -> None:
    summary = summarize([_run(8.0)])

    assert summary["complete"] is True
    assert summary["decision"]["status"] == "accepted"
    assert summary["comparisons"]["stack_over_base_uniform"]["wall_factor"]["mean"] == 0.8
    assert summary["comparisons"]["stack_over_uniform_replay"]["wall_factor"]["mean"] < 1
    assert summary["arms"]["replay_multiscale"]["history_proposal_fraction"]["mean"] == 0.375
    assert summary["break_even_queries_by_seed"]["wall_queries"] == [2]
    assert summary["break_even_queries_by_seed"]["flops_queries"] == [1]


def test_stack_summary_rejects_insufficient_combined_gain() -> None:
    summary = summarize([_run(9.6)])

    assert summary["decision"]["status"] == "rejected"

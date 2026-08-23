from experiments.arllm.run_qwen15b_is_stack import summarize


def _phase(wall: float, main: int, auxiliary: int = 0):
    return {
        "wall_seconds": wall,
        "main_model_estimated_dense_forward_flops": main,
        "auxiliary_model_estimated_dense_forward_flops": auxiliary,
        "total_estimated_dense_forward_flops": main + auxiliary,
    }


def _arm(
    name: str, wall: float, main: int, *, cache_wall=0.0, cache_main=0, cache_aux=0
):
    outputs = [[1, 2], [3, 4]]
    return {
        "name": name,
        "cache_build": _phase(cache_wall, cache_main, cache_aux),
        "online": _phase(wall, main),
        "cold_total": _phase(
            wall + cache_wall,
            main + cache_main,
            cache_aux,
        ),
        "accuracy": 0.5,
        "rollout_reuse_rate": 0.5 if name.startswith("replay") else 0.0,
        "candidate_draws_reused": 8 if "candidate_cache" in name else 0,
        "outputs": outputs,
    }


def _run(stack_wall: float = 5.0):
    return {
        "seed": 1,
        "arms": [
            _arm("fresh_sequential", 10.0, 100),
            _arm("fresh_continuous", 8.0, 102),
            _arm(
                "replay_sequential",
                8.0,
                80,
                cache_wall=4.0,
                cache_main=20,
                cache_aux=10,
            ),
            _arm(
                "replay_candidate_cache",
                7.0,
                70,
                cache_wall=4.0,
                cache_main=20,
                cache_aux=10,
            ),
            _arm(
                "replay_candidate_cache_continuous",
                stack_wall,
                71,
                cache_wall=3.0,
                cache_main=20,
                cache_aux=10,
            ),
        ],
    }


def test_is_stack_summary_reports_factors_exactness_and_break_even() -> None:
    summary = summarize([_run()])

    assert summary["decision"]["status"] == "accepted"
    assert (
        summary["comparisons"]["candidate_cache_on_replay"]["online"][
            "main_model_estimated_dense_forward_flops"
        ]["mean"]
        == 0.875
    )
    assert (
        summary["comparisons"]["continuous_batching_on_cached_replay"]["online"][
            "wall_seconds"
        ]["mean"]
        == 5 / 7
    )
    assert all(
        item["all_runs_token_exact"] for item in summary["execution_exactness"].values()
    )
    assert summary["break_even_queries_by_seed"]["wall_queries"] == [2]
    assert summary["break_even_queries_by_seed"]["total_flops_queries"] == [1]
    assert "consumes each evaluation record once" in summary["break_even_scope"]


def test_is_stack_summary_rejects_a_slow_batching_interaction() -> None:
    summary = summarize([_run(stack_wall=7.0)])

    assert summary["decision"]["status"] == "rejected"


def test_is_stack_summary_rejects_execution_drift() -> None:
    run = _run()
    run["arms"][-1]["outputs"] = [[9], [3, 4]]

    summary = summarize([run])

    assert summary["decision"]["status"] == "rejected"

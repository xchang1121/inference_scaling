from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.arllm.run_qwen15b_isir_screen import build_commands
from experiments.arllm.summarize_qwen15b_isir import summarize_isir_screen


def test_isir_screen_command_grid_has_equal_distinct_state_budget(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=Path("configs/gsm8k_quick.toml"),
        tag="unit",
        draws=2,
        limit=8,
        raw_root=tmp_path / "raw",
        baseline=Path("results/validation/gsm8k_quick_comparison_validated.json"),
        output=tmp_path / "summary.json",
    )
    commands = build_commands(args)
    assert len(commands) == 7
    state_budgets = []
    for command in commands[:-1]:
        pool = int(command[command.index("--iterated-pool-size") + 1])
        updates = int(command[command.index("--iterated-updates") + 1])
        state_budgets.append(1 + updates * (pool - 1))
        assert "--conditional-reward" in command
        assert "frozen_consensus" in command
    assert state_budgets == [9] * 6
    assert commands[-1][commands[-1].index("--baseline") + 1] == str(args.baseline)


def test_isir_screen_summary_uses_question_clustered_pairs(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """[run]
name = "fixture"
sample_count = 2
[models]
base = "models/Qwen2.5-1.5B-Instruct"
base_revision = "fixture-revision"
base_weight_sha256 = "fixture-sha256"
[generation]
max_new_tokens = 16
[conditional_is]
rollout_count = 1
block_size = 8
[iterated_is]
pilot_samples = 2
""",
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    for arm, pool, updates in (("n9-u1", 9, 1), ("n5-u2", 5, 2), ("n3-u4", 3, 4)):
        for draw in range(2):
            directory = raw / "fixture" / f"iterated_conditional_is-study-{arm}-draw{draw}"
            directory.mkdir(parents=True)
            records = []
            for index in (10, 20):
                records.append(
                    {
                        "problem_index": index,
                        "draw_index": draw,
                        "correct": arm != "n3-u4" or index == 10,
                        "elapsed_seconds": 2.0,
                        "backend_delta": {"estimated_dense_forward_flops": 100},
                        "diagnostics": {
                            "pool_size": pool,
                            "updates_per_block": updates,
                            "mean_rollout_ess": 1.0,
                            "mean_rollout_reward": 0.5,
                        },
                    }
                )
            records_path = directory / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "fingerprint": f"{arm}-{draw}",
                        "environment": {
                            "backend": "transformers",
                            "gpu": "fixture",
                        },
                        "effective": {
                            "implementation_sha256": {"fixture.py": "digest"}
                        },
                    }
                ),
                encoding="utf-8",
            )

    report = summarize_isir_screen(
        config_path=config,
        raw_root=raw,
        tag="study",
        draws=2,
        bootstrap_replicates=200,
    )
    assert report["scope"]["dllm_experiments"] is False
    assert [row["main_model_flops_factor_vs_n9_u1"] for row in report["table"]] == [
        1.0,
        1.0,
        1.0,
    ]
    assert report["table"][2]["paired_vs_n9_u1"]["accuracy_difference"] == -0.5

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.arllm.run_qwen15b_bounded_stop_screen import build_commands
from experiments.arllm.summarize_qwen15b_bounded_stop import (
    BOUNDED_STOP_ARMS,
    summarize_bounded_stop_study,
)


def test_bounded_stop_command_grid_is_counterbalanced(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=Path("configs/gsm8k_quick.toml"),
        tag="unit",
        phase="screen",
        draws=2,
        limit=8,
        rollout_count=4,
        evaluation_batch_size=2,
        log_weight_lower=0.0,
        log_weight_upper=10.0,
        raw_root=tmp_path / "raw",
        output=tmp_path / "summary.json",
    )

    commands = build_commands(args)

    assert len(commands) == 5
    arms = [command[command.index("--tag") + 1].split("-")[-2] for command in commands[:-1]]
    assert arms == ["full", "bounded", "bounded", "full"]
    for command in commands[:-1]:
        is_bounded = "bounded" in command[command.index("--tag") + 1]
        assert ("--exact-rollout-early-stop" in command) is is_bounded
        assert command[command.index("--conditional-reward") + 1] == "frozen_consensus"
    summary = commands[-1]
    assert summary[summary.index("--evaluation-batch-size") + 1] == "2"
    assert summary[summary.index("--log-weight-upper") + 1] == "10.0"


def test_bounded_stop_summary_requires_exact_output_and_records_savings(
    tmp_path: Path,
) -> None:
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
candidate_count = 2
rollout_count = 2
block_size = 8
reward_temperature = 0.1
[iterated_is]
pilot_samples = 2
""",
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    for arm, early_stop in BOUNDED_STOP_ARMS:
        for draw in range(2):
            directory = raw / "fixture" / f"conditional_is-study-{arm}-draw{draw}"
            directory.mkdir(parents=True)
            records = []
            for index in (10, 20):
                factor = 0.8 if early_stop else 1.0
                records.append(
                    {
                        "problem_index": index,
                        "draw_index": draw,
                        "correct": index == 10,
                        "output": f"tokens-{index}-{draw}",
                        "elapsed_seconds": 10.0 * factor,
                        "backend_delta": {
                            "estimated_dense_forward_flops": int(100 * factor),
                            "generated_tokens": int(50 * factor),
                        },
                        "diagnostics": {
                            "exact_rollout_early_stop_enabled": early_stop,
                            "reward_source": "frozen_consensus",
                            "configured_rollout_count": 4,
                            "uses_test_gold_oracle": False,
                            "rollout_evaluation_batch_size": 2 if early_stop else 1,
                            "declared_rollout_log_weight_bounds": (
                                [0.0, 10.0] if early_stop else None
                            ),
                            "selected_candidate_indices": [1, 0],
                            "candidate_token_ids_by_step": [
                                [[1], [2]],
                                [[3], [4]],
                            ],
                            "rollout_evaluations_planned": 20,
                            "rollout_evaluations_performed": 12 if early_stop else 20,
                            "rollout_evaluations_skipped": 8 if early_stop else 0,
                            "rollout_evaluation_batches": 4 if early_stop else 2,
                            "exact_early_stop_steps": 1 if early_stop else 0,
                            "selection_invariant_verified_steps": 1 if early_stop else 0,
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

    report = summarize_bounded_stop_study(
        config_path=config,
        raw_root=raw,
        tag="study",
        draws=2,
        questions=2,
        rollout_count=4,
        evaluation_batch_size=2,
        log_weight_lower=0.0,
        log_weight_upper=10.0,
        bootstrap_replicates=200,
    )

    assert report["scope"]["dllm_experiments"] is False
    assert report["decision"]["result"] == "advance_to_confirmation"
    bounded = report["table"][1]
    assert bounded["rollout_skip_fraction"] == pytest.approx(0.4)
    assert bounded["main_model_flops_factor_vs_full"] == pytest.approx(0.8)
    assert report["paired_exact_agreement"]["exact_output_match_fraction"] == 1.0
    assert report["paired_exact_agreement"]["selected_index_match_fraction"] == 1.0

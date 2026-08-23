from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.arllm.run_qwen15b_rqmc_screen import build_commands
from experiments.arllm.summarize_qwen15b_rqmc import (
    RQMC_ARMS,
    summarize_rqmc_study,
)


def test_rqmc_command_grid_is_counterbalanced_and_uses_fixed_reward(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        config=Path("configs/gsm8k_quick.toml"),
        tag="unit",
        phase="screen",
        draws=2,
        limit=8,
        rollout_count=4,
        raw_root=tmp_path / "raw",
        output=tmp_path / "summary.json",
    )

    commands = build_commands(args)

    assert len(commands) == 7
    designs = [
        command[command.index("--rollout-design") + 1] for command in commands[:-1]
    ]
    expected = [design for _, design in RQMC_ARMS]
    assert designs[:3] == expected
    assert designs[3:] == list(reversed(expected))
    assert all(
        command[command.index("--conditional-reward") + 1] == "frozen_consensus"
        for command in commands[:-1]
    )
    assert all(
        command[command.index("--rollout-count") + 1] == "4"
        for command in commands[:-1]
    )


def test_rqmc_summary_checks_paired_candidates_and_applies_gate(tmp_path: Path) -> None:
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
[iterated_is]
pilot_samples = 2
""",
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    for arm, design in RQMC_ARMS:
        for draw in range(2):
            directory = raw / "fixture" / f"conditional_is-study-{arm}-draw{draw}"
            directory.mkdir(parents=True)
            records = []
            for index in (10, 20):
                records.append(
                    {
                        "problem_index": index,
                        "draw_index": draw,
                        "correct": arm == "sobol" or index == 10,
                        "elapsed_seconds": 2.0,
                        "backend_delta": {
                            "estimated_dense_forward_flops": 100,
                            "generated_tokens": 50,
                        },
                        "diagnostics": {
                            "rollout_design": design,
                            "reward_source": "frozen_consensus",
                            "configured_rollout_count": 4,
                            "uses_test_gold_oracle": False,
                            "mean_rollout_ess": 3.0,
                            "mean_within_candidate_log_weight_dispersion": 0.25,
                            "candidate_token_ids_by_step": [[[1], [2]]],
                            "candidate_log_weight_estimates_by_step": [
                                [0.0, 1.0] if arm == "iid" else [0.1, 0.9]
                            ],
                            "selected_candidate_indices": [1],
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

    report = summarize_rqmc_study(
        config_path=config,
        raw_root=raw,
        tag="study",
        draws=2,
        questions=2,
        rollout_count=4,
        bootstrap_replicates=200,
    )

    assert report["scope"]["dllm_experiments"] is False
    assert report["decision"]["result"] == "advance_to_confirmation"
    sobol = report["table"][1]
    assert sobol["paired_vs_iid"]["accuracy_difference"] == pytest.approx(0.5)
    assert sobol["main_model_flops_factor_vs_iid"] == pytest.approx(1.0)
    paired = report["paired_first_step_weight_diagnostics"]["sobol"]
    assert paired["first_step_candidates_identical"] is True
    assert paired["first_step_selected_index_agreement"] == pytest.approx(1.0)
    assert paired["first_step_log_weight_mean_absolute_difference"] == pytest.approx(
        0.1
    )

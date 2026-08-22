from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.arllm.run_qwen15b_mh_suffix_screen import build_commands
from experiments.arllm.summarize_qwen15b_mh_suffix import (
    MH_SUFFIX_ARMS,
    summarize_mh_suffix_screen,
)


def test_mh_suffix_screen_command_grid_is_counterbalanced(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=Path("configs/gsm8k_quick.toml"),
        tag="unit",
        draws=2,
        limit=8,
        raw_root=tmp_path / "raw",
        output=tmp_path / "summary.json",
    )
    commands = build_commands(args)
    assert len(commands) == 7
    schedules = [
        command[command.index("--mh-suffix-schedule") + 1]
        for command in commands[:-1]
    ]
    expected = [schedule for _, schedule in MH_SUFFIX_ARMS]
    assert schedules[:3] == expected
    assert schedules[3:] == list(reversed(expected))
    assert all(
        command[command.index("--method") + 1] == "mh"
        for command in commands[:-1]
    )


def test_mh_suffix_summary_uses_weighted_diagnostics_and_gate(tmp_path: Path) -> None:
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
[mh]
alpha = 4.0
block_size = 8
steps_per_block = 2
""",
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    for arm, schedule in MH_SUFFIX_ARMS:
        for draw in range(2):
            directory = raw / "fixture" / f"mh-study-{arm}-draw{draw}"
            directory.mkdir(parents=True)
            records = []
            for index in (10, 20):
                factor = 1.0 if arm == "uniform" else 0.8
                records.append(
                    {
                        "problem_index": index,
                        "draw_index": draw,
                        "correct": True,
                        "elapsed_seconds": 10.0 * factor,
                        "backend_delta": {
                            "estimated_dense_forward_flops": int(100 * factor),
                            "generated_tokens": int(50 * factor),
                        },
                        "diagnostics": {
                            "suffix_schedule": schedule,
                            "attempts": 4,
                            "accepted": 2,
                            "mean_proposed_suffix_length": 8.0 * factor,
                            "mean_proposed_token_changes": 6.0 * factor,
                            "mean_accepted_token_changes": 3.0 * factor,
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
                        "environment": {"backend": "transformers", "gpu": "fixture"},
                        "effective": {
                            "implementation_sha256": {"fixture.py": "digest"}
                        },
                    }
                ),
                encoding="utf-8",
            )

    report = summarize_mh_suffix_screen(
        config_path=config,
        raw_root=raw,
        tag="study",
        draws=2,
        bootstrap_replicates=200,
    )
    assert report["scope"]["dllm_experiments"] is False
    assert report["decision"]["result"] == "advance_to_confirmation"
    assert report["decision"]["passing_arms"] == ["inverse", "multiscale"]
    inverse = report["table"][1]
    assert inverse["main_model_flops_factor_vs_uniform"] == pytest.approx(0.8)
    assert inverse["mean_proposed_suffix_length"] == pytest.approx(6.4)
    assert inverse["paired_vs_uniform"]["accuracy_difference"] == 0.0

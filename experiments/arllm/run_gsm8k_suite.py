"""Resumable orchestrator for the aligned GSM8K experiment matrix.

Every run is a method name plus the config fields it changes. ``--set`` fields
apply to every run that accepts them and a run's own fields take precedence;
``--components`` selects the experiment families beyond the quality methods.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from experiments.shared.config_overrides import add_config_override_argument
from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides, model_output_cli_arguments
from inference_scaling.arllm.backends import BACKEND_CHOICES
from experiments.shared.methods import AR_METHODS, DEFAULT_AR_METHOD

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = (DEFAULT_AR_METHOD,)
SUPPORTED_METHODS = AR_METHODS
SUITE_COMPONENTS = (
    "matched_target", "replay", "dynamic_is", "async", "passk", "ablations", "budget_curve", "length_ablation",
)
# Child scripts that accept --set.
SET_SCRIPTS = frozenset({"gsm8k_reproduction.py", "gsm8k_passk.py"})
_CLI_OVERRIDES: list[str] = []
_CONFIG_OVERRIDES: list[str] = []


def _run(command: list[str], environment: dict[str, str]) -> None:
    script = Path(command[1]).name
    if script in SET_SCRIPTS:
        command[2:2] = [part for value in _CONFIG_OVERRIDES for part in ("--set", value)]
    if not script.startswith(("summarize_", "plot_", "render_")):
        command[2:2] = _CLI_OVERRIDES
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--methods", nargs="*", choices=SUPPORTED_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--components", nargs="*", choices=SUITE_COMPONENTS, default=[])
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument("--verifier-config", type=Path)
    add_config_override_argument(parser)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ablation-limit", type=int, default=32)
    parser.add_argument("--passk-limit", type=int, default=32)
    parser.add_argument("--passk-draws", type=int, default=8)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("results"),
        help="directory for aggregate reports",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()
    global _CLI_OVERRIDES, _CONFIG_OVERRIDES
    _CLI_OVERRIDES = model_output_cli_arguments(args)
    _CONFIG_OVERRIDES = list(args.config_overrides)
    components = set(args.components)

    with args.config.open("rb") as source:
        loaded_config = tomllib.load(source)
    apply_model_output_overrides(loaded_config, args)
    configured_length = int(loaded_config["generation"]["max_new_tokens"])
    if args.profile == "smoke":
        mh_alphas = (2.0,)
        mh_steps = (1,)
        candidate_counts = (3,)
        rollout_counts = (1,)
        guidance_steps_values = (2,)
        # Batch-normalized rewards are accepted only by the archived block
        # conditional IS, which is also what produced the reported sweep.
        reward_methods = ("block_conditional_is",)
        temperatures = (0.7,)
        budget_beams = (4,)
        budget_samples = (4,)
        budget_candidates = (3,)
        generation_lengths = (32,)
    else:
        mh_alphas = (1.0, 2.0, 4.0, 8.0)
        mh_steps = (1, 2, 5, 10)
        candidate_counts = (3, 5, 10)
        rollout_counts = (1, 5)
        guidance_steps_values = (2, 8, 16)
        reward_methods = ("best_of_n", "block_conditional_is", "conditional_is_small_proposal")
        temperatures = (0.7, 1.5)
        budget_beams = (4, 8)
        budget_samples = (4, 8)
        budget_candidates = (3, 5, 10)
        generation_lengths = (128, 256, 512)
    reward_sources = (
        "log_probability", "sequence_log_probability", "negative_entropy", "self_certainty", "consilience", "verifier",
    )

    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT), existing or "")
    ).rstrip(os.pathsep)
    backend_args = [] if args.backend is None else ["--backend", args.backend]
    verifier_args = [] if args.verifier_config is None else ["--verifier-config", str(args.verifier_config)]
    rl_args = [] if args.rl_adapter is None else ["--rl-adapter", str(args.rl_adapter)]
    limit_args = [] if args.limit is None else ["--limit", str(args.limit)]
    common = ["--config", str(args.config), "--tag", args.tag, *verifier_args, *backend_args, *limit_args]
    summary_common = ["--config", str(args.config), "--tag", args.tag, *limit_args]

    def reproduce(method: str, tag: str, limit: int | None, fields: dict[str, Any] | None = None,
                  reward: str | None = None) -> None:
        """Run one method with the config fields that define this run."""
        command = [
            sys.executable, "experiments/arllm/gsm8k_reproduction.py", "--config", str(args.config),
            *backend_args, *rl_args, *verifier_args, "--method", method, "--tag", tag,
            *([] if limit is None else ["--limit", str(limit)]),
            *([] if reward is None else ["--reward", reward]),
        ]
        for key, value in (fields or {}).items():
            command.extend(("--set", f"{key}={value}"))
        _run(command, environment)

    for method in args.methods:
        reproduce(method, args.tag, args.limit)

    if "matched_target" in components:
        for method in ("verifier_mh", "verifier_conditional_is", "verifier_conditional_is_small_proposal"):
            if method not in args.methods:
                reproduce(method, args.tag, args.limit)

    if "replay" in components:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        replay_output = str(args.summary_root / f"{args.config.stem}_replay_{args.tag}.json")
        _run([sys.executable, "experiments/arllm/gsm8k_replay_benchmark.py", *common,
              "--aggregate-output", replay_output], environment)
        _run([sys.executable, "experiments/arllm/summarize_gsm8k_replay.py", *summary_common,
              "--output", replay_output], environment)
    if "dynamic_is" in components:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "experiments/arllm/gsm8k_dynamic_is_benchmark.py", *common, "--aggregate-output",
              str(args.summary_root / f"{args.config.stem}_dynamic_is_{args.tag}.json")], environment)
    if "async" in components:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        async_limit = args.limit if args.limit is not None else min(args.ablation_limit, 32)
        _run([sys.executable, "experiments/arllm/gsm8k_async_benchmark.py", "--config", str(args.config),
              *backend_args, "--limit", str(async_limit), "--output",
              str(args.summary_root / f"{args.config.stem}_async_grouped_{args.tag}.json")], environment)

    if "passk" in components:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        passk_args = ["--limit", str(args.passk_limit), "--draws", str(args.passk_draws)]
        _run([sys.executable, "experiments/arllm/gsm8k_passk.py", "--config", str(args.config),
              "--tag", f"{args.tag}-passk", *passk_args, "--output",
              str(args.summary_root / f"{args.config.stem}_passk_{args.tag}.json"),
              *backend_args, *rl_args, *verifier_args], environment)
        _run([sys.executable, "experiments/arllm/gsm8k_is_passk.py", "--config", str(args.config),
              "--tag", f"{args.tag}-is-passk", *passk_args, "--workers", str(min(args.passk_draws, 8)), "--output",
              str(args.summary_root / f"{args.config.stem}_is_passk_{args.tag}.json"),
              *backend_args, *verifier_args], environment)

    limit = args.ablation_limit
    reference = {"sampling.temperature": 1.0}
    if "ablations" in components:
        for alpha in mh_alphas:
            reproduce("mh", f"{args.tag}-alpha-{alpha:g}", limit, {"mh.alpha": alpha})
        for steps in mh_steps:
            reproduce("mh", f"{args.tag}-steps-{steps}", limit, {"mh.steps_per_block": steps})
        reproduce("conditional_is", f"{args.tag}-conditional-reference", limit, reference)
        for component_tag, clip in (
            ("conditional-small-proposal-reference", 10.0),
            ("conditional-small-proposal-unclipped", "none"),
        ):
            reproduce("conditional_is_small_proposal", f"{args.tag}-{component_tag}", limit,
                      {**reference, "conditional_is.importance_log_ratio_clip": clip})
        reproduce("best_of_n", f"{args.tag}-best-of-n-reference", limit, reference)
        for candidates in candidate_counts:
            reproduce("conditional_is", f"{args.tag}-candidates-{candidates}-rollouts-3", limit,
                      {"conditional_is.candidate_count": candidates, "conditional_is.rollout_count": 3})
        for rollouts in rollout_counts:
            reproduce("conditional_is", f"{args.tag}-candidates-10-rollouts-{rollouts}", limit,
                      {"conditional_is.candidate_count": 10, "conditional_is.rollout_count": rollouts})
        for guidance_steps in guidance_steps_values:
            reproduce("conditional_is", f"{args.tag}-guidance-steps-{guidance_steps}", limit,
                      {"conditional_is.block_size": configured_length // guidance_steps})
        for reward_source in reward_sources:
            for method in reward_methods:
                reproduce(method, f"{args.tag}-{method}-reward-{reward_source}", limit, reference, reward_source)
        for temperature in temperatures:
            for method in ("beam", "best_of_n", "conditional_is", "conditional_is_small_proposal"):
                reproduce(method, f"{args.tag}-{method}-temperature-{temperature:g}", limit,
                          {"sampling.temperature": temperature})

    if "budget_curve" in components:
        reproduce("beam", f"{args.tag}-beam-reference", limit)
        reproduce("best_of_n", f"{args.tag}-best-of-n-reference", limit, reference)
        reproduce("conditional_is", f"{args.tag}-conditional-reference", limit, reference)
        reproduce("conditional_is_small_proposal", f"{args.tag}-conditional-small-proposal-reference", limit,
                  reference)
        for beams in budget_beams:
            reproduce("beam", f"{args.tag}-budget-beam-{beams}", limit, {"beam.num_beams": beams})
        for samples in budget_samples:
            reproduce("best_of_n", f"{args.tag}-budget-best-of-n-{samples}", limit, {"best_of_n.samples": samples})
        for candidates in budget_candidates:
            for method in ("conditional_is", "conditional_is_small_proposal"):
                reproduce(method, f"{args.tag}-budget-{method}-m{candidates}-k3", limit,
                          {"conditional_is.candidate_count": candidates, "conditional_is.rollout_count": 3})

    if "length_ablation" in components:
        for length in generation_lengths:
            for method in ("base", "best_of_n", "beam", "conditional_is", "conditional_is_small_proposal", "rl_greedy"):
                fields: dict[str, Any] = {"generation.max_new_tokens": length}
                if method.startswith("conditional_is"):
                    fields["conditional_is.block_size"] = max(1, length // 4)
                reproduce(method, f"{args.tag}-length-{length}", limit, fields)


if __name__ == "__main__":
    main()

"""Single-prompt joint-budget IS using the shared model and reward loaders."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import tomllib

from experiments.shared.artifacts import dataclass_snapshot_delta
from experiments.shared.model_cli import (
    add_model_output_arguments,
    apply_model_output_overrides,
    require_full_scope,
)
from inference_scaling.arllm.backends.loader import (
    BACKEND_CHOICES,
    close_backend,
    load_backend_from_config,
    set_backend_override,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.reward_factory import (
    MODEL_REWARD_SOURCES,
    model_reward_from_config,
    reward_temperature_from_config,
)
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.experimental.arllm.joint_budget_is import (
    JointBudgetISConfig,
    run_joint_budget_is,
)
from inference_scaling.shared.generation import generation_config_for_prompt
from inference_scaling.shared.prompting import render_prompt
from inference_scaling.shared.rng import SeedStream


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="optional existing model/runtime TOML"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--budget-forward-tokens",
        type=int,
        required=True,
        help="reserved forward-token slots, including pilot and reward scoring",
    )
    parser.add_argument("--block-sizes", nargs="+", type=int, default=(64, 128, 256))
    parser.add_argument(
        "--candidate-counts", nargs="+", type=int, default=(2, 4, 8, 16)
    )
    parser.add_argument("--rollout-counts", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--pilot-candidates", type=int, default=2)
    parser.add_argument("--pilot-rollouts", type=int, default=2)
    parser.add_argument("--pilot-fraction", type=float, default=0.15)
    parser.add_argument("--planning-mode", choices=("full_horizon", "chunk_adaptive"), default="full_horizon")
    parser.add_argument("--initial-block-size", type=int)
    parser.add_argument("--initial-candidate-count", type=int)
    parser.add_argument("--initial-rollout-count", type=int)
    parser.add_argument("--adjustment-min-improvement", type=float, default=0.1)
    parser.add_argument("--reward", choices=MODEL_REWARD_SOURCES, default="consilience")
    parser.add_argument("--reward-temperature", type=float)
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="JSON destination; default: stdout")
    add_model_output_arguments(parser, exclude=("--proposal-model", "--mh-iterations"))
    return parser


def run(args):
    config = (
        tomllib.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    )
    apply_model_output_overrides(config, args)
    set_backend_override(config, args.backend)
    require_full_scope(config, "joint-budget IS")
    if "base" not in config.get("models", {}):
        raise ValueError("provide --model or models.base in --config")
    config.setdefault("runtime", {}).setdefault("dtype", "bfloat16")
    temperature = args.reward_temperature
    if temperature is None:
        temperature = reward_temperature_from_config(config, source=args.reward)
    # Validate scheduler settings before loading weights.
    options = dict(
        forward_token_budget=args.budget_forward_tokens,
        block_sizes=tuple(args.block_sizes),
        candidate_counts=tuple(args.candidate_counts),
        rollout_counts=tuple(args.rollout_counts),
        pilot_candidates=args.pilot_candidates,
        pilot_rollouts=args.pilot_rollouts,
        pilot_fraction=args.pilot_fraction,
        reward_temperature=temperature,
        reward_forward_passes=1,
        planning_mode=args.planning_mode,
        initial_block_size=args.initial_block_size,
        initial_candidate_count=args.initial_candidate_count,
        initial_rollout_count=args.initial_rollout_count,
        adjustment_min_improvement=args.adjustment_min_improvement,
    )
    JointBudgetISConfig(**options)
    backend = load_backend_from_config(config["models"]["base"], config, role="base")
    try:
        text = render_prompt(
            backend.tokenizer, [{"role": "user", "content": args.prompt}], config
        )
        prompt = backend.encode(text, add_special_tokens=False)
        config, generation = generation_config_for_prompt(
            config, len(prompt), [backend]
        )
        sampling = SamplingConfig(
            temperature=float(config.get("sampling", {}).get("temperature", 1.0)),
            top_p=float(config.get("sampling", {}).get("top_p", 1.0)),
            top_k=config.get("sampling", {}).get("top_k"),
            eos_token_id=backend.tokenizer.eos_token_id,
        )
        reward = model_reward_from_config(
            backend, config, source=args.reward, sampling=sampling
        )
        settings = JointBudgetISConfig(
            total_length=generation["effective_max_new_tokens"], **options
        )
        before = backend.snapshot()
        started = time.perf_counter()
        result = run_joint_budget_is(
            backend, prompt, settings, reward, SeedStream(args.seed), sampling=sampling
        )
        seconds = time.perf_counter() - started
        cost = dataclass_snapshot_delta(
            before,
            backend.snapshot(),
            constant_fields=tuple(
                name
                for name in (
                    "mh_fused_logprobs",
                    "native_suffix_speculation",
                    "maximum_in_flight_requests",
                )
                if hasattr(before, name)
            ),
        )
        steps = [
            {
                "prefix_length": step.evaluation.generated_length_before,
                "plan": asdict(step.plan),
                "estimates": [asdict(item) for item in step.estimates],
                "pilot_reserved_forward_tokens": step.pilot_reserved_cost,
                "remaining_budget": step.remaining_budget,
                "selected_index": step.evaluation.selected_index,
                **({"adjustment": step.adjustment} if step.adjustment is not None else {}),
            }
            for step in result.steps
        ]
        metadata = SamplingScope.from_config(backend, config).describe_output(
            backend, prompt, result.token_ids
        )
        callback = getattr(reward, "describe_completion", None)
        if callback is not None:
            metadata.update(callback(prompt, result.token_ids))
        return {
            "method": "joint_budget_is",
            "seed": args.seed,
            "config": asdict(settings),
            "reward": reward.describe(),
            "generation_budget": generation,
            "text": backend.decode(result.token_ids, skip_special_tokens=False),
            "output_segments": metadata,
            "stopping_reason": result.stopping_reason,
            "reserved_forward_tokens": result.reserved_forward_tokens,
            "pilot_reserved_forward_tokens": result.pilot_reserved_forward_tokens,
            "actual_backend_cost": cost,
            "inference_seconds": seconds,
            "steps": steps,
        }
    finally:
        close_backend(backend)


def main():
    args = build_parser().parse_args()
    payload = json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

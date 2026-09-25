"""GRPO LoRA training on the GSM8K training split, with a resumable cost record.

The prompt is the GSM8K dataset prompt and the reward is the configured
verifier, as in inference. Wall time, generated rollout tokens, peak CUDA
memory and the sampled GPU power integral are recorded so training can be
compared with the per-query cost of inference-time scaling.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inference_scaling.app.records import (
    cached_file_sha256,
    checkpoint_metadata_hashes,
    environment,
    git_state,
    json_sha256,
    source_sha256,
    write_json_atomic,
)
from inference_scaling.datasets.base import Problem
from inference_scaling.datasets.gsm8k import GSM8K
from inference_scaling.shared.compute import dense_forward_flops
from inference_scaling.shared.model.loading import checkpoint_weight_files, resolve_checkpoint_path
from inference_scaling.shared.rewards.verifier import Verifier, VerifierContext, build_verifier


@dataclass
class PowerMonitor:
    """Sample ``nvidia-smi`` in a thread and integrate the GPU power draw."""

    interval_seconds: float
    samples: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _started: float = field(default=0.0, init=False)

    def _sample(self) -> None:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits", "--id=0"],
                check=True, capture_output=True, text=True, timeout=max(1.0, self.interval_seconds),
            ).stdout.strip().splitlines()[0]
            power, utilization, memory, temperature = (float(value.strip()) for value in output.split(","))
            self.samples.append({"seconds": time.perf_counter() - self._started, "power_watts": power,
                                 "utilization_percent": utilization, "memory_mib": memory,
                                 "temperature_c": temperature})
        except Exception as caught:  # monitoring must never abort training
            self.error = f"{type(caught).__name__}: {caught}"

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2.0)
        self._sample()
        pairs = zip(self.samples, self.samples[1:])
        integral = sum((right["seconds"] - left["seconds"]) / 3600.0 * (left["power_watts"] + right["power_watts"]) / 2
                       for left, right in pairs)

        def reduce(name: str, reducer: Any) -> float | None:
            values = [sample[name] for sample in self.samples]
            return reducer(values) if values else None

        return {
            "sample_interval_seconds": self.interval_seconds,
            "samples": len(self.samples),
            "gpu_power_integral_wh": integral,
            "mean_power_watts": reduce("power_watts", lambda values: sum(values) / len(values)),
            "peak_power_watts": reduce("power_watts", max),
            "mean_utilization_percent": reduce("utilization_percent", lambda values: sum(values) / len(values)),
            "peak_nvidia_smi_memory_mib": reduce("memory_mib", max),
            "peak_temperature_c": reduce("temperature_c", max),
            "monitor_error": self.error,
        }


def _text(value: object) -> str:
    """Plain or conversational TRL values as verifier text."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value["content"])
    if isinstance(value, Sequence):
        return "\n".join(_text(item) for item in value)
    raise TypeError(f"unsupported training text value: {type(value).__name__}")


class VerifierReward:
    """The configured verifier as a TRL batched reward function.

    Rows carry the reference answer (``reference``) and problem id, so the
    ``dataset`` source grades each completion as inference does.
    """

    def __init__(self, settings: Mapping[str, Any], dataset: GSM8K) -> None:
        self.settings = settings
        self.dataset = dataset
        self._verifiers: dict[tuple[str, str], Verifier] = {}
        self.calls = self.completions = self.completion_tokens = 0
        self.reward_sum = 0.0
        self.reward_minimum: float | None = None
        self.reward_maximum: float | None = None

    def _verifier(self, prompt: str, reference: str, problem_id: str) -> Verifier:
        key = (prompt, reference)
        if key not in self._verifiers:
            problem = Problem(problem_id, prompt, reference)
            self._verifiers[key] = build_verifier(
                self.settings, context=VerifierContext(prompt, reference, {"problem_id": problem_id}),
                grade=lambda text: self.dataset.grade(text, problem),
            )
        return self._verifiers[key]

    def __call__(self, prompts: Sequence[object], completions: Sequence[object],
                 completion_ids: Sequence[Sequence[int]] | None = None, *, reference: Sequence[str],
                 problem_id: Sequence[str], **_: object) -> list[float]:
        rewards = []
        for prompt, completion, answer, identifier in zip(prompts, completions, reference, problem_id, strict=True):
            text = _text(prompt)
            rewards.append(self._verifier(text, str(answer), str(identifier)).score(text, _text(completion)))
        self.calls += 1
        self.completions += len(rewards)
        self.completion_tokens += sum(len(tokens) for tokens in completion_ids or ())
        if rewards:
            self.reward_sum += sum(rewards)
            self.reward_minimum = min(rewards + ([] if self.reward_minimum is None else [self.reward_minimum]))
            self.reward_maximum = max(rewards + ([] if self.reward_maximum is None else [self.reward_maximum]))
        return rewards

    def snapshot(self, num_generations: int) -> dict[str, Any]:
        return {"reward_calls": self.calls, "generated_completions": self.completions,
                "generated_prompt_groups": self.completions // num_generations,
                "generated_completion_tokens": self.completion_tokens, "reward_sum": self.reward_sum,
                "observed_minimum_reward": self.reward_minimum, "observed_maximum_reward": self.reward_maximum}


def _latest_checkpoint(output: Path) -> Path | None:
    checkpoints = [(int(path.name.rsplit("-", 1)[1]), path) for path in output.glob("checkpoint-*")
                   if path.is_dir() and path.name.rsplit("-", 1)[1].isdigit()]
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _combine(previous: Mapping[str, Any], segment: Mapping[str, Any]) -> dict[str, Any]:
    """Rollout totals across resumed segments."""

    total: dict[str, Any] = {key: segment[key] + previous.get(key, 0) for key in (
        "reward_calls", "generated_completions", "generated_prompt_groups", "generated_completion_tokens",
        "reward_sum")}
    for name, reducer in (("observed_minimum_reward", min), ("observed_maximum_reward", max)):
        values = [value for value in (previous.get(name), segment[name]) if value is not None]
        total[name] = reducer(values) if values else None
    total["observed_mean_reward"] = (total["reward_sum"] / total["generated_completions"]
                                     if total["generated_completions"] else None)
    return total


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value.item() if hasattr(value, "item") else str(value)


def run(settings: Mapping[str, Any]) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    grpo = settings["grpo"]
    model, trainer_settings = grpo["model"], dict(grpo["trainer"])
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires CUDA")
    base = resolve_checkpoint_path(str(model["path"]), revision=model["revision"], cache_dir=model["cache_dir"],
                                   local_files_only=model["local_files_only"])
    cache_dir = Path(str(settings["hash_cache_dir"]))
    weights = {path.relative_to(base).as_posix(): cached_file_sha256(path, cache_dir=cache_dir)
               for path in checkpoint_weight_files(base)}
    digest = next(iter(weights.values())) if len(weights) == 1 else json_sha256(weights)
    if model["weight_sha256"] is not None and digest != model["weight_sha256"]:
        raise ValueError(f"weights of {model['path']} hash to {digest}, not {model['weight_sha256']}")

    train, test = GSM8K(settings["gsm8k"]["train"]), GSM8K(settings["gsm8k"]["test"])
    overlap = {problem.question for problem in train.problems} & {problem.question for problem in test.problems}
    if overlap:
        raise ValueError(f"GSM8K train/test leakage: {len(overlap)} identical questions")
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": train.prompt(problem)}], "reference": problem.answer,
         "problem_id": problem.id}
        for problem in train.problems
    ])

    output = Path(str(grpo["output"]))
    output.mkdir(parents=True, exist_ok=True)
    effective = {"grpo": grpo, "train": train.describe(), "test_sha256": test.source_sha256,
                 "weight_sha256": weights, "metadata_sha256": checkpoint_metadata_hashes(base),
                 "source_sha256": json_sha256(source_sha256())}
    # Extending the step count or checkpoint cadence resumes the same run.
    resumable = json.loads(json.dumps(effective))
    for key in ("max_steps", "save_steps", "logging_steps"):
        resumable["grpo"]["trainer"].pop(key, None)
    fingerprint = json_sha256(resumable)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError(f"{output} holds a run with other data or hyperparameters; choose another grpo.output")
    manifest = {"fingerprint": fingerprint, "status": "initializing", "effective": effective,
                "overlapping_questions": len(overlap), "git": git_state(), "environment": environment()}
    write_json_atomic(manifest_path, manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model["tokenizer"] or base), revision=model["tokenizer_revision"], cache_dir=model["cache_dir"],
        local_files_only=model["local_files_only"], trust_remote_code=model["trust_remote_code"],
        **model["tokenizer_kwargs"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = GRPOConfig(output_dir=str(output), **trainer_settings, model_init_kwargs={
        **model["model_kwargs"], "local_files_only": True, "trust_remote_code": model["trust_remote_code"]})
    reward = VerifierReward(grpo["verifier"], train)
    started = time.perf_counter()
    trainer = GRPOTrainer(model=str(base), reward_funcs=reward, args=config, train_dataset=dataset,
                          processing_class=tokenizer,
                          peft_config=LoraConfig(**grpo["lora"], task_type=TaskType.CAUSAL_LM))
    initialization_seconds = time.perf_counter() - started
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())

    checkpoint = _latest_checkpoint(output) if grpo["resume"] else None
    cost_path = output / "training_cost.json"
    previous = json.loads(cost_path.read_text(encoding="utf-8")) if checkpoint and cost_path.is_file() else {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = PowerMonitor(float(grpo["power_sample_seconds"]))
    monitor.start()
    manifest.update(status="training", resume_from_checkpoint=None if checkpoint is None else str(checkpoint))
    write_json_atomic(manifest_path, manifest)
    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=None if checkpoint is None else str(checkpoint))
    except BaseException:
        manifest.update(status="interrupted", elapsed_before_interruption_seconds=time.perf_counter() - started)
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        power = monitor.stop()
    seconds = time.perf_counter() - started

    trainer.save_model(str(output))
    trainer.save_state()
    tokenizer.save_pretrained(output)
    state = trainer.state
    generations = int(trainer_settings["num_generations"])
    rollouts = _combine(previous.get("rollouts", {}), reward.snapshot(generations))
    compute_options = {
        "generated_completions": rollouts["generated_completions"], "total_parameters": total,
        "trainable_parameters": trainable, "optimizer_steps": int(state.global_step),
        "gradient_checkpointing": bool(config.gradient_checkpointing), "reference_scoring": config.beta != 0,
    }
    model_tokens = max((int(entry["num_tokens"]) for entry in state.log_history if "num_tokens" in entry), default=0)
    compute: dict[str, Any] | None
    try:
        compute = estimate_grpo_compute_from_logs(
            log_history=state.log_history,
            sequences_per_optimizer_step=config.per_device_train_batch_size * config.gradient_accumulation_steps,
            **compute_options,
        ).as_dict()
    except ValueError:
        compute = (estimate_grpo_compute(model_sequence_tokens=model_tokens, **compute_options).as_dict()
                   if model_tokens >= rollouts["generated_completions"] else None)
    write_json_atomic(cost_path, {
        "status": "complete",
        "fingerprint": fingerprint,
        "initialization_seconds_excluded_from_training_cost": initialization_seconds,
        "training_wall_seconds": seconds,
        "cumulative_training_wall_seconds": float(previous.get("cumulative_training_wall_seconds", 0.0)) + seconds,
        "cumulative_gpu_power_integral_wh": float(previous.get("cumulative_gpu_power_integral_wh", 0.0))
        + float(power["gpu_power_integral_wh"]),
        "global_step": state.global_step,
        "epoch": state.epoch,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trainer_metrics": _jsonable(result.metrics),
        "trainer_reported_model_tokens": model_tokens,
        "primary_compute": compute,
        "rollouts": rollouts,
        "gpu_monitor": power,
        "resume_from_checkpoint": None if checkpoint is None else str(checkpoint),
    })
    manifest.update(status="complete", global_step=state.global_step)
    write_json_atomic(manifest_path, manifest)
    print(f"GRPO complete: {output} (step {state.global_step})", flush=True)


@dataclass(frozen=True, slots=True)
class GRPOComputeEstimate:
    trainer_observed_prompt_plus_completion_tokens: int
    generated_completions: int
    rollout_generation_forward_token_slots: int
    reference_scoring_forward_token_slots: int
    policy_forward_backward_equivalent_token_slots: int
    total_forward_equivalent_token_slots: int
    total_parameters: int
    trainable_parameters: int
    optimizer_steps: int
    estimated_dense_model_flops: int
    estimated_optimizer_flops: int
    estimated_total_flops: int
    estimated_total_petaflops: float
    accounting_basis: str
    definition: str
    exclusions: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _grpo_estimate(
    *, observed_tokens: int, generated_completions: int, rollout_slots: int, sequence_slots: int,
    total_parameters: int, trainable_parameters: int, optimizer_steps: int, gradient_checkpointing: bool,
    reference_scoring: bool, accounting_basis: str, definition: str,
) -> GRPOComputeEstimate:
    """Reference scoring is one forward over the sequences; a frozen-base LoRA update two, three with checkpointing."""

    reference_slots = sequence_slots if reference_scoring else 0
    policy_slots = (3 if gradient_checkpointing else 2) * sequence_slots
    total_slots = rollout_slots + reference_slots + policy_slots
    dense_flops = dense_forward_flops(total_parameters, total_slots)
    optimizer_flops = 10 * trainable_parameters * optimizer_steps
    return GRPOComputeEstimate(
        observed_tokens, generated_completions, rollout_slots, reference_slots, policy_slots, total_slots,
        total_parameters, trainable_parameters, optimizer_steps, dense_flops, optimizer_flops,
        dense_flops + optimizer_flops, (dense_flops + optimizer_flops) / 1e15, accounting_basis, definition,
        "quadratic attention, elementwise kernels, reward parsing, data loading, tokenization, sampling, and host work",
    )


def estimate_grpo_compute(
    *,
    model_sequence_tokens: int,
    generated_completions: int,
    total_parameters: int,
    trainable_parameters: int,
    optimizer_steps: int,
    gradient_checkpointing: bool,
    reference_scoring: bool,
) -> GRPOComputeEstimate:
    """Estimate this LoRA GRPO run from observed model-token counts.

    For a frozen dense base, a policy forward/backward pass is approximately
    two forward-equivalent dense passes. Gradient-checkpoint recomputation adds
    one more. The adapter overhead is included conservatively through the total
    parameter count. AdamW is charged only to trainable parameters.
    """

    if any(value < 0 for value in (model_sequence_tokens, generated_completions, total_parameters,
                                   trainable_parameters, optimizer_steps)):
        raise ValueError("GRPO compute inputs must be non-negative")
    if generated_completions > model_sequence_tokens:
        raise ValueError("each generated completion requires at least one model token")
    return _grpo_estimate(
        observed_tokens=model_sequence_tokens, generated_completions=generated_completions,
        rollout_slots=model_sequence_tokens - generated_completions, sequence_slots=model_sequence_tokens,
        total_parameters=total_parameters, trainable_parameters=trainable_parameters,
        optimizer_steps=optimizer_steps, gradient_checkpointing=gradient_checkpointing,
        reference_scoring=reference_scoring,
        accounting_basis=("non-padding trainer token count; use estimate_grpo_compute_from_logs "
                          "when per-step mean and maximum completion lengths are available"),
        definition=("rollout generation uses observed prompt+completion tokens minus one input slot per "
                    "completion; reference scoring uses one forward pass when beta is nonzero; a frozen-base LoRA "
                    "policy update uses two forward-equivalent passes plus one more when gradient checkpointing "
                    "recomputes activations; dominant dense FLOPs are 2 * total parameters * forward-equivalent "
                    "token slots; AdamW is 10 * trainable parameters * optimizer steps"),
    )


def estimate_grpo_compute_from_logs(
    *,
    log_history: Sequence[dict[str, Any]],
    sequences_per_optimizer_step: int,
    generated_completions: int,
    total_parameters: int,
    trainable_parameters: int,
    optimizer_steps: int,
    gradient_checkpointing: bool,
    reference_scoring: bool,
) -> GRPOComputeEstimate:
    """Reconstruct padded GRPO token slots from trainer step metrics.

    TRL reports the cumulative non-padding model tokens, the batch mean
    completion length, and the mean of each microbatch's maximum completion
    length. Since every optimizer step has a fixed number of sequences, these
    values recover the prompt tokens and the padded prompt+completion tensor
    shapes used by generation, reference scoring, and policy training.
    """

    if sequences_per_optimizer_step <= 0:
        raise ValueError("sequences_per_optimizer_step must be positive")
    if generated_completions != sequences_per_optimizer_step * optimizer_steps:
        raise ValueError("generated completion count does not match batch size times optimizer steps")
    keys = ("num_tokens", "completions/mean_length", "completions/max_length", "step")
    step_logs = sorted({int(entry["step"]): entry for entry in log_history
                        if all(key in entry for key in keys)}.values(), key=lambda entry: int(entry["step"]))
    if not step_logs:
        raise ValueError("trainer log history has no complete GRPO step metrics")
    prior_tokens = prior_step = 0
    rollout_slots = sequence_slots = 0.0
    for entry in step_logs:
        current_step = int(entry["step"])
        if current_step <= prior_step:
            raise ValueError("trainer steps are not strictly increasing")
        sequences = sequences_per_optimizer_step * (current_step - prior_step)
        prior_step = current_step
        cumulative_tokens = int(entry["num_tokens"])
        if cumulative_tokens < prior_tokens:
            raise ValueError("trainer num_tokens is not cumulative")
        observed_step_tokens, prior_tokens = cumulative_tokens - prior_tokens, cumulative_tokens
        mean_completion = float(entry["completions/mean_length"])
        maximum_completion = float(entry["completions/max_length"])
        if maximum_completion < mean_completion or mean_completion < 0:
            raise ValueError("invalid trainer completion-length metrics")
        prompt_tokens = observed_step_tokens - mean_completion * sequences
        if prompt_tokens < -1e-6:
            raise ValueError("completion metrics exceed trainer-observed model tokens")
        prompt_tokens = max(0.0, prompt_tokens)
        padded_completion_tokens = maximum_completion * sequences
        sequence_slots += prompt_tokens + padded_completion_tokens
        rollout_slots += prompt_tokens + max(0.0, padded_completion_tokens - sequences)
    if prior_step != optimizer_steps:
        raise ValueError("trainer log history does not cover the requested optimizer-step count")
    return _grpo_estimate(
        observed_tokens=prior_tokens, generated_completions=generated_completions,
        rollout_slots=round(rollout_slots), sequence_slots=round(sequence_slots),
        total_parameters=total_parameters, trainable_parameters=trainable_parameters,
        optimizer_steps=optimizer_steps, gradient_checkpointing=gradient_checkpointing,
        reference_scoring=reference_scoring,
        accounting_basis=("padded forward token slots reconstructed per optimizer step from cumulative num_tokens, "
                          "mean completion length, and microbatch maximum completion length"),
        definition=("generation counts repeated prompts and every padded decode row except the final generated "
                    "token; reference scoring counts each padded full sequence; a frozen-base LoRA policy update "
                    "uses two forward-equivalent passes plus one gradient-checkpoint recomputation; dominant dense "
                    "FLOPs are 2 * total parameters * reconstructed token slots; AdamW is 10 * trainable parameters "
                    "* optimizer steps"),
    )


__all__ = ["GRPOComputeEstimate", "PowerMonitor", "VerifierReward", "estimate_grpo_compute",
           "estimate_grpo_compute_from_logs", "run"]

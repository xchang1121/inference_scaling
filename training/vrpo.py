"""VRPO for LLaDA: verifier-scored preference pairs, then a LoRA adapter.

``preferences`` samples candidates from the base model on GSM8K training
problems and keeps the highest- and lowest-reward distinct completions (the
reference solution may be scored as one more completion). ``train`` optimizes
the variance-reduced masked-diffusion preference loss against the frozen base.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from inference_scaling.app.records import (
    checkpoint_metadata_hashes,
    json_sha256,
    load_jsonl,
    snapshot_delta,
    source_sha256,
    write_json_atomic,
)
from inference_scaling.datasets.gsm8k import GSM8K
from inference_scaling.app.dllm import pinned_weight_hashes
from inference_scaling.dllm.backends.llada import active_parameter_counts
from inference_scaling.dllm.backends.loader import load_llada_backend
from inference_scaling.dllm.config import VRPOSamplingConfig, sampling_from_settings
from inference_scaling.dllm.training.preferences import select_scored_preference_pair
from inference_scaling.dllm.training.vrpo import (
    AdapterDisabledReference,
    estimate_vrpo_preference_loss,
    vrpo_forward_token_slots,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.rewards.verifier import VerifierContext, build_verifier
from inference_scaling.shared.model.loading import release_accelerator_memory
from inference_scaling.shared.rng import SeedStream


def preferences(settings: Mapping[str, Any]) -> None:
    vrpo = settings["vrpo"]
    options = vrpo["preferences"]
    dataset = GSM8K({**settings["gsm8k"]["train"], "selection": options["selection"]})
    data_path, manifest_path = Path(str(options["data"])), Path(str(options["manifest"]))
    effective = {"vrpo": {key: vrpo[key] for key in ("model", "engine", "prompt", "sampling", "max_new_tokens",
                                                     "preferences", "verifier")},
                 "train": dataset.describe(), "weight_sha256": pinned_weight_hashes(vrpo["model"], Path(str(settings["hash_cache_dir"]))),
                 "metadata_sha256": checkpoint_metadata_hashes(Path(str(vrpo["model"]["path"]))),
                 "source_sha256": json_sha256(source_sha256())}
    fingerprint = json_sha256(effective)
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError(f"{manifest_path} belongs to other preference settings")
    else:
        write_json_atomic(manifest_path, {"fingerprint": fingerprint, "status": "running", **effective})
    records = load_jsonl(data_path)
    if any(record["fingerprint"] != fingerprint for record in records):
        raise ValueError(f"{data_path} holds records of other preference settings")
    done = {record["problem_id"] for record in records}
    pairs, wanted = sum(record["status"] == "pair" for record in records), int(options["pairs"])
    backend = load_llada_backend({**vrpo["model"], "adapter": None}, vrpo["engine"])
    sampling = sampling_from_settings(vrpo["sampling"])
    maximum = int(vrpo["max_new_tokens"])
    length = maximum - maximum % sampling.block_length
    seeds = SeedStream(int(options["seed"]))
    reference_included = bool(options["include_reference_completion"])
    before, started = backend.snapshot(), time.perf_counter()
    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with data_path.open("a", encoding="utf-8") as sink:
            for problem in dataset.problems:
                if pairs >= wanted:
                    break
                if problem.id in done:
                    continue
                prompt_text = dataset.prompt(problem)
                prompt = backend.encode_chat(prompt_text, system_text=vrpo["prompt"]["system"])
                samples = backend.sample_batch([
                    DiffusionGenerationRequest(prefix=prompt, generation_length=length, sampling=sampling,
                                               seed=seeds.derive("vrpo-preference", problem.id, draw),
                                               request_id=f"vrpo-preference:{problem.id}:{draw}", stop_at_eos=True)
                    for draw in range(int(options["num_generations"]))
                ])
                texts = [backend.decode(sample.token_ids) for sample in samples]
                verifier = build_verifier(vrpo["verifier"], context=VerifierContext(prompt_text, problem.answer),
                                          grade=partial(dataset.grade, problem=problem))
                solution = str(problem.metadata["solution"])
                rewards = verifier.score_batch(prompt_text, [*texts, solution] if reference_included else texts)
                pair = select_scored_preference_pair(
                    candidate_texts=texts, candidate_rewards=rewards[:len(texts)],
                    reference_text=solution if reference_included else None,
                    reference_reward=rewards[-1] if reference_included else None,
                )
                grades = [dataset.grade(text, problem) for text in texts]
                record: dict[str, Any] = {
                    "fingerprint": fingerprint, "problem_id": problem.id, "question": problem.question,
                    "reference": problem.answer,
                    "reference_completion_reward": rewards[-1] if reference_included else None,
                    "candidates": [{"text": text, "answer": grade.answer, "correct": grade.correct, "reward": reward}
                                   for text, grade, reward in zip(texts, grades, rewards[:len(texts)], strict=True)],
                    "status": "skipped_equal_rewards",
                }
                if pair is not None:
                    record.update(status="pair", prompt=prompt_text, chosen=pair.chosen, rejected=pair.rejected,
                                  chosen_source=pair.chosen_source, rejected_source=pair.rejected_source)
                    pairs += 1
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                done.add(problem.id)
                print(f"preferences problems={len(done)} pairs={pairs}/{wanted}", flush=True)
        if pairs < wanted:
            raise RuntimeError(f"the candidate pool yielded {pairs} pairs, fewer than the {wanted} requested")
        write_json_atomic(manifest_path, {
            "fingerprint": fingerprint, "status": "complete", **effective, "records": len(done),
            "preference_pairs": pairs, "elapsed_seconds": time.perf_counter() - started,
            "generation_compute": snapshot_delta(before, backend.snapshot()),
        })
    finally:
        del backend
        release_accelerator_memory()


def train(settings: Mapping[str, Any]) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    vrpo = settings["vrpo"]
    options, model_settings, engine = vrpo["training"], vrpo["model"], vrpo["engine"]
    device = str(engine["device"])
    steps, accumulation = int(options["max_steps"]), int(options["gradient_accumulation_steps"])
    manifest = json.loads(Path(str(vrpo["preferences"]["manifest"])).read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("the VRPO preference data is incomplete; run the vrpo_preferences stage")
    pairs = [record for record in load_jsonl(Path(str(vrpo["preferences"]["data"]))) if record["status"] == "pair"]
    output = Path(str(options["output"]))
    effective = {"training": {key: value for key, value in options.items() if key not in {"max_steps", "save_steps"}},
                 "lora": vrpo["lora"], "prompt": vrpo["prompt"], "max_new_tokens": vrpo["max_new_tokens"],
                 "preference_fingerprint": manifest["fingerprint"],
                 "metadata_sha256": checkpoint_metadata_hashes(Path(str(model_settings["path"]))),
                 "source_sha256": json_sha256(source_sha256())}
    fingerprint = json_sha256(effective)
    state_path, optimizer_path = output / "training_state.json", output / "optimizer.pt"
    start, metrics, elapsed_before = 0, [], 0.0
    slots = {"current_policy": 0, "reference_policy": 0, "total": 0}
    if options["resume"] and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError(f"{output} holds a VRPO run with other settings")
        start, metrics, elapsed_before = int(state["completed_updates"]), list(state["metrics"]), float(state["elapsed_seconds"])
        slots = dict(state["forward_token_slots"])
        if start >= steps and (output / "training_cost.json").is_file():
            print(f"VRPO already complete: {output}", flush=True)
            return
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty and vrpo.training.resume is false")

    path = str(model_settings["path"])
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=bool(model_settings["trust_remote_code"]))
    base = AutoModel.from_pretrained(
        path, trust_remote_code=bool(model_settings["trust_remote_code"]), low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, str(engine["dtype"])),
        **({} if engine["attn_implementation"] is None else {"attn_implementation": engine["attn_implementation"]}),
    ).to(device)
    if options["gradient_checkpointing"]:
        base.gradient_checkpointing_enable()
    base.config.use_cache = False
    model = (PeftModel.from_pretrained(base, output, is_trainable=True) if start
             else get_peft_model(base, LoraConfig(**vrpo["lora"], task_type=None)))
    model.train()
    reference = AdapterDisabledReference(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(options["learning_rate"]))
    if start:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    estimator = VRPOSamplingConfig(timestep_samples=int(options["timestep_samples"]),
                                   masks_per_timestep=int(options["masks_per_timestep"]),
                                   antithetic=bool(options["antithetic"]))
    system = vrpo["prompt"]["system"]
    maximum = int(vrpo["max_new_tokens"])

    def encode(text: str) -> tuple[int, ...]:
        messages = ([{"role": "system", "content": system}] if system is not None else []) + [
            {"role": "user", "content": text}]
        return tuple(int(value) for value in tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                                           tokenize=True))

    def completion(text: str) -> tuple[int, ...]:
        values = tuple(int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"][:maximum])
        if not values:
            raise ValueError("a VRPO completion is empty after tokenization")
        return values

    encoded = [(encode(str(pair["prompt"])), completion(str(pair["chosen"])), completion(str(pair["rejected"])),
                str(pair["problem_id"])) for pair in pairs]
    order = list(range(len(encoded)))
    random.Random(int(options["seed"])).shuffle(order)
    seeds = SeedStream(int(options["seed"]))
    total_parameters, active_parameters = active_parameter_counts(model)
    optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for microstep in range(start * accumulation, steps * accumulation):
            prompt, chosen, rejected, problem_id = encoded[order[microstep % len(order)]]
            estimate = estimate_vrpo_preference_loss(
                model, reference, prompt=prompt, chosen=chosen, rejected=rejected,
                mask_token_id=int(model_settings["mask_token_id"]), config=estimator, beta=float(options["beta"]),
                seed=seeds.derive("vrpo-train", microstep, problem_id),
            )
            (estimate.loss / accumulation).backward()
            for name, value in vrpo_forward_token_slots(prompt_length=len(prompt), chosen_length=len(chosen),
                                                        rejected_length=len(rejected), config=estimator).items():
                slots[name] += value
            if (microstep + 1) % accumulation:
                continue
            update = (microstep + 1) // accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(options["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metrics.append({"update": update, "problem_id": problem_id, "loss": float(estimate.loss.detach()),
                            "preference_score": float(estimate.preference_score.detach()),
                            "grad_norm": float(grad_norm.detach()),
                            "elapsed_seconds": elapsed_before + time.perf_counter() - started})
            print(json.dumps(metrics[-1]), flush=True)
            if update % int(options["save_steps"]) == 0 or update == steps:
                output.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(output))
                torch.save(optimizer.state_dict(), optimizer_path)
                write_json_atomic(state_path, {
                    "fingerprint": fingerprint, "completed_updates": update, "metrics": metrics,
                    "forward_token_slots": slots, "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                })
        forward = 2.0 * active_parameters * slots["current_policy"]
        recompute = forward if options["gradient_checkpointing"] else 0.0
        reference_forward = 2.0 * active_parameters * slots["reference_policy"]
        write_json_atomic(output / "training_cost.json", {
            "fingerprint": fingerprint, "completed_updates": steps, "microsteps": steps * accumulation,
            "preference_pairs": len(encoded), "total_parameters": total_parameters,
            "active_parameters": active_parameters,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "forward_token_slots": slots,
            # Backward is estimated as two policy forwards; checkpointing recomputes one forward.
            "estimated_flops": {"current_policy_forward": forward, "reference_forward": reference_forward,
                                "current_policy_backward": 2.0 * forward, "gradient_checkpoint_recompute": recompute,
                                "total": 3.0 * forward + reference_forward + recompute},
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        })
    finally:
        del optimizer, reference, model, base
        release_accelerator_memory()


__all__ = ["preferences", "train"]

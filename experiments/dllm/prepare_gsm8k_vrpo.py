"""Build verified GSM8K preference pairs for LLaDA VRPO training."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.dllm.runtime import (
    capped_generation_length,
    file_sha256,
    json_fingerprint,
    llada_snapshot_delta,
    sampling_from_section,
)
from experiments.dllm.profiles import apply_execution_profile
from experiments.shared.paired_protocol import load_pairing
from inference_scaling.dllm.backends import load_llada_backend
from inference_scaling.dllm.preferences import select_verified_preference_pair
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream

IMPLEMENTATION_FILES = (
    "experiments/dllm/prepare_gsm8k_vrpo.py",
    "experiments/dllm/profiles.py",
    "src/inference_scaling/dllm/preferences.py",
    "src/inference_scaling/dllm/backends/llada.py",
    "src/inference_scaling/dllm/backends/loader.py",
    "src/inference_scaling/shared/evaluation/gsm8k.py",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/train.jsonl"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--candidate-pool-size", type=int)
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    config = apply_execution_profile(config, args.profile)
    training = config["vrpo_training"]
    output_path = args.output or Path(str(training["preference_data"]))
    manifest_path = args.manifest or Path(str(training["preference_manifest"]))
    max_pairs = int(args.max_pairs or training["preference_pairs"])
    pool_size = int(args.candidate_pool_size or training["candidate_pool_size"])
    generations = int(training["num_generations"])
    for name, value in (
        ("max_pairs", max_pairs),
        ("candidate_pool_size", pool_size),
        ("num_generations", generations),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if max_pairs > pool_size:
        raise ValueError("max_pairs cannot exceed candidate_pool_size")

    model_dir = Path(str(config["model"]["path"]))
    weights = tuple(str(value) for value in config["model"]["weight_files"])
    expected_hashes = tuple(str(value) for value in config["model"]["weight_sha256"])
    actual_hashes: dict[str, str] = {}
    for name, expected_hash in zip(weights, expected_hashes, strict=True):
        path = model_dir / name
        actual = file_sha256(path)
        if actual != expected_hash:
            raise ValueError(f"LLaDA weight hash does not match the manifest: {path}")
        actual_hashes[name] = actual

    problems = select_problems(
        load_gsm8k(args.data, split="train"),
        pool_size,
        seed=int(training["seed"]),
    )
    effective = {
        "config": config,
        "profile": args.profile,
        "problem_indices": [problem.index for problem in problems],
        "max_pairs": max_pairs,
        "candidate_pool_size": pool_size,
        "num_generations": generations,
        "model_weight_sha256": actual_hashes,
        "implementation_sha256": {
            path: file_sha256(Path(path)) for path in IMPLEMENTATION_FILES
        },
    }
    fingerprint = json_fingerprint(effective)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("fingerprint") != fingerprint:
            raise ValueError("existing VRPO preference manifest has another fingerprint")
    else:
        manifest_path.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "status": "running", **effective},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    records = _load_records(output_path)
    if any(record.get("fingerprint") != fingerprint for record in records):
        raise ValueError("existing VRPO preference records have another fingerprint")
    completed = {int(record["problem_index"]) for record in records}
    pair_count = sum(record.get("status") == "pair" for record in records)
    backend = load_llada_backend(config, "base")
    sampling = sampling_from_section(config["generation"])
    generation_length = capped_generation_length(
        prompt_length=0,
        maximum=int(config["generation"]["max_new_tokens"]),
        sampling=sampling,
    )
    seeds = SeedStream(int(training["seed"]))
    before = backend.snapshot()
    started = time.perf_counter()
    try:
        with output_path.open("a", encoding="utf-8", buffering=1) as output:
            for problem in problems:
                if pair_count >= max_pairs:
                    break
                if problem.index in completed:
                    continue
                prompt_text = gsm8k_prompt(problem.question)
                prompt = backend.encode_chat(prompt_text)
                requests = [
                    DiffusionGenerationRequest(
                        prefix=prompt,
                        generation_length=generation_length,
                        sampling=sampling,
                        seed=seeds.derive("vrpo-preference", problem.index, draw),
                        request_id=f"vrpo-preference:{problem.index}:{draw}",
                    )
                    for draw in range(generations)
                ]
                samples = backend.sample_batch(requests)
                texts = [backend.decode(sample.token_ids) for sample in samples]
                pair = select_verified_preference_pair(
                    candidate_texts=texts,
                    gold_solution=problem.gold_solution,
                    gold_answer=problem.gold_answer,
                )
                record: dict[str, Any] = {
                    "fingerprint": fingerprint,
                    "problem_index": problem.index,
                    "question": problem.question,
                    "gold_answer": str(problem.gold_answer),
                    "candidates": [
                        {
                            "draw": draw,
                            "text": text,
                            "prediction": (
                                str(prediction) if prediction is not None else None
                            ),
                            "correct": prediction == problem.gold_answer,
                        }
                        for draw, (text, prediction) in enumerate(
                            (
                                (text, extract_numeric_answer(text))
                                for text in texts
                            )
                        )
                    ],
                }
                if pair is None:
                    record["status"] = "skipped_all_correct"
                else:
                    record.update(
                        status="pair",
                        prompt=prompt_text,
                        chosen=pair.chosen,
                        rejected=pair.rejected,
                        chosen_source=pair.chosen_source,
                        rejected_candidate_index=pair.rejected_candidate_index,
                    )
                    pair_count += 1
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed.add(problem.index)
                print(
                    f"preferences rows={len(completed)}/{len(problems)} pairs={pair_count}/{max_pairs}",
                    flush=True,
                )
        if pair_count < max_pairs:
            raise RuntimeError(
                f"candidate pool yielded {pair_count} pairs, fewer than requested {max_pairs}"
            )
        compute = llada_snapshot_delta(before, backend.snapshot())
        manifest = {
            "fingerprint": fingerprint,
            "status": "complete",
            **effective,
            "records": len(records),
            "preference_pairs": pair_count,
            "elapsed_seconds": time.perf_counter() - started,
            "generation_compute": compute,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        del backend
        gc.collect()
        if str(config["runtime"]["device"]).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

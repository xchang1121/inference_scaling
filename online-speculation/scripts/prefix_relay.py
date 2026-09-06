"""Train and benchmark PrefixRelay on an existing frozen local diffusion adapter."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from blockspec.benchmark import aggregate, continuation_prompts
from blockspec.checkpoint import base_fingerprint, implementation_fingerprint, load_checkpoint, load_hf_base
from blockspec.data import assert_split_files_disjoint, load_sequences
from blockspec.decoding import generate_ar, generate_speculative
from blockspec.execution import FixedShapeExecutor
from blockspec.relay import (RelayConfig, RelayHead, RelayLearner, generate_relay, load_relay, save_relay)
from blockspec.sampling import SamplingConfig


def emit(row):
    print(json.dumps(row), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "benchmark"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="validation file; always checked against training data")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True, help="exclusive train output or benchmark input")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prompts", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--train-requests", type=int, default=64)
    parser.add_argument("--train-tokens", type=int, default=128)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--threshold", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--online", action="store_true", help="include an additional head-continuation arm")
    args = parser.parse_args()
    if args.mode == "train" and args.head.exists():
        parser.error("select a new checkpoint output path")
    if args.repeats < 2 or args.repeats % 2 or min(args.tokens, args.train_tokens, args.train_requests) < 1:
        parser.error("positive budgets and a positive even repeat count required")
    assert_split_files_disjoint(args.train_data, args.data)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    config = payload["config"]
    del payload
    model = load_hf_base(args.base, rank=config["adapter_rank"], alpha=config["adapter_alpha"], device="cuda")
    model, _ = load_checkpoint(args.adapter, model=model, device="cuda")
    model.eval().requires_grad_(False).set_attention_backend("grouped")
    binding = {"base": base_fingerprint(model), "adapter": sha(args.adapter)}
    head = (RelayHead(RelayConfig(model.config.vocab_size, model.config.hidden_size, args.rank)).to("cuda")
            if args.mode == "train" else load_relay(args.head, binding=binding, device="cuda")[0])
    source = implementation_fingerprint()
    provenance = {"implementation_sha256": source, "script_sha256": sha(__file__), "binding": binding,
                  "validation_sha256": sha(args.data), "train_sha256": sha(args.train_data),
                  "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "head_parameters": sum(p.numel() for p in head.parameters()), "precision": "float32",
                  "device": torch.cuda.get_device_name(), "torch": torch.__version__}
    engine = FixedShapeExecutor(model, capacity=args.prompt_length + max(args.tokens, args.train_tokens, 32),
                                max_query=args.block_size)
    clean = [(n, False, None) for n in range(1, args.block_size + 1)]
    ordinary = [(n, True, None) for n in range(2, args.block_size + 1)]
    relay = [(n, True, model.config.num_hidden_layers) for n in range(2, args.block_size + 1)]
    engine.prepare(clean + ordinary + relay)
    setups = {"ar": engine.signature_seconds[(1, False, None)],
              "parallel": sum(engine.signature_seconds[k] for k in clean + ordinary),
              "relay": sum(engine.signature_seconds[k] for k in clean + relay)}
    emit({"stage": "start", **provenance, "engine_setup_seconds": setups})
    options = {"block_size": args.block_size, "sampling": SamplingConfig(args.temperature), "executor": engine}
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    prompts = continuation_prompts(load_sequences(args.data, model.config.vocab_size),
                                   count=args.prompts, length=args.prompt_length)
    # Warm all paths using a training-record prefix, including every scheduled query shape.
    sequences = load_sequences(args.train_data, model.config.vocab_size)
    warm = sequences[0][:args.prompt_length].reshape(1, -1).cuda()
    generate_ar(model, warm, 32, sampling=options["sampling"], executor=engine, generator=rng)
    generate_speculative(model, warm, 32, **options, generator=rng)
    generate_relay(model, head, warm, 32, **options, generator=rng)
    if args.mode == "train":
        learner = RelayLearner(head, lr=args.lr, interval=args.interval,
                               sampling=SamplingConfig(args.temperature if args.temperature > 0 else 1.))
        crop_rng = torch.Generator().manual_seed(args.seed)
        eligible = [s for s in sequences if len(s) >= args.prompt_length]
        start = time.perf_counter()
        total = 0
        for request in range(args.train_requests):
            sequence = eligible[int(torch.randint(len(eligible), (), generator=crop_rng))]
            offset = int(torch.randint(len(sequence) - args.prompt_length + 1, (), generator=crop_rng))
            prompt = sequence[offset:offset + args.prompt_length].reshape(1, -1).cuda()
            result = generate_relay(model, head, prompt, args.train_tokens, **options, learner=learner, generator=rng)
            total += len(result.tokens)
            if request % 8 == 0 or request + 1 == args.train_requests:
                emit({"stage": "training", "request": request + 1, "updates": learner.updates,
                      "examples": learner.examples, "seconds": time.perf_counter() - start, **learner.last_metrics})
        metadata = {**provenance, "training_seconds": time.perf_counter() - start, "generated_tokens": total,
                    "updates": learner.updates, "examples": learner.examples,
                    "update_seconds": learner.update_seconds, "last_metrics": learner.last_metrics}
        if base_fingerprint(model) != binding["base"]:
            raise RuntimeError("frozen base changed during head training")
        save_relay(args.head, head, binding=binding, metadata=metadata)
        emit({"stage": "trained", "head_sha256": sha(args.head), **metadata})
        return
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    arms = ["ar", "parallel", "relay", "scheduled"] + (["online"] if args.online else [])
    rows = {arm: [] for arm in arms}
    repeats = []
    greedy_matches = {arm: 0 for arm in arms if arm != "ar"}
    for repeat in range(args.repeats):
        head.load_state_dict(initial)
        learner = (RelayLearner(head, lr=args.lr, interval=args.interval,
                                sampling=SamplingConfig(args.temperature if args.temperature > 0 else 1.))
                   if args.online else None)
        online_state = {k: v.clone() for k, v in initial.items()}
        per_repeat = {arm: [] for arm in arms}
        for index, prompt in enumerate(prompts):
            order = arms if (repeat + index) % 2 == 0 else arms[::-1]
            outputs = {}
            for arm in order:
                head.load_state_dict(online_state if arm == "online" else initial)
                generator = torch.Generator(device="cuda").manual_seed(args.seed + 10000 * repeat + index)
                if arm == "ar":
                    result = generate_ar(model, prompt.cuda(), args.tokens, sampling=options["sampling"],
                                         executor=engine, generator=generator)
                elif arm == "parallel":
                    result = generate_speculative(model, prompt.cuda(), args.tokens, **options, generator=generator)
                else:
                    result = generate_relay(model, head, prompt.cuda(), args.tokens, **options, generator=generator,
                                            threshold=args.threshold if arm == "scheduled" else 0.,
                                            learner=learner if arm == "online" else None)
                if arm == "online":
                    online_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                row = result.summary()
                rows[arm].append(row)
                per_repeat[arm].append(row)
                outputs[arm] = result.tokens
            if args.temperature == 0:
                for arm in greedy_matches:
                    greedy_matches[arm] += outputs[arm] == outputs["ar"]
            emit({"stage": "benchmark", "repeat": repeat, "request": index,
                  "tps": {arm: per_repeat[arm][-1]["tps"] for arm in arms}})
        repeats.append({arm: aggregate(per_repeat[arm]) for arm in arms})
    summary = {}
    for arm in arms:
        setup = setups.get(arm, setups["relay"])
        summary[arm] = aggregate(rows[arm], engine_setup_seconds=setup)
        if arm not in ("ar", "parallel"):
            summary[arm]["verified_tokens"] = sum(r["verified_tokens"] for r in rows[arm])
            for key in ("depth_proposed", "depth_accepted"):
                summary[arm][key] = [sum(r[key][i] for r in rows[arm]) for i in range(args.block_size - 1)]
    emit({"stage": "complete", **provenance, "head_sha256": sha(args.head), "summary": summary,
          "repetitions": repeats, "greedy_matches": greedy_matches if args.temperature == 0 else None,
          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
          "frozen_base_unchanged": base_fingerprint(model) == binding["base"]})


if __name__ == "__main__":
    main()

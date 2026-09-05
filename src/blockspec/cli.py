"""Small executable end-to-end pipeline; no imported research implementation."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import adapter_state, base_fingerprint, load_checkpoint, load_hf_base, save_checkpoint
from .data import load_sequences
from .decoding import generate_ar, generate_speculative
from .distillation import offline_step, paired_loss
from .model import Decoder, ModelConfig
from .online import OnlineConfig, OnlineLearner
from .training import TrainingConfig, train_adapter


def toy_sequences(count, length, vocab, *, device, generator):
    """A learnable periodic stream, not a natural-language speed benchmark."""
    starts = torch.randint(1, vocab - 1, (count, 1), device=device, generator=generator)
    positions = torch.arange(length - 1, device=device)[None]
    # Token 0 is BOS; 1..V-2 form a deterministic cycle; V-1 is unused EOS.
    body = 1 + ((starts - 1 + positions) % (vocab - 2))
    return torch.cat((torch.zeros(count, 1, device=device, dtype=torch.long), body), dim=1)


def demo(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    config = ModelConfig(vocab_size=16, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, adapter_rank=args.rank,
                         adapter_alpha=float(args.rank))
    model = Decoder(config).to(device)
    train = toy_sequences(64, 33, config.vocab_size, device=device, generator=rng)
    model.train_base_only()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=0.003, weight_decay=0)
    for step in range(args.base_steps):
        batch = train[(step % 8) * 8:(step % 8 + 1) * 8]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        optimizer.step()
    print(json.dumps({"stage": "base", "steps": args.base_steps, "loss": float(loss.detach())}), flush=True)
    model.train_adapters_only()
    frozen = base_fingerprint(model)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=0.003, weight_decay=0)
    fixed_noise = torch.randint(config.vocab_size, train[:8].shape, device=device, generator=rng)
    with torch.no_grad():
        before = float(paired_loss(model, train[:8], args.block_size, noisy=fixed_noise))
    blocks = [2, 4, args.block_size]
    for step in range(args.adapter_steps):
        block = blocks[min(2, 3 * step // args.adapter_steps)]
        stats = offline_step(model, optimizer, train[(step % 8) * 8:(step % 8 + 1) * 8],
                             block, kind=args.loss, generator=rng)
    with torch.no_grad():
        after = float(paired_loss(model, train[:8], args.block_size, noisy=fixed_noise))
    assert base_fingerprint(model) == frozen, "offline training changed base weights"
    print(json.dumps({"stage": "offline", "steps": args.adapter_steps,
                      "fixed_noise_l1_before": before, "fixed_noise_l1_after": after,
                      "last_train": stats}), flush=True)
    if args.checkpoint:
        save_checkpoint(args.checkpoint, model, metadata={"task": "synthetic-cycle", "seed": args.seed})
        loaded, _ = load_checkpoint(args.checkpoint, device=device)
        assert base_fingerprint(loaded) == frozen
        for name, value in adapter_state(model).items():
            assert torch.equal(value, adapter_state(loaded)[name])
        model = loaded.train_adapters_only()
    prompt = train[:1, :5]
    ar = generate_ar(model, prompt, args.tokens)
    static = generate_speculative(model, prompt, args.tokens, block_size=args.block_size,
                                  generator=torch.Generator(device=device).manual_seed(args.seed + 2))
    learner = OnlineLearner(model, OnlineConfig(stride=args.update_stride, replay_blocks=2,
                                               learning_rate=0.001, loss=args.loss))
    original_adapter = adapter_state(model)
    online = generate_speculative(model, prompt, args.tokens, block_size=args.block_size,
                                  generator=torch.Generator(device=device).manual_seed(args.seed + 2),
                                  learner=learner)
    assert ar.tokens == static.tokens == online.tokens, "greedy outputs differ"
    assert base_fingerprint(model) == frozen, "online training changed base weights"
    changed = any(not torch.equal(v, adapter_state(model)[k]) for k, v in original_adapter.items())
    print(json.dumps({"stage": "decode", "ar": ar.summary(), "static": static.summary(),
                      "online": online.summary(), "greedy_identical": True,
                      "base_unchanged": True, "online_adapter_changed": changed,
                      "scope": "synthetic correctness smoke; not a speed claim"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("demo", help="base training -> offline adapter -> checkpoint -> online decode")
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--seed", type=int, default=314159)
    run.add_argument("--base-steps", type=int, default=120)
    run.add_argument("--adapter-steps", type=int, default=240)
    run.add_argument("--rank", type=int, default=8)
    run.add_argument("--block-size", type=int, default=4)
    run.add_argument("--tokens", type=int, default=128)
    run.add_argument("--update-stride", type=int, default=8)
    run.add_argument("--loss", choices=["l1", "tv", "forward_kl", "reverse_kl"], default="l1")
    run.add_argument("--checkpoint", type=Path)
    train = sub.add_parser("train", help="independently train a fresh adapter on local JSONL sequences")
    train.add_argument("--base", type=Path, required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--sequence-length", type=int, default=128)
    train.add_argument("--blocks", default="2,4,6,8")
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--bos-id", type=int, default=0)
    train.add_argument("--seed", type=int, default=314159)
    train.add_argument("--text-data", action="store_true", help="use local HF tokenizer for text records")
    train.add_argument("--loss", choices=["l1", "tv", "forward_kl", "reverse_kl"], default="l1")
    train.add_argument("--warmup-loss", choices=["forward_kl", "reverse_kl"], default="reverse_kl")
    args = parser.parse_args()
    if args.command == "train":
        if args.output.exists():
            parser.error("output already exists; choose a new checkpoint path")
        config = TrainingConfig(steps=args.steps, batch_size=args.batch_size,
                                sequence_length=args.sequence_length,
                                blocks=tuple(int(b) for b in args.blocks.split(",")),
                                learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                                warmup_loss=args.warmup_loss, loss=args.loss, seed=args.seed)
        torch.manual_seed(args.seed)
        model = load_hf_base(args.base, rank=args.rank, device=args.device, dtype=getattr(torch, args.dtype))
        fingerprint = base_fingerprint(model)
        tokenizer = None
        if args.text_data:
            from .tokenizer import LocalTokenizer
            tokenizer = LocalTokenizer(args.base)
        data = load_sequences(args.data, model.config.vocab_size, tokenizer=tokenizer)
        def progress(stats):
            if stats["step"] == 1 or stats["step"] % 25 == 0 or stats["step"] == args.steps:
                print(json.dumps(stats), flush=True)
        result = train_adapter(model, data, config, bos_id=args.bos_id, progress=progress)
        if base_fingerprint(model) != fingerprint:
            raise RuntimeError("offline training changed frozen base weights")
        save_checkpoint(args.output, model, adapter_only=True,
                        metadata={"training": result, "base_source": str(args.base), "seed": args.seed})
        print(json.dumps({"checkpoint": str(args.output), **result}), flush=True)
        return
    if min(args.base_steps, args.adapter_steps, args.rank, args.tokens, args.threads) < 1:
        parser.error("steps, rank, tokens and threads must be positive")
    if args.block_size < 2:
        parser.error("block size must be >=2")
    demo(args)


if __name__ == "__main__":
    main()

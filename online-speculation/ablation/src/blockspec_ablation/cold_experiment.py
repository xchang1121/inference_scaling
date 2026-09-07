"""Cold-start serving curve with a measured time budget and fixed held-out probes."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import time

import numpy as np
import torch

from blockspec import reporting
from blockspec.commands.evaluate import prompt_ids, prompt_texts
from blockspec.measurement import compare
from blockspec.parallel.fitting import FitConfig, frozen_fingerprint
from blockspec.parallel.online import portable
from blockspec.parallel.sampling import ProposalSampler
from blockspec.parallel.training import distillation_loss, sample_anchors
from blockspec.parallel.weights import load_public
from blockspec.sampling import SamplingConfig
from blockspec.sampling_execution import SamplingExecutor
from blockspec_ablation.cold_start import ColdStartService, FullBlockLearner, ServiceConfig, synchronize


@torch.no_grad()
def measure(service, prompts, tokens, *, seed, fixed_batches=None):
    """Research measurements: separate from delivered requests and gate decisions."""
    records, clean = [], []
    for index, prompt in enumerate(prompts):
        for name in (("ar", "candidate") if index % 2 == 0 else ("candidate", "ar")):
            if name == "candidate":
                with service.learner.candidate():
                    result = service._generate(prompt, tokens, speculative=True, seed=seed + index)
            else:
                result = service._generate(prompt, tokens, speculative=False, seed=seed + index)
                if fixed_batches is None:
                    sequence = torch.cat((prompt.cpu(), torch.tensor([result.tokens])), 1)
                    sequence = sequence[:, -service.fit.sequence_length:].long()
                    if sequence.shape[1] >= service.model.config.block_size:
                        rng = torch.Generator().manual_seed(seed + index + 1000)
                        anchors = sample_anchors(sequence, service.model.config.block_size,
                                                 service.fit.anchors_per_sequence, generator=rng)
                        clean.append((sequence, anchors))
            records.append({"request": index, "method": name, **result.summary()})
    batches = clean if fixed_batches is None else fixed_batches
    losses = []
    with service.learner.candidate(), service.learner.autocast():
        for sequence, anchors in batches:
            losses.append(float(distillation_loss(service.model, sequence.to(service.learner.device),
                                                  anchors.to(service.learner.device), chunk_rows=service.fit.chunk_rows)))
    metrics = {"step": service.learner.steps, "validation_kl": sum(losses) / len(losses) if losses else None,
               "candidate_over_ar": compare(records, "candidate", "ar", len(prompts), np.random.default_rng(seed))}
    for name in ("ar", "candidate"):
        selected = [row for row in records if row["method"] == name]
        count = sum(row["tokens"] for row in selected)
        elapsed = sum(row["seconds"] for row in selected)
        forwards = sum(row["decode_forwards"] for row in selected)
        metrics[name] = {"tokens": count, "seconds": elapsed, "tps": count / elapsed,
                         "decode_tpf": (count - sum(row["prefill_output_tokens"] for row in selected)) / forwards}
    return metrics, batches


def run(args):
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    fit = FitConfig(steps=args.steps, warmup_steps=args.warmup_steps, sequence_length=args.sequence_length,
                    anchors_per_sequence=args.anchors, accumulate=args.accumulate, learning_rate=args.learning_rate,
                    chunk_rows=args.chunk_rows, precision="bf16", seed=args.seed)
    settings = ServiceConfig(fraction=args.fraction, replay_records=args.replay_records, probe_every=args.probe_every,
                              probe_tokens=args.probe_tokens, publish_margin=args.publish_margin, seed=args.seed)
    training_texts = prompt_texts(args.prompts, args.requests, offset=args.offset)
    heldout_texts = prompt_texts(args.heldout_prompts, args.heldout_count, offset=args.heldout_offset)
    gate_texts = prompt_texts(args.heldout_prompts, args.gate_count, offset=args.heldout_offset + args.heldout_count)
    if (set(training_texts) & set(heldout_texts + gate_texts) or set(heldout_texts) & set(gate_texts)):
        raise ValueError("delivered, gate and held-out questions must be disjoint")
    model = load_public(args.model, device="cuda", dtype=torch.bfloat16).set_backend("sdpa")
    prompts = prompt_ids(args.model, args.requests, path=args.prompts, offset=args.offset, empty_system=True)
    heldout = prompt_ids(args.model, args.heldout_count, path=args.heldout_prompts,
                         offset=args.heldout_offset, empty_system=True)
    gate = prompt_ids(args.model, args.gate_count, path=args.heldout_prompts,
                      offset=args.heldout_offset + args.heldout_count, empty_system=True)
    sampling = SamplingConfig(1.)
    executor = SamplingExecutor(model.config.vocab_size, model.config.block_size, sampling, device="cuda")
    sampler = ProposalSampler(sampling, executor=executor)
    # Common model/sampling warm-up belongs to benchmark setup. The service
    # constructor separately charges its cold parameter initialization.
    from blockspec.parallel import MaskedAttentionBranch, generate_ar
    generate_ar(MaskedAttentionBranch(model), prompts[0], 8, sampler=sampler)
    base = frozen_fingerprint(model)
    service = ColdStartService(model, fit, settings, gate_prompts=gate, sampling=sampling,
                               sampler=sampler, retain_batches=args.offline_replay, deterministic_updates=args.deterministic)
    if not all(torch.equal(p, dict(layer.attention.ar.named_parameters())[name])
               for layer in model.layers for name, p in layer.attention.draft.named_parameters()):
        raise AssertionError("cold initialization must equal AR attention")
    if args.resume is not None:
        start = time.perf_counter()
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        read_seconds = time.perf_counter() - start
        service.load_state_dict(state)
        service.budget.charge("resume_io", read_seconds)
        del state
    curve, streams, research_seconds = [], [], 0.
    fixed_batches = None

    def checkpoint_curve():
        nonlocal fixed_batches, research_seconds
        synchronize(model)
        start = time.perf_counter()
        row, fixed_batches = measure(service, heldout, args.heldout_tokens,
                                      seed=args.seed + 200000, fixed_batches=fixed_batches)
        synchronize(model)
        research_seconds += time.perf_counter() - start
        curve.append(row)
        print(reporting.dumps({"curve": row, "service": service.summary()}), flush=True)

    checkpoint_curve()
    for index, prompt in enumerate(prompts):
        _, row = service.serve(prompt, args.tokens, seed=args.seed + index)
        streams.append(row)
        if row["update"] or row["probe"] or (index + 1) % args.log_every == 0:
            print(reporting.dumps(row), flush=True)
        if service.learner.steps and service.learner.steps % args.curve_every == 0 and curve[-1]["step"] != service.learner.steps:
            checkpoint_curve()
    if curve[-1]["step"] != service.learner.steps:
        checkpoint_curve()
    result = {"method": "cold-start-full-block-kl", "initialization": "AR attention copy",
              "deterministic": args.deterministic,
              "fit": asdict(fit), "controller": asdict(settings), "sampling": asdict(sampling),
              "stream": service.summary(), "curve": curve, "requests": streams,
              "research_measurement_seconds": research_seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "frozen_ar_unchanged": base == frozen_fingerprint(model)}
    if args.checkpoint is not None:
        # Private continuation state stays outside publication summaries.
        start = time.perf_counter()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = service.state_dict()
        with args.checkpoint.open("xb") as handle:
            torch.save(state, handle)
        result["shutdown_checkpoint_seconds"] = time.perf_counter() - start
    if args.offline_replay:
        # The same realized microbatches distinguish implementation parity from
        # a separate offline data-access or optimization-budget comparison.
        reference_weights = portable(service.learner.master)
        transcript = service.transcript
        del service.learner.optimizer
        service.learner.master.clear()
        synchronize(model)
        start = time.perf_counter()
        offline = FullBlockLearner(model, fit, deterministic=args.deterministic)
        for batches in transcript:
            offline.step(batches)
        synchronize(model)
        result["offline_replay"] = {"steps": offline.steps, "seconds": time.perf_counter() - start,
                                     "identical_parameters": all(torch.equal(p.cpu(), reference_weights[name])
                                                                 for name, p in offline.master.items()),
                                     "frozen_ar_unchanged": base == frozen_fingerprint(model)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        reporting.dump(result, handle, indent=2)
    print(reporting.dumps({key: value for key, value in result.items() if key != "requests"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--heldout-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    for name, default in (("requests", 128), ("offset", 0), ("tokens", 256), ("steps", 64), ("warmup-steps", 4),
                           ("sequence-length", 256), ("anchors", 4), ("accumulate", 1), ("chunk-rows", 32),
                           ("replay-records", 128), ("probe-every", 8), ("probe-tokens", 32), ("gate-count", 2),
                           ("heldout-count", 4), ("heldout-tokens", 128), ("heldout-offset", 64),
                           ("curve-every", 4), ("log-every", 8), ("seed", 743)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--fraction", type=float, default=.01)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--publish-margin", type=float, default=1.10)
    parser.add_argument("--offline-replay", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists() or (args.checkpoint is not None and args.checkpoint.exists()):
        parser.error("new result and checkpoint destinations required")
    if args.resume is not None and args.offline_replay:
        parser.error("full cold-start batch replay uses a fresh learning stream")
    if any(getattr(args, key) < 1 for key in ("requests", "tokens", "curve_every", "log_every", "heldout_count",
                                               "heldout_tokens", "gate_count")):
        parser.error("positive request, output and measurement sizes required")
    run(args)


if __name__ == "__main__":
    main()

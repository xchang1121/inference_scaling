"""Budgeted clean-sequence distillation, starting from the frozen AR attention."""

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
import math
import os
import time

import torch

from blockspec.parallel import MaskedAttentionBranch, generate, generate_ar
from blockspec.parallel.fitting import FitConfig, frozen_fingerprint
from blockspec.parallel.online import portable
from blockspec.parallel.sampling import ProposalSampler
from blockspec.parallel.training import distillation_update, sample_anchors
from blockspec.parallel.weights import source_identity
from blockspec.sampling import SamplingConfig


def synchronize(model):
    device = model.embedding.weight.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def update_determinism(enabled):
    """Scope deterministic backward to training; restore the serving runtime."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@dataclass
class TimeBudget:
    """Measured extra time earns credit at `fraction` of delivered generation time.

    Admission uses a duration estimate. An operation that exceeds that estimate
    creates debt, which subsequent delivered requests repay before more work.
    """

    fraction: float = .01
    service_seconds: float = 0.
    costs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= .1:
            raise ValueError("time fraction must lie in (0, 0.1]")

    @property
    def spent(self):
        return sum(self.costs.values())

    @property
    def credit(self):
        return self.fraction * self.service_seconds - self.spent

    def delivered(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("finite nonnegative service duration required")
        self.service_seconds += seconds

    def charge(self, kind, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("finite nonnegative maintenance duration required")
        self.costs[kind] = self.costs.get(kind, 0.) + seconds

    def admits(self, estimate):
        if not math.isfinite(estimate) or estimate <= 0:
            raise ValueError("finite positive operation estimate required")
        return self.credit >= estimate

    def summary(self):
        return {"fraction": self.fraction, "service_seconds": self.service_seconds,
                "costs": dict(self.costs), "extra_seconds": self.spent,
                "extra_over_service": self.spent / self.service_seconds if self.service_seconds else None,
                "remaining_credit_seconds": self.credit, "debt_seconds": max(0., -self.credit)}


class CleanReplay:
    """Bounded completed records, with the offline contiguous crop/anchor rule."""

    def __init__(self, capacity, sequence_length, block_size, seed):
        if (any(type(n) is not int or n < 1 for n in (capacity, sequence_length, block_size))
                or sequence_length < block_size or block_size < 2):
            raise ValueError("positive replay capacity and complete training blocks required")
        self.records = deque(maxlen=capacity)
        self.sequence_length, self.block_size = sequence_length, block_size
        self.rng = torch.Generator().manual_seed(seed)
        self.anchors_rng = torch.Generator().manual_seed(seed + 1)

    def append(self, tokens):
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.long:
            raise ValueError("one complete integer-token record required")
        if tokens.shape[1] >= self.block_size:
            self.records.append(tokens.detach().cpu().clone())

    def batch(self, count, anchors_per_sequence):
        # Variable completed lengths use separate microbatches, each with an
        # identical per-row mean objective and equal record weight.
        if not self.records or type(count) is not int or count < 1:
            raise ValueError("nonempty replay and positive microbatch count required")
        batches = []
        for _ in range(count):
            index = int(torch.randint(len(self.records), (), generator=self.rng))
            record = self.records[index]
            length = min(self.sequence_length, record.shape[1])
            start = int(torch.randint(record.shape[1] - length + 1, (), generator=self.rng))
            tokens = record[:, start:start + length].clone()
            anchors = sample_anchors(tokens, self.block_size, anchors_per_sequence, generator=self.anchors_rng)
            batches.append((tokens, anchors))
        return batches


class FullBlockLearner:
    """Full-attention FP32 masters with an independent serving version.

    Construction copies AR attention into every draft layer. Both offline replay
    and live learning call the same full-distribution update kernel.
    """

    def __init__(self, model, config, *, deterministic=False):
        if (not isinstance(config, FitConfig) or config.sequence_length < model.config.block_size
                or config.batch_size != 1 or any(p.requires_grad for p in model.parameters())
                or any(p.dtype not in (torch.float32, torch.bfloat16) for p in model.parameters())):
            raise ValueError("frozen FP32/BF16 execution and batch-one full-block configuration required")
        self.model, self.config = model, config
        self.device = model.embedding.weight.device
        if type(deterministic) is not bool:
            raise ValueError("explicit deterministic update policy required")
        if (self.device.type == "cuda" and deterministic
                and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8")):
            raise ValueError("deterministic CUDA updates require CUBLAS_WORKSPACE_CONFIG before GPU initialization")
        if config.precision == "bf16" and self.device.type != "cuda":
            raise ValueError("BF16 autocast requires CUDA")
        model.initialize_draft_from_ar()
        model.set_backend(config.backend).eval()
        self.execution = {name: p for name, p in model.named_parameters() if ".attention.draft." in name}
        self.master = {name: torch.nn.Parameter(p.detach().float().clone()) for name, p in self.execution.items()}
        self.optimizer = torch.optim.AdamW(list(self.master.values()), lr=config.learning_rate,
                                           weight_decay=config.weight_decay, foreach=False)
        self.frozen = [(p, p._version) for name, p in model.named_parameters() if name not in self.execution]
        self.deterministic = deterministic
        self.steps = self.serving_step = 0

    def check(self):
        if (self.model.backend != self.config.backend or any(p.requires_grad for p in self.model.parameters())
                or any(p._version != version for p, version in self.frozen)):
            raise RuntimeError("AR/shared parameters and serving execution must remain fixed")

    def runtime(self):
        return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device_type": self.device.type,
                "device_capability": list(torch.cuda.get_device_capability(self.device)) if self.device.type == "cuda" else None,
                "deterministic": self.deterministic, "matmul_precision": torch.get_float32_matmul_precision(),
                "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if self.device.type == "cuda" else None}

    def autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.config.precision == "bf16" else nullcontext())

    def step(self, batches):
        self.check()
        if self.steps >= self.config.steps:
            raise ValueError("full learning schedule has finished")
        batches = [(tokens.to(self.device), anchors.to(self.device)) for tokens, anchors in batches]
        with update_determinism(self.deterministic):
            metrics = distillation_update(self.model, batches, self.optimizer,
                                          learning_rate=self.config.rate(self.steps), clip_grad=self.config.clip_grad,
                                          chunk_rows=self.config.chunk_rows, autocast=self.autocast, master=self.master)
        self.steps += 1
        self.check()
        return {"step": self.steps, **metrics}

    @torch.no_grad()
    def publish(self):
        self.check()
        for name, parameter in self.execution.items():
            parameter.copy_(self.master[name])
        self.serving_step = self.steps

    @contextmanager
    def candidate(self):
        """Request-boundary trial; restore serving weights even when a probe fails."""
        self.check()
        saved = {name: p.detach().clone() for name, p in self.execution.items()}
        old_step = self.serving_step
        try:
            self.publish()
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.execution.items():
                    parameter.copy_(saved[name])
            self.serving_step = old_step


@dataclass(frozen=True)
class ServiceConfig:
    fraction: float = .01
    replay_records: int = 128
    probe_every: int = 16
    probe_tokens: int = 64
    publish_margin: float = 1.10
    initial_step_estimate: float = .5
    seed: int = 731

    def __post_init__(self):
        TimeBudget(self.fraction)
        if (any(type(n) is not int or n < 1 for n in (self.replay_records, self.probe_every, self.probe_tokens))
                or not math.isfinite(self.publish_margin) or self.publish_margin <= 1
                or not math.isfinite(self.initial_step_estimate) or self.initial_step_estimate <= 0):
            raise ValueError("positive replay/probe sizes, cost estimate and publication margin > 1 required")


class ColdStartService:
    """AR service plus budget-admitted learning and paired request-boundary trials.

    The caller supplies separate gate prompts. Held-out reporting prompts remain
    separate from both replay and the publication gate. Calls are serialized.
    """

    def __init__(self, model, fit, config=ServiceConfig(), *, gate_prompts=(), sampling=SamplingConfig(1.),
                 sampler=None, retain_batches=False, retain_records=False, deterministic_updates=False):
        synchronize(model)
        start = time.perf_counter()
        self.learner = FullBlockLearner(model, fit, deterministic=deterministic_updates)
        self.model, self.config, self.fit = model, config, fit
        self.branch = MaskedAttentionBranch(model)
        self.budget = TimeBudget(config.fraction)
        self.replay = CleanReplay(config.replay_records, fit.sequence_length, model.config.block_size, fit.seed)
        self.sampling = sampling
        self.sampler = ProposalSampler(sampling) if sampler is None else sampler
        self.gate_prompts = tuple(prompt.detach().clone() for prompt in gate_prompts)
        self.speculating = False
        self.next_probe = config.probe_every
        self.step_estimate = config.initial_step_estimate
        self.probe_estimate = 0.
        self.probe_legs = 2
        self.tokens = self.requests = 0
        self.transcript = [] if retain_batches else None
        self.completed_records = [] if retain_records else None
        self.last_probe = None
        synchronize(model)
        self.budget.charge("setup", time.perf_counter() - start)

    def _timed(self, kind, action):
        synchronize(self.model)
        start = time.perf_counter()
        try:
            return action()
        finally:
            synchronize(self.model)
            self.budget.charge(kind, time.perf_counter() - start)

    def _generate(self, prompt, tokens, *, speculative, seed):
        generator = torch.Generator(device=prompt.device).manual_seed(seed)
        method = generate if speculative else generate_ar
        return method(self.branch, prompt, tokens, sampling=self.sampling,
                      eos_id=self.model.config.eos_token_id, generator=generator, sampler=self.sampler)

    def _probe(self):
        baseline, candidate, ar = [], [], []
        # Each pair has identical prompt, output limit and random seed. The
        # method order alternates to reduce systematic first-run timing bias.
        for index, prompt in enumerate(self.gate_prompts):
            seed = self.config.seed + 100000 + index
            pair = {}
            names = ["baseline", "candidate"] + (["ar"] if self.speculating else [])
            if index % 2:
                names.reverse()
            for name in names:
                if name == "candidate":
                    with self.learner.candidate():
                        pair[name] = self._generate(prompt, self.config.probe_tokens, speculative=True, seed=seed)
                else:
                    pair[name] = self._generate(prompt, self.config.probe_tokens,
                                               speculative=self.speculating and name != "ar", seed=seed)
            baseline.append(pair["baseline"])
            candidate.append(pair["candidate"])
            ar.append(pair["ar"] if self.speculating else pair["baseline"])
        ratios = [new.tps / old.tps for old, new in zip(baseline, candidate, strict=True)]
        versus_ar = [new.tps / old.tps for old, new in zip(ar, candidate, strict=True)]
        total = lambda rows: sum(len(r.tokens) for r in rows) / sum(r.seconds for r in rows)
        return {"step": self.learner.steps, "paired_ratios": ratios,
                "ratio": total(candidate) / total(baseline),
                "candidate_over_ar": total(candidate) / total(ar),
                "fallback": self.speculating and total(baseline) < total(ar),
                "passed": all(ratio >= self.config.publish_margin for ratio in ratios + versus_ar)}

    def _probe_cost(self):
        legs = 3 if self.speculating else 2
        if self.probe_estimate:
            return self.probe_estimate * legs / self.probe_legs
        current_tps = self.tokens / self.budget.service_seconds
        return 2 * legs * len(self.gate_prompts) * self.config.probe_tokens / current_tps

    def serve(self, prompt, max_new_tokens, *, seed):
        synchronize(self.model)
        start = time.perf_counter()
        speculative = self.speculating
        result = self._generate(prompt, max_new_tokens, speculative=speculative, seed=seed)
        synchronize(self.model)
        service_seconds = time.perf_counter() - start
        self.budget.delivered(service_seconds)
        self.tokens += len(result.tokens)
        self.requests += 1
        maintenance_start, previous_spent = time.perf_counter(), self.budget.spent
        update, probe = None, None

        def collect():
            if result.tokens:
                complete = torch.cat((prompt.detach().cpu(), torch.tensor([result.tokens], dtype=torch.long)), 1)
                self.replay.append(complete)
                if self.completed_records is not None and complete.shape[1] >= self.model.config.block_size:
                    self.completed_records.append(complete)

        self._timed("collection", collect)
        probe_due = bool(self.gate_prompts) and self.learner.steps >= self.next_probe
        if (not probe_due and self.replay.records and self.learner.steps < self.fit.steps
                and self.budget.admits(self.step_estimate)):
            before = self.budget.spent

            def train():
                batches = self.replay.batch(self.fit.accumulate, self.fit.anchors_per_sequence)
                metrics = self.learner.step(batches)
                if self.transcript is not None:
                    self.transcript.append(batches)
                return metrics

            update = self._timed("training", train)
            elapsed = self.budget.spent - before
            self.step_estimate = max(elapsed * 1.2, self.step_estimate * .8)
            update["seconds"] = elapsed
        if self.gate_prompts and self.learner.steps >= self.next_probe:
            if self.budget.admits(self._probe_cost()):
                before = self.budget.spent
                self.probe_legs = 3 if self.speculating else 2
                probe = self._timed("validation", self._probe)
                self.probe_estimate = 1.2 * (self.budget.spent - before)
                if probe["passed"]:
                    self._timed("publication", self.learner.publish)
                    self.speculating = True
                elif probe.get("fallback", False):
                    self.speculating = False
                self.last_probe = probe
                self.next_probe = self.learner.steps + self.config.probe_every
        residual = time.perf_counter() - maintenance_start - (self.budget.spent - previous_spent)
        self.budget.charge("scheduling", max(0., residual))
        return result, {"request": self.requests, "mode": "speculative" if speculative else "ar",
                        "service_seconds": service_seconds, "update": update, "probe": probe, **self.summary()}

    def summary(self):
        elapsed = self.budget.service_seconds + self.budget.spent
        return {"delivered_tokens": self.tokens, "steps": self.learner.steps,
                "serving_step": self.learner.serving_step, "speculating": self.speculating,
                "net_tps": self.tokens / elapsed if elapsed else 0., "budget": self.budget.summary()}

    def state_dict(self):
        """Private request-boundary state, including unpublished learning progress."""
        self.learner.check()
        return {"format": "cold-start-research-v1", "fit": asdict(self.fit), "controller": asdict(self.config),
                "sampling": asdict(self.sampling), "source": source_identity(getattr(self.model, "source", {})),
                "runtime": self.learner.runtime(), "model_config": self.model.config.to_dict(),
                "base_fingerprint": frozen_fingerprint(self.model),
                "master": portable(self.learner.master), "optimizer": portable(self.learner.optimizer.state_dict()),
                "serving": portable(self.learner.execution), "step": self.learner.steps,
                "serving_step": self.learner.serving_step, "speculating": self.speculating,
                "gate_prompts": portable(self.gate_prompts),
                "replay": portable(list(self.replay.records)), "replay_rng": self.replay.rng.get_state(),
                "anchors_rng": self.replay.anchors_rng.get_state(), "stream": self.summary(),
                "next_probe": self.next_probe, "step_estimate": self.step_estimate,
                "probe_estimate": self.probe_estimate, "probe_legs": self.probe_legs, "requests": self.requests}

    def load_state_dict(self, state):
        """Restore the learning/serving pair; newly incurred loading time is charged."""
        synchronize(self.model)
        start = time.perf_counter()
        self.learner.check()
        if self.requests or self.learner.steps or self.replay.records:
            raise ValueError("restore into a fresh service instance")
        if "gate_prompts" in state and (not isinstance(state["gate_prompts"], (tuple, list))
                or len(state["gate_prompts"]) != len(self.gate_prompts) or any(
                    not isinstance(saved, torch.Tensor) or saved.dtype != prompt.dtype
                    or not torch.equal(saved.cpu(), prompt.cpu())
                    for saved, prompt in zip(state["gate_prompts"], self.gate_prompts, strict=True))):
            raise ValueError("restore requires the same publication gate prompts")
        source = source_identity(getattr(self.model, "source", {}))
        if (state.get("format") != "cold-start-research-v1" or state.get("fit") != asdict(self.fit)
                or state.get("controller") != asdict(self.config)
                or state.get("sampling", asdict(SamplingConfig(1.))) != asdict(self.sampling)
                or source_identity(state.get("source", {})) != source
                or ("runtime" in state and state["runtime"] != self.learner.runtime())
                or ("model_config" in state and state["model_config"] != self.model.config.to_dict())
                or (not source and "base_fingerprint" not in state)
                or ("base_fingerprint" in state and state["base_fingerprint"] != frozen_fingerprint(self.model))):
            raise ValueError("saved stream must match its frozen base and training/sampling policies")
        for key, expected in (("master", self.learner.master), ("serving", self.learner.execution)):
            saved = state.get(key, {})
            if saved.keys() != expected.keys() or any(
                    p.shape != expected[name].shape or p.dtype != expected[name].dtype or not torch.isfinite(p).all()
                    for name, p in saved.items()):
                raise ValueError("matching finite learning and serving tensors required")
        if (type(state.get("step")) is not int or not 0 <= state["step"] <= self.fit.steps
                or type(state.get("serving_step")) is not int or not 0 <= state["serving_step"] <= state["step"]
                or type(state.get("speculating")) is not bool
                or (state["speculating"] and state["serving_step"] == 0)
                or type(state.get("next_probe")) is not int or state["next_probe"] < 1
                or state.get("probe_legs", 2) not in (2, 3)
                or type(state.get("requests", 0)) is not int or state.get("requests", 0) < 0
                or any(not math.isfinite(state.get(key, -1)) or state[key] < 0
                       for key in ("step_estimate", "probe_estimate")) or state["step_estimate"] == 0):
            raise ValueError("consistent learning, publication and trial counters required")
        records = state.get("replay", [])
        if len(records) > self.config.replay_records or any(
                not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != 1 or x.dtype != torch.long
                or x.shape[1] < self.model.config.block_size or ((x < 0) | (x >= self.model.config.vocab_size)).any()
                for x in records):
            raise ValueError("bounded clean replay with valid token records required")
        stream = state["stream"]
        budget = TimeBudget(self.config.fraction)
        budget.delivered(stream["budget"]["service_seconds"])
        for kind, elapsed in stream["budget"]["costs"].items():
            budget.charge(kind, elapsed)
        if type(stream.get("delivered_tokens")) is not int or stream["delivered_tokens"] < 0:
            raise ValueError("nonnegative delivered token count required")
        # Validate random states before changing parameters.
        replay_rng, anchor_rng = torch.Generator(), torch.Generator()
        replay_rng.set_state(state["replay_rng"].cpu())
        anchor_rng.set_state(state["anchors_rng"].cpu())
        with torch.no_grad():
            for name, p in self.learner.master.items():
                p.copy_(state["master"][name])
            for name, p in self.learner.execution.items():
                p.copy_(state["serving"][name])
        self.learner.optimizer.load_state_dict(state["optimizer"])
        self.learner.steps, self.learner.serving_step = state["step"], state["serving_step"]
        self.speculating, self.next_probe = state["speculating"], state["next_probe"]
        self.step_estimate, self.probe_estimate = state["step_estimate"], state["probe_estimate"]
        self.probe_legs = state.get("probe_legs", 2)
        self.replay.records.clear()
        self.replay.records.extend(x.detach().cpu().clone() for x in records)
        self.replay.rng, self.replay.anchors_rng = replay_rng, anchor_rng
        self.tokens, self.requests = stream["delivered_tokens"], state.get("requests", 0)
        if self.transcript is not None:
            self.transcript.clear()
        # Constructor setup and restoration are new work in this process.
        startup = self.budget.spent
        synchronize(self.model)
        budget.charge("resume", startup + time.perf_counter() - start)
        self.budget = budget

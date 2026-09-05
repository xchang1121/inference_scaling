"""Real request-local Online Uno: last-MLP low-rank gradient updates.

Frozen Uno features -> teacher distribution -> backward -> fixed-address BF16
publication after commit. No extra teacher/trunk forward, no KV-dependent weights.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


def eligible_rows(block_size: int, committed: int) -> int:
    """Noise positions on the verified history, including first rejection.

    Tail truncation can only make this mask conservative. B includes the clean root.
    """
    if block_size < 2 or not 1 <= committed <= block_size + 1:
        raise ValueError("invalid completed linear Uno cycle")
    return min(block_size - 1, committed - 1)


def replay_hidden(a, u, residual, left, right, weight, eps, n_groups, mask=None):
    """Differentiable replay; match serving casts, with no frozen-trunk graph."""
    delta = F.linear(F.linear(a, left.to(a.dtype)), right.to(a.dtype))
    if mask is not None:
        delta = delta * mask[:, None].to(delta.dtype)
    changed = u + delta
    summed = (changed.float() + residual.float()).to(u.dtype)
    grouped = summed.float().reshape(*summed.shape[:-1], n_groups, -1)
    unit = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return unit.reshape_as(summed).to(u.dtype) * weight


@torch.no_grad()
def distillation_signal(draft_logits, teacher_logits, head_weight, valid=None):
    """Use actual serving q, avoiding a second LM-head forward and softmax AD.

    Valid only while the cached proposals and learner have the same version.
    Cast dL/dlogits to the original output dtype, as ordinary mixed-precision AD.
    """
    p = teacher_logits.float().softmax(-1)
    logq = draft_logits.float().log_softmax(-1)
    if valid is None:
        gradient_logits = ((logq.exp() - p) / p.shape[0]).to(head_weight.dtype)
        loss = F.kl_div(logq, p, reduction="batchmean")
    else:
        denominator = valid.sum().clamp_min(1)
        gradient_logits = ((logq.exp() - p) * (valid / denominator)[:, None]).to(head_weight.dtype)
        loss = (F.kl_div(logq, p, reduction="none").sum(-1) * valid).sum() / denominator
    gradient_hidden = gradient_logits @ head_weight
    return loss, gradient_hidden


class FastWeights:
    def __init__(self, input_size, hidden_size, norm_weight, head_weight, *,
                 eps=1e-6, n_groups=1, rank=8, max_block=16, lr=0.001, stride=16,
                 seed=1729, replay_blocks=4, training_backend="eager"):
        if rank < 1 or stride < 1 or max_block < 2 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("positive rank, stride, learning rate and block >= 2 required")
        if head_weight.requires_grad or norm_weight.requires_grad:
            raise ValueError("teacher head and norm must be frozen")
        if not 1 <= replay_blocks <= stride:
            raise ValueError("replay_blocks must be within 1..stride")
        self.rank, self.stride, self.lr = rank, stride, lr
        self.replay_blocks = replay_blocks
        self.eps, self.n_groups = eps, n_groups
        self.norm_weight, self.head_weight = norm_weight, head_weight
        device, dtype = head_weight.device, head_weight.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        self.initial_left = torch.randn(rank, input_size, generator=generator,
                                        device=device, dtype=torch.float32) / math.sqrt(input_size)
        self.left = nn.Parameter(self.initial_left.clone())
        self.right = nn.Parameter(torch.zeros(hidden_size, rank, device=device, dtype=torch.float32))
        # Outside the base model: never register fast weights under frozen Uno.
        self.serve_left = self.left.detach().to(dtype).clone()
        self.serve_right = self.right.detach().to(dtype).clone()
        self.features = torch.empty(max_block, input_size, device=device, dtype=dtype)
        self.before_delta = torch.empty(max_block, hidden_size, device=device, dtype=dtype)
        self.residual = torch.empty_like(self.before_delta)
        self.training_graph = None
        self.training_backend = training_backend
        if training_backend not in {"eager", "cuda_graph"}:
            raise ValueError("unknown training backend")
        self.reset()
        if training_backend == "cuda_graph":
            from native_update_graph import CapturedUpdate
            self.training_graph = CapturedUpdate(self, replay_hidden, distillation_signal)

    @torch.no_grad()
    def reset(self):
        self.left.copy_(self.initial_left)
        self.right.zero_()
        self.serve_left.copy_(self.left)
        self.serve_right.zero_()
        if self.training_graph is None:
            self.optimizer = torch.optim.Adam([self.left, self.right], lr=self.lr, foreach=False)
        else:
            self.training_graph.reset()
        self.version = 0
        self.cycles = 0
        self.update_seconds = 0.0
        self.last_teacher = None
        self.last_draft = None
        self.events = []
        self.examples = []

    def remember(self, rows, block_size):
        if rows == 0:
            return
        if self.training_graph is not None:
            self.training_graph.remember(rows, block_size)
            self.examples.append(dict(rows=rows, version=self.version))
            self.examples = self.examples[-self.replay_blocks:]
            return
        # Features are invariant to fast weights; the bounded window is emptied
        # at each update, so every example was generated by the current version.
        with torch.inference_mode(False), torch.no_grad():
            self.examples.append(dict(
                a=self.features[:block_size].detach().clone(),
                u=self.before_delta[:block_size].detach().clone(),
                residual=self.residual[:block_size].detach().clone(),
                teacher=self.last_teacher, draft=self.last_draft, rows=rows,
                version=self.version,
            ))
        self.examples = self.examples[-self.replay_blocks:]

    def attach(self, model, context_getter=None):
        if context_getter is None:
            from nano_vllm_uno.utils.context import get_context
        else:
            get_context = context_getter

        down = model.model.layers[-1].mlp.down_proj
        final_norm = model.model.norm
        original_down, original_norm = down.forward, final_norm.forward

        def down_forward(a):
            u = original_down(a)
            context = get_context()
            if context.lora_enabled:
                n = a.shape[0]
                self.features[:n].copy_(a)
                self.before_delta[:n].copy_(u)
                delta = F.linear(F.linear(a, self.serve_left), self.serve_right)
                # Seed row is always zero; OFF graphs execute original_down only.
                u = u + delta * context.lora_mask[:n, None].to(u.dtype)
            return u

        def norm_forward(x, residual=None):
            if get_context().lora_enabled:
                self.residual[:x.shape[0]].copy_(residual)
            return original_norm(x, residual)

        down.forward, final_norm.forward = down_forward, norm_forward

    def logits(self, a, u, residual, mask=None):
        hidden = replay_hidden(a, u, residual, self.left, self.right,
                               self.norm_weight, self.eps, self.n_groups, mask)
        return F.linear(hidden, self.head_weight)

    def update(self, rows, *, audit=False, block_size=None):
        if rows == 0:
            return
        if self.training_graph is not None:
            if not self.examples:
                self.remember(rows, block_size or rows + 1)
            return self.training_graph.update(audit=audit)
        started = time.perf_counter()
        # run_cycle uses inference_mode. Copies here must be normal tensors for AD.
        with torch.inference_mode(False), torch.enable_grad():
            block_size = block_size or rows + 1
            if not self.examples:
                self.remember(rows, block_size)
            examples = self.examples
            if any(e["version"] != self.version for e in examples):
                raise RuntimeError("cached draft logits are stale; cannot reuse the head gradient")
            a, u, residual = [torch.cat([e[key] for e in examples]) for key in ("a", "u", "residual")]
            mask = torch.ones(a.shape[0], device=a.device, dtype=a.dtype)
            mask[::block_size] = 0
            indices = torch.tensor([j * block_size + i for j, e in enumerate(examples)
                                    for i in range(1, e["rows"] + 1)], device=a.device)
            teacher = torch.cat([e["teacher"][:e["rows"]] for e in examples])
            original_logits = torch.cat([e["draft"][:e["rows"]] for e in examples])
            rows = len(indices)
            self.optimizer.zero_grad(set_to_none=True)
            full_hidden = replay_hidden(a, u, residual, self.left, self.right,
                                        self.norm_weight, self.eps, self.n_groups, mask)
            hidden = full_hidden[indices]
            loss, gradient_hidden = distillation_signal(original_logits, teacher, self.head_weight)
            hidden.backward(gradient_hidden)
            grad_norm = nn.utils.clip_grad_norm_([self.left, self.right], 1.0,
                                                error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                if not bool(torch.isfinite(self.left).all() & torch.isfinite(self.right).all()):
                    raise FloatingPointError("non-finite online weights; abort request")
                self.serve_left.copy_(self.left)
                self.serve_right.copy_(self.right)
                metrics = [loss.detach(), grad_norm.detach(), self.right.norm()]
                if audit:
                    # Audit the complete replay shape before selecting loss rows.
                    # Keep pre-update hidden values; parameters were just published.
                    logits = F.linear(full_hidden.detach(), self.head_weight)[indices]
                    new_logits = self.logits(a, u, residual, mask)[indices]
                    metrics.extend([
                        (logits - original_logits).abs().max().float(),
                        (new_logits - logits).abs().max().float(),
                        F.kl_div(new_logits.float().log_softmax(-1), teacher.float().softmax(-1), reduction="batchmean"),
                    ])
                values = torch.stack(metrics).float().tolist()
        # Same stream, followed by a CPU read above: publication completes before replay.
        self.version += 1
        elapsed = time.perf_counter() - started
        self.update_seconds += elapsed
        event = dict(cycle=self.cycles, version=self.version, rows=rows,
                     replay_blocks=len(examples),
                     kl_before=values[0], grad_norm=values[1], right_norm=values[2],
                     seconds=elapsed)
        if audit:
            event.update(replay_max_logit_error=values[3],
                         same_features_logit_change=values[4], kl_after=values[5])
        self.events.append(event)
        self.examples = []

    def snapshot(self):
        return dict(algorithm="last_mlp_online_lora", rank=self.rank, stride=self.stride,
                    training_backend=self.training_backend,
                    training_graph_replays=self.version if self.training_graph is not None else 0,
                    replay_blocks=self.replay_blocks,
                    learning_rate=self.lr, trainable_parameters=self.left.numel() + self.right.numel(),
                    optimizer_steps=self.version, model_weight_updates=self.version,
                    teacher_weight_updates=0, offline_uno_weight_updates=0,
                    cycles=self.cycles, update_seconds=self.update_seconds,
                    publication="after commit, same CUDA stream, fixed addresses",
                    loss="full-vocabulary forward KL, T=1, verified-prefix mask",
                    gradient="cached serving q; analytic frozen-head VJP; same-version replay",
                    events=self.events)


@contextmanager
def extended_runner(*, fused_norm=False, fast_weights=False, rank=8, stride=16, lr=0.001,
                    replay_blocks=4, training_backend="cuda_graph"):
    """Scoped construction hook; pinned source files remain unchanged.

    Single-thread, single-GPU XLLM only. Install before warmup and graph capture.
    """
    from nano_vllm_uno.engine import llm_engine

    original = llm_engine.ModelRunner

    class Runner(original):
        def warmup_model(self):
            if self.world_size != 1 or self.config.max_num_seqs != 1 or self.config.tree_verify_size:
                raise ValueError("extensions require TP=1, batch=1, linear Uno")
            if (fused_norm or fast_weights) and self.config.hf_config.model_type not in {"xllm", "k2_aurora", "k2_horizon"}:
                raise ValueError("extensions currently support XLLM/K2 only")
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.fused_norm_count = 0
            if fused_norm:
                from native_norm import install_fused_norms
                self.fused_norm_count = install_fused_norms(self.model)
            self.fast_weights = None
            if fast_weights:
                cfg = self.config.hf_config
                self.fast_weights = FastWeights(
                    cfg.intermediate_size, cfg.hidden_size, self.model.model.norm.weight,
                    self.model.lm_head.weight, eps=cfg.rms_norm_eps,
                    n_groups=getattr(cfg, "layernorm_num_groups", 1), rank=rank,
                    max_block=self.config.max_diffusion_block_size, stride=stride, lr=lr,
                    replay_blocks=replay_blocks,
                    training_backend=training_backend,
                )
                self.fast_weights.attach(self.model)
            super().warmup_model()

    llm_engine.ModelRunner = Runner
    try:
        yield
    finally:
        llm_engine.ModelRunner = original


def frozen_digest(model):
    """Exact parameter+buffer byte hash; exclude KV, include unpacked/packed Uno.

    Used outside benchmark timing. Attention KV is runtime state, not weights.
    """
    digest = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if name.endswith(("k_cache", "v_cache")):
            continue
        digest.update(name.encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def generate_fast(engine, ids, params, budget, *, learn=True, audit=False):
    state = engine.model_runner.fast_weights
    if state is None or params.diffusion_block_size < 2:
        raise ValueError("fast weights must be installed before graph capture; B >= 2")
    if engine.scheduler.waiting or engine.scheduler.running:
        raise RuntimeError("request-local online wrapper requires an idle engine")
    state.reset()
    decoder = engine.model_runner.two_pass_decoder
    original_step, original_block = engine.step, decoder.run_block
    collecting = False

    def run_block(seqs, tokens, **kwargs):
        logits = original_block(seqs, tokens, **kwargs)
        if collecting:
            if kwargs.get("lora_mask_batch") is not None:
                state.last_draft = logits[0, 1:].detach().clone()
            else:
                state.last_teacher = logits[0, :-1].detach().clone()
        return logits

    def step():
        nonlocal collecting
        if engine.scheduler.waiting:
            return original_step()
        state.cycles += 1
        version = state.version
        window = getattr(state, "replay_blocks", 1)
        collecting = learn and (state.cycles - 1) % state.stride >= state.stride - window
        output, count = original_step()
        if count >= 0 or state.version != version:
            raise RuntimeError("online cycle must commit before any parameter publication")
        if collecting:
            # Do not pay for an update when there is no subsequent draft in this request.
            if engine.scheduler.running:
                rows = eligible_rows(params.diffusion_block_size, -count)
                if window > 1:
                    state.remember(rows, params.diffusion_block_size)
                if state.cycles % state.stride == 0:
                    state.update(rows, audit=audit, block_size=params.diffusion_block_size)
            state.last_teacher = state.last_draft = None
        collecting = False
        return output, count

    engine.step, decoder.run_block = step, run_block
    try:
        output = engine.generate([ids], params, use_tqdm=False, request_max_tokens=[budget])[0]
        diagnostics = state.snapshot()
        return output, diagnostics
    finally:
        engine.step, decoder.run_block = original_step, original_block
        # Do not leak the previous request's adaptation into another request/method.
        with torch.no_grad():
            state.serve_right.zero_()
        state.examples = []

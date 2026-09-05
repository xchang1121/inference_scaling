"""Fixed-address Online Uno update, including backward, Adam, and publication.

The graph has its own pool; it is never replayed concurrently with serving.
CPU eligibility/version checks remain outside capture, after verifier commit.
"""
from __future__ import annotations

import math
import time

import torch


class CapturedUpdate:
    def __init__(self, state, replay_hidden, signal, block_size=8):
        if state.left.device.type != "cuda":
            raise ValueError("captured update requires CUDA")
        self.state, self.replay_hidden, self.signal = state, replay_hidden, signal
        self.block_size = block_size
        self.capacity = state.replay_blocks
        self.count = 0
        n = block_size * self.capacity
        device, dtype = state.head_weight.device, state.head_weight.dtype
        self.a = torch.zeros(n, state.features.shape[1], device=device, dtype=dtype)
        self.u = torch.zeros(n, state.before_delta.shape[1], device=device, dtype=dtype)
        self.residual = torch.zeros_like(self.u)
        self.draft = torch.zeros(n, state.head_weight.shape[0], device=device, dtype=dtype)
        self.teacher = torch.zeros_like(self.draft)
        self.valid = torch.zeros(n, device=device, dtype=torch.float32)
        self.gate = torch.ones_like(self.valid)
        self.gate[::block_size] = 0
        self.optimizer = torch.optim.Adam([state.left, state.right], lr=state.lr,
                                           capturable=True, fused=True)
        state.optimizer = self.optimizer
        state.left.grad = torch.zeros_like(state.left)
        state.right.grad = torch.zeros_like(state.right)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.enable_grad():
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side), torch.enable_grad():
            self._body()
        self.reset()

    def _body(self):
        s = self.state
        self.optimizer.zero_grad(set_to_none=False)
        hidden = self.replay_hidden(self.a, self.u, self.residual, s.left, s.right,
                                     s.norm_weight, s.eps, s.n_groups, self.gate)
        loss, gradient = self.signal(self.draft, self.teacher, s.head_weight, self.valid)
        hidden.backward(gradient)
        self.hidden = hidden.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_([s.left, s.right], 1.0,
                                                   error_if_nonfinite=False, foreach=False)
        self.optimizer.step()
        with torch.no_grad():
            finite = torch.isfinite(grad_norm) & torch.isfinite(loss)
            finite = finite & torch.isfinite(s.left).all() & torch.isfinite(s.right).all()
            self.metrics = torch.stack((loss, grad_norm, s.right.norm(), finite.float()))
            s.serve_left.copy_(s.left)
            s.serve_right.copy_(s.right)

    @torch.no_grad()
    def reset(self):
        s = self.state
        s.left.copy_(s.initial_left)
        s.right.zero_()
        s.serve_left.copy_(s.left)
        s.serve_right.zero_()
        # Never replace these tensors: captured Adam reads their exact addresses.
        for parameter in (s.left, s.right):
            for value in self.optimizer.state[parameter].values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
            parameter.grad.zero_()
        self.clear()

    @torch.no_grad()
    def clear(self):
        self.count = 0
        self.valid.zero_()

    @torch.no_grad()
    def remember(self, rows, block_size):
        if block_size != self.block_size:
            raise ValueError("captured update block shape cannot change")
        slot = self.count % self.capacity
        start, end = slot * block_size, (slot + 1) * block_size
        s = self.state
        self.a[start:end].copy_(s.features[:block_size])
        self.u[start:end].copy_(s.before_delta[:block_size])
        self.residual[start:end].copy_(s.residual[:block_size])
        self.draft[start + 1:end].copy_(s.last_draft)
        self.teacher[start + 1:end].copy_(s.last_teacher)
        self.valid[start:end].zero_()
        self.valid[start + 1:start + rows + 1].fill_(1)
        self.count += 1

    def update(self, *, audit=False):
        s = self.state
        if not s.examples or any(e["version"] != s.version for e in s.examples):
            raise RuntimeError("missing or stale feedback for captured update")
        started = time.perf_counter()
        self.graph.replay()
        # Single blocking read, before any following draft can observe the version.
        values = self.metrics.tolist()
        if values[3] != 1.0 or not all(math.isfinite(v) for v in values):
            self.reset()
            s.examples = []
            s.last_teacher = s.last_draft = None
            raise FloatingPointError("non-finite captured update; reset and abort before decoding")
        event = dict(cycle=s.cycles, version=s.version + 1,
                     rows=sum(e["rows"] for e in s.examples), replay_blocks=len(s.examples),
                     kl_before=values[0], grad_norm=values[1], right_norm=values[2])
        if audit:
            with torch.no_grad():
                selected = self.valid.bool()
                old_logits = torch.nn.functional.linear(self.hidden, s.head_weight)[selected]
                new_logits = s.logits(self.a, self.u, self.residual, self.gate)[selected]
                teacher = self.teacher[selected].float().softmax(-1)
                extra = torch.stack([
                    (old_logits - self.draft[selected]).abs().max().float(),
                    (new_logits - old_logits).abs().max().float(),
                    torch.nn.functional.kl_div(new_logits.float().log_softmax(-1), teacher, reduction="batchmean"),
                ]).tolist()
            event.update(replay_max_logit_error=extra[0], same_features_logit_change=extra[1], kl_after=extra[2])
        self.clear()
        s.examples = []
        s.version += 1
        event["seconds"] = time.perf_counter() - started
        s.update_seconds += event["seconds"]
        s.events.append(event)

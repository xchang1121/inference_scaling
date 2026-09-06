"""Online convex mixing of sparse proposal temperatures on a frozen drafter."""

from dataclasses import dataclass
import math
import time

import torch


def project_simplex(values):
    """Euclidean projection of each last-axis vector onto nonnegative unit sum."""
    ordered = values.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1) - 1
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    rho = (ordered - cumulative / ranks > 0).sum(-1, keepdim=True) - 1
    threshold = cumulative.gather(-1, rho) / (rho + 1).to(values.dtype)
    return (values - threshold).clamp_min(0)


@dataclass
class CalibrationFeedback:
    indices: torch.Tensor
    experts: torch.Tensor
    mixed: torch.Tensor
    baseline: torch.Tensor


def mix_rows(baseline, weights, powers, identity_index, top_k):
    """Pure tensor proposal map, shared by eager and graph execution."""
    compact, indices = baseline[1:].topk(min(top_k, baseline.shape[-1]), dim=-1)
    experts = compact[:, None, :].pow(powers)
    experts = experts / experts.sum(-1, keepdim=True).clamp_min(torch.finfo(experts.dtype).tiny)
    experts[:, identity_index] = compact
    mixed = (experts * weights[:len(compact), :, None]).sum(1)
    q = torch.zeros_like(baseline)
    q[0] = baseline[0]
    q[1:].scatter_(-1, indices, mixed)
    return q, CalibrationFeedback(indices, experts, mixed, compact)


class OverlapMix:
    """Per-depth simplex updates minimize TV on the actually reached prefixes.

    Temperatures act on the original truncated proposal's positive support.
    The identity expert is exact. The target law and clean root are unchanged.
    Stored proposal snapshots are consumed before an online update.
    """

    def __init__(self, block_size, top_k, *, temperatures=(.5, .75, 1., 1.25, 1.5),
                 learning_rate=.5, interval=8, adaptive=True, fixed_temperature=1.,
                 diagnostics=False, device="cpu", dtype=torch.float32):
        if (type(block_size) is not int or block_size < 2 or type(top_k) is not int or top_k < 1
                or not temperatures or len(set(temperatures)) != len(temperatures)
                or any(not math.isfinite(t) or t <= 0 for t in temperatures)
                or 1. not in temperatures or fixed_temperature not in temperatures
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or type(interval) is not int or interval < 1):
            raise ValueError("block >=2, positive bounded support, distinct positive temperatures including identity required")
        self.block_size, self.top_k = block_size, top_k
        self.temperatures = tuple(temperatures)
        self.learning_rate, self.interval, self.adaptive = learning_rate, interval, adaptive
        self.diagnostics = diagnostics
        self.fixed_index = self.temperatures.index(fixed_temperature)
        self.identity_index = self.temperatures.index(1.)
        self.weights = torch.zeros(block_size - 1, len(temperatures), device=device, dtype=dtype)
        self.powers = self.weights.new_tensor(temperatures).reciprocal()[None, :, None]
        self.gradient = torch.zeros_like(self.weights)
        self.counts = self.weights.new_zeros(block_size - 1, 1)
        self.steps = torch.zeros_like(self.counts)
        self.totals = self.weights.new_zeros(block_size - 1, len(temperatures) + 3)
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.weights.zero_()
        self.weights[:, self.fixed_index] = 1
        self.gradient.zero_()
        self.counts.zero_()
        self.steps.zero_()
        self.totals.zero_()
        self.updates = self.feedback_blocks = 0
        self.update_seconds = 0.

    @torch.no_grad()
    def propose(self, baseline):
        if (baseline.ndim != 2 or not 2 <= len(baseline) <= self.block_size
                or baseline.device != self.weights.device or baseline.dtype != self.weights.dtype):
            raise ValueError("matching floating probability rows with one clean root required")
        return mix_rows(baseline, self.weights, self.powers, self.identity_index, self.top_k)

    @torch.no_grad()
    def observe(self, feedback, teacher):
        n = len(teacher)
        if not n or not (self.adaptive or self.diagnostics):
            return
        if n > len(feedback.indices):
            raise ValueError("teacher feedback must be an actually reached proposal prefix")
        p = teacher.gather(-1, feedback.indices[:n])
        experts, q = feedback.experts[:n], feedback.mixed[:n]
        if self.diagnostics:
            expert_tv = 1 - torch.minimum(p[:, None, :], experts).sum(-1)
            tv = 1 - torch.minimum(p, q).sum(-1)
            # Columns: count, actual pre-update mixture TV, support mass, expert TVs.
            self.totals[:n] += torch.cat((torch.ones_like(tv[:, None]), tv[:, None],
                                         p.sum(-1, keepdim=True), expert_tv), -1)
        self.feedback_blocks += 1
        if self.adaptive:
            subgradient = -(experts * (q < p)[:, None, :]).sum(-1)
            self.gradient[:n] += subgradient
            self.counts[:n] += 1
            if self.feedback_blocks % self.interval == 0:
                self.step()

    @torch.no_grad()
    def step(self):
        if self.weights.is_cuda:
            torch.cuda.synchronize(self.weights.device)
        start = time.perf_counter()
        valid = self.counts > 0
        gradient = self.gradient / self.counts.clamp_min(1)
        rate = self.learning_rate / (self.steps + 1).sqrt()
        updated = project_simplex(self.weights - rate * gradient)
        self.weights.copy_(torch.where(valid, updated, self.weights))
        self.steps += valid
        self.gradient.zero_()
        self.counts.zero_()
        self.updates += 1
        if self.weights.is_cuda:
            torch.cuda.synchronize(self.weights.device)
        self.update_seconds += time.perf_counter() - start

    def metrics(self):
        values = self.totals.detach().cpu()
        count = values[:, :1].clamp_min(1)
        return {"temperatures": self.temperatures, "weights": self.weights.detach().cpu().tolist(),
                "depth_observations": values[:, 0].tolist(),
                "depth_preupdate_tv": (values[:, 1] / count[:, 0]).tolist(),
                "depth_support_mass": (values[:, 2] / count[:, 0]).tolist(),
                "depth_expert_tv": (values[:, 3:] / count).tolist(),
                "updates": self.updates, "feedback_blocks": self.feedback_blocks,
                "update_seconds": self.update_seconds}

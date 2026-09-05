"""Synchronous continuation of the actual draft adapter after committed rounds."""

from collections import deque
from dataclasses import dataclass
import math
import time

import torch

from .distillation import divergence
from .model import Cache, is_adapter


def synchronize(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class Feedback:
    inputs: torch.Tensor          # [1, seed + noise]
    cache: Cache | None            # base-only prefix before the seed
    teacher_logits: torch.Tensor  # [valid noise positions, vocabulary]
    valid: int                    # accepted positions + first rejection, if any

    def detached(self):
        cache = None if self.cache is None else tuple((k.detach(), v.detach()) for k, v in self.cache)
        return Feedback(self.inputs.detach().clone(), cache,
                        self.teacher_logits.detach().clone(), self.valid)


@dataclass(frozen=True)
class OnlineConfig:
    stride: int = 16
    replay_blocks: int = 4
    learning_rate: float = 1e-4
    loss: str = "l1"
    clip_norm: float = 1.0

    def __post_init__(self):
        if self.stride < 1 or self.replay_blocks < 1:
            raise ValueError("positive update stride and replay capacity required")
        if not all(math.isfinite(v) and v > 0 for v in (self.learning_rate, self.clip_norm)):
            raise ValueError("finite positive optimizer scales required")
        if self.loss not in ("l1", "tv", "forward_kl", "reverse_kl"):
            raise ValueError("unknown online loss")


class OnlineLearner:
    """No hidden second teacher or per-request adapter reset.

    Replay recomputes a differentiable noisy student forward. Verification logits
    are detached targets, not a claim that the full-model backward is free.
    Old prefix KV is base-only; temporary adapter-dependent noisy KV is discarded.
    """

    def __init__(self, model, config=OnlineConfig()):
        self.model, self.config = model, config
        model.train_adapters_only()
        parameters = model.adapter_parameters()
        if not parameters:
            raise ValueError("online continuation needs trainable adapter parameters")
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0)
        self.replay = deque(maxlen=config.replay_blocks)
        self.rounds, self.updates, self.version = 0, 0, 0
        self.update_seconds, self.last_loss = 0.0, None

    def clear_replay(self):
        """Release request-specific KV and targets; keep learned weights/optimizer."""
        self.replay.clear()

    def observe(self, feedback, *, may_update=True):
        self.rounds += 1
        if feedback.valid:
            if feedback.inputs.shape[0] != 1 or not 0 < feedback.valid < feedback.inputs.shape[1]:
                raise ValueError("invalid replay dimensions")
            if feedback.teacher_logits.shape != (feedback.valid, self.model.config.vocab_size):
                raise ValueError("teacher rows must match the valid prefix")
            self.replay.append(feedback.detached())
        if may_update and self.replay and self.rounds % self.config.stride == 0:
            return self.update()
        return None

    def update(self):
        if not self.replay:
            return None
        if any(p.requires_grad for n, p in self.model.named_parameters() if not is_adapter(n)):
            raise RuntimeError("the target/base model must stay frozen")
        synchronize(self.model)
        start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        positions = sum(item.valid for item in self.replay)
        total = 0.0
        # Backward one sample at a time: do not retain R full activation graphs.
        with torch.enable_grad():
            for item in self.replay:
                mask = torch.ones_like(item.inputs, dtype=torch.bool)
                mask[:, 0] = False
                logits = self.model(item.inputs, cache=item.cache, adapter_mask=mask)
                loss = divergence(logits[0, 1:1 + item.valid], item.teacher_logits,
                                  self.config.loss).sum() / positions
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite online loss")
                loss.backward()
                total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.model.adapter_parameters(), self.config.clip_norm,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        synchronize(self.model)
        elapsed = time.perf_counter() - start
        self.updates += 1
        self.version += 1
        self.update_seconds += elapsed
        self.last_loss = total
        return {"loss": total, "seconds": elapsed, "positions": positions, "version": self.version}

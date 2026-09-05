"""Past-only throughput control and deterministic-proposal verification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .torch_sampling import VerificationResult


def verify_target_draws(proposals: Tensor, target_draws: Tensor) -> VerificationResult:
    """Verify delta proposals using independent draws from all target rows.

    target_draws includes one lookahead. The caller must compute its rows on
    [uncached seed, proposals] and use fresh random draws (or greedy argmax).
    The compact payload causes a single host transfer, never one per token.
    """
    if proposals.ndim != 1 or target_draws.ndim != 1:
        raise ValueError("proposals and target_draws must be one-dimensional")
    count = proposals.numel()
    if count < 1 or target_draws.numel() != count + 1:
        raise ValueError("one target draw per proposal plus lookahead is required")
    if proposals.device != target_draws.device:
        raise ValueError("proposals and target draws must share a device")
    if proposals.dtype != torch.long or target_draws.dtype != torch.long:
        raise ValueError("token IDs must use torch.long")
    mismatch = proposals.ne(target_draws[:-1])
    first = torch.where(
        mismatch.any(),
        mismatch.to(torch.long).argmax(),
        torch.full((), count, device=proposals.device, dtype=torch.long),
    )
    packed = torch.cat((target_draws, first.reshape(1))).tolist()
    accepted = int(packed[-1])
    return VerificationResult(
        committed=tuple(packed[: accepted + 1]),
        accepted_spec_tokens=accepted,
        rejected_index=accepted if accepted < count else None,
        used_lookahead=accepted == count,
    )


def tail_after_commit(
    predictions: Tensor,
    *,
    committed_tokens: int,
    refill: bool,
    max_candidates: int,
) -> Tensor:
    """Align verifier row indices with the next real output position."""
    if predictions.ndim != 1 or committed_tokens < 1 or max_candidates < 1:
        raise ValueError("invalid recycling tail bounds")
    start = committed_tokens - int(refill)
    if start > predictions.numel():
        raise ValueError("commit exceeds verifier prediction frontier")
    return predictions[start : start + max_candidates]


@dataclass(frozen=True)
class RecyclingConfig:
    block_size: int = 8
    policy: str = "tps"
    max_recycle_depth: int = 4
    min_candidates: int = 1
    ema_decay: float = 0.8
    throughput_margin: float = 0.05
    exploration_trials: int = 2
    probe_interval: int = 16

    def validate(self) -> None:
        if self.block_size < 2:
            raise ValueError("recycling requires block_size >= 2")
        if self.policy not in {"disabled", "always", "bounded", "tps"}:
            raise ValueError("unknown recycling policy")
        if self.max_recycle_depth < 1 or self.min_candidates < 1:
            raise ValueError("depth and candidate bounds must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must lie in [0, 1)")
        if not math.isfinite(self.throughput_margin) or self.throughput_margin < 0:
            raise ValueError("throughput_margin must be finite and nonnegative")
        if self.exploration_trials < 1 or self.probe_interval < 1:
            raise ValueError("exploration and probe bounds must be positive")


@dataclass
class _ThroughputEstimate:
    observations: int = 0
    tokens_ema: float = 0.0
    seconds_ema: float = 0.0
    total_tokens: int = 0
    total_seconds: float = 0.0
    last_probe: int = 0

    @property
    def tps(self) -> float:
        return self.tokens_ema / self.seconds_ema if self.seconds_ema > 0 else 0.0

    def observe(self, tokens: int, seconds: float, decay: float) -> None:
        weight = decay if self.observations else 0.0
        self.tokens_ema = weight * self.tokens_ema + (1 - weight) * tokens
        self.seconds_ema = weight * self.seconds_ema + (1 - weight) * seconds
        self.observations += 1
        self.total_tokens += tokens
        self.total_seconds += seconds


class RecyclingController:
    """Ratio-of-EMAs controller; empirical policy, not a global optimality claim."""

    def __init__(self, config: RecyclingConfig) -> None:
        config.validate()
        self.config = config
        self.refill = _ThroughputEstimate()
        self.recycle: dict[int, _ThroughputEstimate] = {}
        self.decisions = 0

    @staticmethod
    def bucket(candidates: int) -> int:
        return min(8, 1 << max(0, int(candidates).bit_length() - 1))

    def decide(self, *, candidates: int, depth: int) -> tuple[bool, str]:
        self.decisions += 1
        config = self.config
        if config.policy == "disabled":
            return False, "disabled"
        if candidates < config.min_candidates:
            return False, "empty-or-short-tail"
        if config.policy == "always":
            return True, "always"
        if depth >= config.max_recycle_depth:
            return False, "depth-refill"
        if config.policy == "bounded":
            return True, "bounded"
        estimate = self.recycle.setdefault(self.bucket(candidates), _ThroughputEstimate())
        if not self.refill.observations:
            return False, "refill-uninitialized"
        if estimate.observations < config.exploration_trials:
            estimate.last_probe = self.decisions
            return True, "explore"
        if estimate.tps >= self.refill.tps * (1 + config.throughput_margin):
            estimate.last_probe = self.decisions
            return True, "tps-exploit"
        if self.decisions - estimate.last_probe >= config.probe_interval:
            estimate.last_probe = self.decisions
            return True, "periodic-probe"
        return False, "below-tps-margin"

    def observe(
        self, *, recycle: bool, candidates: int, tokens: int, seconds: float
    ) -> None:
        if tokens < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("feedback requires positive tokens and finite elapsed time")
        if recycle and candidates < 1:
            raise ValueError("a recycle observation must have candidates")
        estimate = (
            self.recycle.setdefault(self.bucket(candidates), _ThroughputEstimate())
            if recycle
            else self.refill
        )
        estimate.observe(tokens, seconds, self.config.ema_decay)

    def snapshot(self) -> dict[str, object]:
        return {
            "decisions": self.decisions,
            "refill": {**asdict(self.refill), "tps_ema": self.refill.tps},
            "recycle": {
                str(key): {**asdict(value), "tps_ema": value.tps}
                for key, value in sorted(self.recycle.items())
            },
        }

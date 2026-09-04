"""Request-local low-rank logit residuals for post-verification adaptation.

The base model and the offline Uno adapter stay frozen.  This module consumes
detached draft hidden states and verifier logits, so optimizer ownership can be
audited independently from the large model.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastResidualConfig:
    rank: int = 8
    alpha: float = 8.0
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    forward_kl_weight: float = 1.0
    tv_weight: float = 0.5
    old_q_weight: float = 0.15
    elastic_weight: float = 1e-6
    max_gradient_norm: float = 1.0
    validation_stride: int = 5
    reset_margin: float = 0.05
    rollback_tolerance: float = 0.01

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive.")
        if self.alpha <= 0 or self.learning_rate <= 0:
            raise ValueError("alpha and learning_rate must be positive.")
        if self.validation_stride < 2:
            raise ValueError("validation_stride must be at least two.")
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive.")
        if self.reset_margin < 0 or self.rollback_tolerance < 0:
            raise ValueError("reset and rollback tolerances cannot be negative.")


@dataclass(frozen=True)
class ResidualFeedback:
    """One verifier row projected onto a fixed draft/target top-k union."""

    hidden: Tensor
    token_ids: Tensor
    base_logits: Tensor
    target_probabilities: Tensor
    old_probabilities: Tensor
    temperature: float
    weight: float = 1.0
    position: int = 0

    def validate(self, *, hidden_size: int, vocabulary_size: int) -> None:
        if self.hidden.ndim != 1 or self.hidden.numel() != hidden_size:
            raise ValueError("feedback hidden state has the wrong shape.")
        support = self.token_ids.numel()
        if self.token_ids.ndim != 1 or support < 2:
            raise ValueError("feedback token_ids must be a rank-one support of size >= 2.")
        if self.token_ids.dtype != torch.long:
            raise ValueError("feedback token_ids must use torch.long.")
        if bool(torch.any((self.token_ids < 0) | (self.token_ids >= vocabulary_size)).item()):
            raise ValueError("feedback token id lies outside the vocabulary.")
        if torch.unique(self.token_ids).numel() != support:
            raise ValueError("feedback token ids must be unique.")
        for name, value in (
            ("base_logits", self.base_logits),
            ("target_probabilities", self.target_probabilities),
            ("old_probabilities", self.old_probabilities),
        ):
            if value.ndim != 1 or value.numel() != support:
                raise ValueError(f"feedback {name} has the wrong shape.")
            if not bool(torch.all(torch.isfinite(value)).item()):
                raise ValueError(f"feedback {name} must be finite.")
        for name, probabilities in (
            ("target", self.target_probabilities),
            ("old", self.old_probabilities),
        ):
            if bool(torch.any(probabilities < 0).item()):
                raise ValueError(f"feedback {name} probabilities cannot be negative.")
            if not math.isclose(float(probabilities.sum().item()), 1.0, abs_tol=2e-5):
                raise ValueError(f"feedback {name} probabilities must sum to one.")
        if self.temperature <= 0:
            raise ValueError("feedback temperature must be positive.")
        if self.weight <= 0:
            raise ValueError("feedback weight must be positive.")


def feedback_from_logits(
    *,
    hidden: Tensor,
    base_logits: Tensor,
    adjusted_logits: Tensor,
    target_logits: Tensor,
    top_k: int,
    temperature: float,
    weight: float = 1.0,
    position: int = 0,
) -> ResidualFeedback:
    """Create a detached top-k-union distillation item.

    Target and old-draft probabilities are normalized on the same fixed union.
    This makes the surrogate differentiable while retaining every token that is
    top-k under either side of the current verifier comparison.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    for name, logits in (
        ("base_logits", base_logits),
        ("adjusted_logits", adjusted_logits),
        ("target_logits", target_logits),
    ):
        if logits.ndim != 1:
            raise ValueError(f"{name} must be rank one.")
        if not bool(torch.all(torch.isfinite(logits)).item()):
            raise ValueError(f"{name} must be finite.")
    if not (base_logits.shape == adjusted_logits.shape == target_logits.shape):
        raise ValueError("base, adjusted, and target logits must share a vocabulary.")
    if hidden.ndim != 1 or not bool(torch.all(torch.isfinite(hidden)).item()):
        raise ValueError("hidden must be a finite rank-one tensor.")

    support_size = min(int(top_k), int(base_logits.numel()))
    draft_ids = torch.topk(adjusted_logits, support_size).indices
    target_ids = torch.topk(target_logits, support_size).indices
    token_ids = torch.unique(torch.cat((draft_ids, target_ids)), sorted=True)
    selected_target = target_logits.index_select(0, token_ids).float() / temperature
    selected_old = adjusted_logits.index_select(0, token_ids).float() / temperature
    return ResidualFeedback(
        hidden=hidden.detach().float().clone(),
        token_ids=token_ids.detach().clone(),
        base_logits=base_logits.index_select(0, token_ids).detach().float().clone(),
        target_probabilities=torch.softmax(selected_target, dim=0).detach().clone(),
        old_probabilities=torch.softmax(selected_old, dim=0).detach().clone(),
        temperature=float(temperature),
        weight=float(weight),
        position=int(position),
    )


class FastResidualHead(nn.Module):
    """A zero-initialized low-rank correction from hidden states to logits."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocabulary_size: int,
        rank: int,
        alpha: float,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or vocabulary_size < 2 or rank < 1 or alpha <= 0:
            raise ValueError("invalid fast residual dimensions or alpha.")
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.rank = int(rank)
        self.scale = float(alpha / rank)
        self.down = nn.Linear(hidden_size, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, vocabulary_size, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        nn.init.zeros_(self.up.weight)

    def features(self, hidden: Tensor) -> Tensor:
        return self.down(hidden.float())

    def forward(self, hidden: Tensor) -> Tensor:
        return self.up(self.features(hidden)) * self.scale

    def selected(self, hidden: Tensor, token_ids: Tensor) -> Tensor:
        if hidden.ndim != 1 or hidden.numel() != self.hidden_size:
            raise ValueError("selected correction expects one hidden-state row.")
        token_ids = token_ids.to(device=self.up.weight.device, dtype=torch.long)
        selected_up = self.up.weight.index_select(0, token_ids)
        return F.linear(self.features(hidden), selected_up) * self.scale


@dataclass(frozen=True)
class LossReport:
    objective: float
    forward_kl: float
    total_variation: float
    old_q_kl: float


@dataclass(frozen=True)
class FastUpdateReport:
    items: int
    training_items: int
    validation_items: int
    reset_to_offline: bool
    rolled_back: bool
    validation_before: LossReport
    validation_after: LossReport
    static_validation: LossReport
    training_objective: float
    gradient_norm: float
    fast_weight_l2: float


class FastResidualLearner:
    """Transactional optimizer for one request-local residual head."""

    def __init__(self, head: FastResidualHead, config: FastResidualConfig) -> None:
        config.validate()
        if head.rank != config.rank or not math.isclose(
            head.scale,
            config.alpha / config.rank,
        ):
            raise ValueError("fast residual head and learner configuration disagree.")
        self.head = head
        self.config = config
        self._offline_state = self._clone_head_state()
        self.optimizer = self._new_optimizer()

    def _new_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _clone_head_state(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.head.state_dict().items()}

    def reset_to_offline(self) -> None:
        self.head.load_state_dict(self._offline_state)
        self.optimizer = self._new_optimizer()

    def decay_toward_offline(self, factor: float) -> None:
        if not 0 <= factor <= 1:
            raise ValueError("decay factor must lie in [0, 1].")
        with torch.no_grad():
            for name, value in self.head.state_dict().items():
                baseline = self._offline_state[name].to(value.device)
                value.copy_(baseline + factor * (value - baseline))

    def corrected_logits(self, hidden: Tensor, base_logits: Tensor) -> Tensor:
        return base_logits.float() + self.head(hidden)

    def _validate_items(self, items: Sequence[ResidualFeedback]) -> None:
        if not items:
            raise ValueError("fast residual update requires feedback items.")
        device = next(self.head.parameters()).device
        for item in items:
            item.validate(
                hidden_size=self.head.hidden_size,
                vocabulary_size=self.head.vocabulary_size,
            )
            tensors = (
                item.hidden,
                item.token_ids,
                item.base_logits,
                item.target_probabilities,
                item.old_probabilities,
            )
            if any(tensor.device != device for tensor in tensors):
                raise ValueError("feedback and fast residual head must share one device.")

    def _loss_tensors(
        self,
        items: Sequence[ResidualFeedback],
        *,
        include_old_q: bool,
        zero_correction: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weighted_kl = torch.zeros((), device=items[0].hidden.device)
        weighted_tv = torch.zeros_like(weighted_kl)
        weighted_old = torch.zeros_like(weighted_kl)
        total_weight = 0.0
        for item in items:
            if zero_correction:
                correction = torch.zeros_like(item.base_logits)
            else:
                correction = self.head.selected(item.hidden, item.token_ids)
            log_q = torch.log_softmax(
                (item.base_logits + correction) / item.temperature,
                dim=0,
            )
            q = log_q.exp()
            target = item.target_probabilities
            old = item.old_probabilities
            forward_kl = torch.sum(target * (torch.log(target.clamp_min(1e-12)) - log_q))
            total_variation = 0.5 * torch.sum(torch.abs(target - q))
            old_q_kl = torch.sum(old * (torch.log(old.clamp_min(1e-12)) - log_q))
            weighted_kl = weighted_kl + item.weight * forward_kl
            weighted_tv = weighted_tv + item.weight * total_variation
            weighted_old = weighted_old + item.weight * old_q_kl
            total_weight += item.weight
        denominator = max(total_weight, 1e-12)
        mean_kl = weighted_kl / denominator
        mean_tv = weighted_tv / denominator
        mean_old = weighted_old / denominator
        objective = (
            self.config.forward_kl_weight * mean_kl
            + self.config.tv_weight * mean_tv
            + (self.config.old_q_weight * mean_old if include_old_q else 0.0)
        )
        return objective, mean_kl, mean_tv, mean_old

    def evaluate(
        self,
        items: Sequence[ResidualFeedback],
        *,
        zero_correction: bool = False,
    ) -> LossReport:
        self._validate_items(items)
        with torch.no_grad():
            objective, forward_kl, tv, old_q_kl = self._loss_tensors(
                items,
                include_old_q=False,
                zero_correction=zero_correction,
            )
        return LossReport(
            objective=float(objective.item()),
            forward_kl=float(forward_kl.item()),
            total_variation=float(tv.item()),
            old_q_kl=float(old_q_kl.item()),
        )

    def fast_weight_l2(self) -> float:
        total = torch.zeros((), device=next(self.head.parameters()).device)
        with torch.no_grad():
            for name, value in self.head.state_dict().items():
                baseline = self._offline_state[name].to(value.device)
                total = total + torch.sum((value - baseline) ** 2)
        return float(torch.sqrt(total).item())

    def update(self, items: Sequence[ResidualFeedback]) -> FastUpdateReport:
        self._validate_items(items)
        validation = list(items[:: self.config.validation_stride])
        training = [
            item
            for index, item in enumerate(items)
            if index % self.config.validation_stride != 0
        ]
        if not training:
            training = list(items)

        static_validation = self.evaluate(validation, zero_correction=True)
        validation_before = self.evaluate(validation)
        reset = validation_before.objective > (
            static_validation.objective * (1.0 + self.config.reset_margin) + 1e-8
        )
        if reset:
            self.reset_to_offline()
            validation_before = self.evaluate(validation)

        head_snapshot = self._clone_head_state()
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad(set_to_none=True)
        objective, _, _, _ = self._loss_tensors(
            training,
            include_old_q=True,
            zero_correction=False,
        )
        if self.config.elastic_weight:
            objective = objective + self.config.elastic_weight * torch.mean(
                self.head.up.weight**2
            )
        rolled_back = False
        gradient_norm = math.nan
        if bool(torch.isfinite(objective).item()):
            objective.backward()
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.head.parameters(),
                self.config.max_gradient_norm,
                error_if_nonfinite=False,
            )
            gradient_norm = float(gradient_norm_tensor.item())
            if math.isfinite(gradient_norm):
                self.optimizer.step()
            else:
                rolled_back = True
        else:
            rolled_back = True

        validation_after = self.evaluate(validation)
        if (
            not math.isfinite(validation_after.objective)
            or validation_after.objective
            > validation_before.objective * (1.0 + self.config.rollback_tolerance) + 1e-8
        ):
            rolled_back = True
        if rolled_back:
            self.head.load_state_dict(head_snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            validation_after = self.evaluate(validation)

        return FastUpdateReport(
            items=len(items),
            training_items=len(training),
            validation_items=len(validation),
            reset_to_offline=reset,
            rolled_back=rolled_back,
            validation_before=validation_before,
            validation_after=validation_after,
            static_validation=static_validation,
            training_objective=float(objective.detach().item()),
            gradient_norm=gradient_norm,
            fast_weight_l2=self.fast_weight_l2(),
        )


def assert_optimizer_isolated(
    *,
    base_model: nn.Module,
    head: FastResidualHead,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    """Fail unless only fast-head parameters are trainable and optimized."""

    base_parameters = list(base_model.parameters())
    trainable_base = [parameter for parameter in base_parameters if parameter.requires_grad]
    if trainable_base:
        raise RuntimeError(f"base model has {len(trainable_base)} trainable parameters.")
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head_parameters}
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if optimizer_ids != head_ids or len(optimizer_parameters) != len(head_parameters):
        raise RuntimeError("optimizer parameter set is not exactly the fast residual head.")
    base_ids = {id(parameter) for parameter in base_parameters}
    overlap = base_ids & optimizer_ids
    if overlap:
        raise RuntimeError("optimizer unexpectedly owns base-model parameters.")
    return {
        "base_parameter_tensors": len(base_parameters),
        "trainable_base_parameter_tensors": 0,
        "fast_parameter_tensors": len(head_parameters),
        "fast_trainable_parameters": sum(parameter.numel() for parameter in head_parameters),
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "base_optimizer_overlap": 0,
    }

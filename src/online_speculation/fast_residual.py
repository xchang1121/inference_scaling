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


def feedback_batch_from_logits(
    *,
    hidden_rows: Tensor,
    base_logits: Tensor,
    adjusted_logits: Tensor,
    target_logits: Tensor,
    top_k: int,
    temperature: float,
    weights: Sequence[float],
    positions: Sequence[int] | None = None,
) -> list[ResidualFeedback]:
    """Vectorized validation/top-k followed by variable-size union packing."""

    if hidden_rows.ndim != 2:
        raise ValueError("hidden_rows must have shape [rows, hidden].")
    if base_logits.ndim != 2:
        raise ValueError("base_logits must have shape [rows, vocabulary].")
    if not (
        base_logits.shape == adjusted_logits.shape == target_logits.shape
        and hidden_rows.size(0) == base_logits.size(0)
    ):
        raise ValueError("batched feedback tensors have incompatible shapes.")
    rows, vocabulary_size = base_logits.shape
    if len(weights) != rows:
        raise ValueError("one feedback weight is required per row.")
    if positions is None:
        positions = list(range(rows))
    if len(positions) != rows:
        raise ValueError("one feedback position is required per row.")
    if top_k < 1 or temperature <= 0:
        raise ValueError("top_k and temperature must be positive.")
    finite = (
        torch.isfinite(hidden_rows).all()
        & torch.isfinite(base_logits).all()
        & torch.isfinite(adjusted_logits).all()
        & torch.isfinite(target_logits).all()
    )
    if not bool(finite.item()):
        raise ValueError("batched feedback tensors must be finite.")

    support_size = min(int(top_k), int(vocabulary_size))
    draft_top_ids = torch.topk(adjusted_logits, support_size, dim=-1).indices
    target_top_ids = torch.topk(target_logits, support_size, dim=-1).indices
    feedback = []
    for row, (weight, position) in enumerate(zip(weights, positions)):
        if weight <= 0:
            continue
        token_ids = torch.unique(
            torch.cat((draft_top_ids[row], target_top_ids[row])),
            sorted=True,
        )
        selected_target = target_logits[row].index_select(0, token_ids).float()
        selected_old = adjusted_logits[row].index_select(0, token_ids).float()
        feedback.append(
            ResidualFeedback(
                hidden=hidden_rows[row].detach().float().clone(),
                token_ids=token_ids.detach().clone(),
                base_logits=(
                    base_logits[row].index_select(0, token_ids).detach().float().clone()
                ),
                target_probabilities=(
                    torch.softmax(selected_target / temperature, dim=0).detach().clone()
                ),
                old_probabilities=(
                    torch.softmax(selected_old / temperature, dim=0).detach().clone()
                ),
                temperature=float(temperature),
                weight=float(weight),
                position=int(position),
            )
        )
    return feedback


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
        numeric_checks = []
        for item in items:
            if item.hidden.ndim != 1 or item.hidden.numel() != self.head.hidden_size:
                raise ValueError("feedback hidden state has the wrong shape.")
            support = item.token_ids.numel()
            if item.token_ids.ndim != 1 or support < 2:
                raise ValueError("feedback support has the wrong shape.")
            if item.token_ids.dtype != torch.long:
                raise ValueError("feedback token_ids must use torch.long.")
            for value in (
                item.base_logits,
                item.target_probabilities,
                item.old_probabilities,
            ):
                if value.ndim != 1 or value.numel() != support:
                    raise ValueError("feedback values do not match their support.")
            if item.temperature <= 0 or item.weight <= 0:
                raise ValueError("feedback temperature and weight must be positive.")
            tensors = (
                item.hidden,
                item.token_ids,
                item.base_logits,
                item.target_probabilities,
                item.old_probabilities,
            )
            if any(tensor.device != device for tensor in tensors):
                raise ValueError("feedback and fast residual head must share one device.")
            sorted_ids = torch.sort(item.token_ids).values
            unique_ids = (
                torch.ones((), device=device, dtype=torch.bool)
                if support == 1
                else torch.all(sorted_ids[1:] != sorted_ids[:-1])
            )
            numeric_checks.extend(
                (
                    torch.isfinite(item.hidden).all(),
                    torch.isfinite(item.base_logits).all(),
                    torch.isfinite(item.target_probabilities).all(),
                    torch.isfinite(item.old_probabilities).all(),
                    torch.all(
                        (item.token_ids >= 0)
                        & (item.token_ids < self.head.vocabulary_size)
                    ),
                    unique_ids,
                    torch.all(item.target_probabilities >= 0),
                    torch.all(item.old_probabilities >= 0),
                    torch.isclose(
                        item.target_probabilities.sum(),
                        torch.ones((), device=device),
                        atol=2e-5,
                    ),
                    torch.isclose(
                        item.old_probabilities.sum(),
                        torch.ones((), device=device),
                        atol=2e-5,
                    ),
                )
            )
        if not bool(torch.stack(numeric_checks).all().item()):
            raise ValueError("feedback numeric invariants failed.")

    def _loss_tensors(
        self,
        items: Sequence[ResidualFeedback],
        *,
        include_old_q: bool,
        zero_correction: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rows = len(items)
        max_support = max(item.token_ids.numel() for item in items)
        device = items[0].hidden.device
        token_ids = torch.zeros((rows, max_support), device=device, dtype=torch.long)
        base_logits = torch.zeros((rows, max_support), device=device)
        target = torch.zeros_like(base_logits)
        old = torch.zeros_like(base_logits)
        support_mask = torch.zeros(
            (rows, max_support),
            device=device,
            dtype=torch.bool,
        )
        for row, item in enumerate(items):
            support = item.token_ids.numel()
            token_ids[row, :support] = item.token_ids
            base_logits[row, :support] = item.base_logits
            target[row, :support] = item.target_probabilities
            old[row, :support] = item.old_probabilities
            support_mask[row, :support] = True
        hidden = torch.stack([item.hidden for item in items])
        temperatures = torch.tensor(
            [item.temperature for item in items],
            device=device,
        ).unsqueeze(1)
        weights = torch.tensor([item.weight for item in items], device=device)
        if zero_correction:
            correction = torch.zeros_like(base_logits)
        else:
            features = self.head.features(hidden)
            selected_up = self.head.up.weight[token_ids]
            correction = torch.einsum("nr,nmr->nm", features, selected_up)
            correction = correction * self.head.scale
        adjusted = ((base_logits + correction) / temperatures).masked_fill(
            ~support_mask,
            -torch.inf,
        )
        log_q = torch.log_softmax(adjusted, dim=1)
        q = torch.where(support_mask, log_q.exp(), 0.0)
        safe_log_q = torch.where(support_mask, log_q, 0.0)
        forward_kl_rows = torch.sum(
            target * (torch.log(target.clamp_min(1e-12)) - safe_log_q),
            dim=1,
        )
        tv_rows = 0.5 * torch.sum(torch.abs(target - q), dim=1)
        old_q_kl_rows = torch.sum(
            old * (torch.log(old.clamp_min(1e-12)) - safe_log_q),
            dim=1,
        )
        denominator = weights.sum().clamp_min(1e-12)
        mean_kl = torch.sum(weights * forward_kl_rows) / denominator
        mean_tv = torch.sum(weights * tv_rows) / denominator
        mean_old = torch.sum(weights * old_q_kl_rows) / denominator
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

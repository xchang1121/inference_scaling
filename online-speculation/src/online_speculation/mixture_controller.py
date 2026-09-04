"""Scalar verifier-feedback controller for static-anchored proposal mixtures."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmaMixtureConfig:
    max_candidate_weight: float = 0.25
    evaluation_interval: int = 4
    warmup_observations: int = 2
    ema_decay: float = 0.75
    activation_margin: float = 0.0005
    deactivation_margin: float = 0.0005

    def validate(self) -> None:
        scalar_values = (
            self.max_candidate_weight,
            self.ema_decay,
            self.activation_margin,
            self.deactivation_margin,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("controller scalar parameters must be finite.")
        if not 0.0 < self.max_candidate_weight <= 1.0:
            raise ValueError("max_candidate_weight must lie in (0, 1].")
        if not isinstance(self.evaluation_interval, int) or not isinstance(
            self.warmup_observations, int
        ):
            raise ValueError("controller intervals and warmup must be integers.")
        if self.evaluation_interval < 1 or self.warmup_observations < 1:
            raise ValueError("controller intervals and warmup must be positive.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1).")
        if self.activation_margin < 0.0 or self.deactivation_margin < 0.0:
            raise ValueError("controller margins cannot be negative.")


@dataclass(frozen=True)
class MixtureControllerEvent:
    cycle: int
    observation: int
    instantaneous_advantage: float
    ema_advantage: float
    previous_weight: float
    next_weight: float
    action: str


class VerifierEmaMixtureController:
    """Gate a capped candidate mixture using only past verifier evidence.

    Positive advantage means the candidate has lower filtered TV than static on
    the just-verified rows. Any action applies only to the next proposal cycle.
    """

    def __init__(self, config: EmaMixtureConfig) -> None:
        config.validate()
        self.config = config
        self.weight = 0.0
        self.observations = 0
        self.ema_advantage: float | None = None
        self.events: list[MixtureControllerEvent] = []

    def should_evaluate(self, cycle: int) -> bool:
        if cycle < 1:
            raise ValueError("cycle numbers are one-based and positive.")
        return cycle % self.config.evaluation_interval == 0

    def observe(self, *, cycle: int, advantage: float) -> MixtureControllerEvent:
        if not self.should_evaluate(cycle):
            raise ValueError("controller observation is off its evaluation schedule.")
        if not math.isfinite(advantage):
            raise ValueError("controller advantage must be finite.")
        self.observations += 1
        if self.ema_advantage is None:
            self.ema_advantage = float(advantage)
        else:
            self.ema_advantage = self.config.ema_decay * self.ema_advantage + (
                1.0 - self.config.ema_decay
            ) * float(advantage)

        previous = self.weight
        action = (
            "warmup" if self.observations < self.config.warmup_observations else "hold"
        )
        if self.observations >= self.config.warmup_observations:
            if (
                self.weight == 0.0
                and self.ema_advantage > self.config.activation_margin
            ):
                self.weight = self.config.max_candidate_weight
                action = "activate"
            elif (
                self.weight > 0.0
                and self.ema_advantage < -self.config.deactivation_margin
            ):
                self.weight = 0.0
                action = "deactivate"

        event = MixtureControllerEvent(
            cycle=cycle,
            observation=self.observations,
            instantaneous_advantage=float(advantage),
            ema_advantage=self.ema_advantage,
            previous_weight=previous,
            next_weight=self.weight,
            action=action,
        )
        self.events.append(event)
        return event

    def report(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "observations": self.observations,
            "final_weight": self.weight,
            "final_ema_advantage": self.ema_advantage,
            "events": [asdict(event) for event in self.events],
        }

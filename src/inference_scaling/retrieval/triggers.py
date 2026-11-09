"""Retrieval decisions made by the controller, using only the observed past."""
from dataclasses import dataclass
import math
import re
import statistics


@dataclass(frozen=True)
class TriggerConfig:
    kind: str = "periodic"
    interval: int = 128
    minimum_tokens: int = 32
    cooldown: int = 128
    window: int = 32
    surprisal_threshold: float = 1.5
    low_probability: float = .15
    low_fraction: float = .25
    sc_drop_fraction: float = .05
    pattern: str = r"\b(?:uncertain|not sure|need to (?:know|check)|version|idempot\w*|timeout|retry|header|cursor|lease|quota|webhook)\b"

    def __post_init__(self):
        if self.kind not in {"never", "periodic", "low_probability", "surprisal", "sc_drop", "rule", "hybrid"}:
            raise ValueError("unknown retrieval trigger")
        if min(self.interval, self.minimum_tokens, self.cooldown, self.window) <= 0:
            raise ValueError("trigger token limits must be positive")
        if not 0 < self.low_probability <= 1 or not 0 <= self.low_fraction <= 1 or not 0 <= self.sc_drop_fraction < 1:
            raise ValueError("invalid trigger probability threshold")
        if not math.isfinite(self.surprisal_threshold) or self.surprisal_threshold < 0:
            raise ValueError("invalid surprisal threshold")
        re.compile(self.pattern)

    def decide(self, *, generated: int, last_attempt: int, text: str,
               logprobs: list[float], statistics_values: list[float]) -> tuple[bool, dict]:
        values = logprobs[-self.window:]
        sc = [v for v in statistics_values if math.isfinite(v)]
        mean_sc = statistics.fmean(sc[-self.window:]) if sc else None
        baseline = statistics.fmean(sc[:self.window]) if len(sc) >= 2*self.window else None
        signals = {"mean_surprisal": -statistics.fmean(values) if values else None,
                   "low_probability_fraction": statistics.fmean(v < math.log(self.low_probability) for v in values) if values else None,
                   "mean_self_certainty": mean_sc, "baseline_self_certainty": baseline,
                   "rule_match": bool(re.search(self.pattern, text[-800:], re.IGNORECASE))}
        ready = generated >= self.minimum_tokens and (last_attempt < 0 or generated-last_attempt >= self.cooldown)
        low = bool(values) and signals["low_probability_fraction"] >= self.low_fraction
        surprise = bool(values) and signals["mean_surprisal"] >= self.surprisal_threshold
        drop = baseline is not None and mean_sc is not None and mean_sc < baseline*(1-self.sc_drop_fraction)
        decision = {"never": False, "periodic": generated >= (self.interval if last_attempt < 0 else last_attempt+self.interval),
                    "low_probability": low, "surprisal": surprise, "sc_drop": drop,
                    "rule": signals["rule_match"], "hybrid": low or surprise or drop or signals["rule_match"]}[self.kind]
        return ready and decision, signals

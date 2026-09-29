"""Think-only budgeted IS for one request, then one ordinary answer from the chosen thought.

The model, engine, base sampling policy, IS planning and grids and the
Consilience reward are those of ``settings/inference.json``; the request's
effort profile sets the template's reasoning effort, the output limit, the
forward-token budget and the wall-clock limit. The budget grows to the least
one that lets the planner finish a long prompt, so a request is never refused.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from inference_scaling.app.ar import confidence_reward, confidence_statistic, joint_budget_config
from inference_scaling.arllm.algorithms.joint_budget_is import run_joint_budget_is
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.serve.protocol import Conversation, ProtocolError, Reply, parse_tool_calls
from inference_scaling.shared.model.generation import generation_budget
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True)
class Prepared:
    prompt: TokenSequence
    effort: str
    profile: Mapping[str, Any]
    maximum: int
    tools: list[dict[str, Any]] | None


class Reasoner:
    def __init__(self, settings: Mapping[str, Any], serve: Mapping[str, Any], backend: Any | None = None) -> None:
        self.ar, self.reward_settings, self.serve = settings["ar"], settings["rewards"]["consilience"], serve
        self.raw: Any = backend
        self._seeds = itertools.count()
        self._seed_lock = threading.Lock()

    def load(self) -> None:
        if self.raw is None:
            self.raw = load_backend(self.ar["model"], self.ar["engine"], seed=int(self.serve["server"]["seed"]), logprobs=0)

    def close(self) -> None:
        close_backend(self.raw)
        self.raw = None

    def effort(self, requested: str | None) -> str:
        name = requested or self.serve["default_effort"]
        name = self.serve["effort_aliases"].get(name, name)
        if name not in self.serve["efforts"]:
            raise ProtocolError(f"unknown reasoning effort {requested!r}")
        return name

    def prepare(self, conversation: Conversation) -> Prepared:
        """The rendered prompt and the output limit; cheap enough to answer before reasoning."""

        effort = self.effort(conversation.effort)
        profile = self.serve["efforts"][effort]
        options = {**self.ar["prompt"]["chat_template_kwargs"], "enable_thinking": True,
                   "reasoning_effort": profile["reasoning_effort"]}
        if conversation.tools:
            options["tools"] = conversation.tools
        try:
            rendered = self.raw.tokenizer.apply_chat_template(conversation.messages, tokenize=False,
                                                              add_generation_prompt=True, **options)
            prompt = tuple(self.raw.encode(rendered, add_special_tokens=False))
            requested = min(int(profile["max_new_tokens"]), int(conversation.max_tokens or profile["max_new_tokens"]))
            maximum = int(generation_budget(requested, len(prompt), [self.raw],
                                            context_window=self.ar["engine"]["context_window"])["effective_max_new_tokens"])
        except ValueError as error:
            raise ProtocolError(str(error)) from error
        return Prepared(prompt, effort, profile, maximum, conversation.tools)

    def generate(self, prepared: Prepared) -> Reply:
        with self._seed_lock:
            seeds = SeedStream(int(self.serve["server"]["seed"])).derive("request", next(self._seeds))
        prompt, maximum, raw = prepared.prompt, prepared.maximum, self.raw
        sampling = self.ar["sampling"]
        policy = SamplingConfig(temperature=float(sampling["temperature"]), top_p=float(sampling["top_p"]),
                                top_k=sampling["top_k"], eos_token_id=raw.tokenizer.eos_token_id)
        thinking_format = thinking_format_from_backend(raw, output_settings_from_config(self.ar))
        scope = SamplingScope.from_config(raw, self.ar).for_prompt(prompt)
        if scope.scope == "thinking" and self.reward_settings["scope"] == "full":
            scope = scope.full_fallback("reward_uses_full_sequence")
        backend = StatisticRecorder(scope.wrap(raw, prompt), confidence_statistic("consilience", self.reward_settings))
        reward = confidence_reward("consilience", self.reward_settings, backend, raw, policy, thinking_format)
        # The planner needs at least a length probe and two finished candidates: 2 prompts and 3 outputs.
        budget = max(int(prepared.profile["forward_token_budget"]), 2 * len(prompt) + 3 * maximum)
        config = joint_budget_config(self.ar["algorithms"]["is"], reward, total_length=maximum, forward_token_budget=budget)
        start = time.monotonic()
        result = run_joint_budget_is(backend, prompt, config, reward.batch, SeedStream(seeds), sampling=policy,
                                     deadline=start + float(prepared.profile["max_seconds"]))
        tokens, info = scope.finish(raw, prompt, result.token_ids, max_new_tokens=maximum, sampling=policy,
                                    seed=SeedStream(seeds).derive("final-content"))
        # An unfinished thought has no answer; it is returned as the reasoning.
        complete = info["thinking_status"] == "complete"
        thinking, content = (info["thinking_text"], info["content_text"]) if complete else (info["content_text"], "")
        content, calls = parse_tool_calls(content, prepared.tools) if prepared.tools else (content.strip(), ())
        finish: Literal["stop", "tool_calls", "length"] = (
            "length" if not info["ended_by_eos"] else "tool_calls" if calls else "stop")
        return Reply(thinking.strip(), content, calls, finish, len(prompt), len(tokens), {
            "effort": prepared.effort, "steps": len(result.steps), "stopping_reason": result.stopping_reason,
            "forward_token_budget": budget, "forward_tokens": result.actual_forward_tokens,
            "sampling_scope": info["sampling_scope"], "seconds": round(time.monotonic() - start, 3)})


__all__ = ["Prepared", "Reasoner"]

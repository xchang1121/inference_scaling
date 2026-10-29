"""Think-only conditional IS for one request, then one ordinary answer from the chosen thought.

The model, engine, base sampling policy, fixed IS parameters and the
Consilience reward and length policy are those of ``settings/inference.json``.
Requests may select an effort or a thinking budget, disable thinking, and cap
the combined output length. An optional
wall-clock limit is checked only between complete IS steps; the default uses
logical stopping so execution speed cannot change the number of IS moves.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from inference_scaling.app.ar import conditional_is_config, confidence_reward, confidence_statistic
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.thinking_budget import budgeted_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.serve.protocol import Conversation, ProtocolError, Reply, parse_tool_calls
from inference_scaling.shared.model.generation import generation_budget
from inference_scaling.shared.model.prompting import render_prompt
from inference_scaling.shared.model.thinking import resolve_thinking
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True)
class Prepared:
    prompt: TokenSequence
    effort: str | None
    budget: Mapping[str, Any]
    maximum: int
    tools: list[dict[str, Any]] | None
    ar: Mapping[str, Any]


class Reasoner:
    def __init__(self, settings: Mapping[str, Any], backend: Any | None = None) -> None:
        self.ar, self.reward_settings, self.serve = settings["ar"], settings["rewards"]["consilience"], settings["serve"]
        self.seed = int(settings["run"]["seed"])
        self.raw: Any = backend
        self._seeds = itertools.count()
        self._seed_lock = threading.Lock()

    def load(self) -> None:
        if self.raw is None:
            self.raw = load_backend(self.ar["model"], self.ar["engine"], seed=self.seed, logprobs=0)

    def close(self) -> None:
        close_backend(self.raw)
        self.raw = None

    def prepare(self, conversation: Conversation) -> Prepared:
        """The rendered prompt and the output limit; cheap enough to answer before reasoning."""

        try:
            ar, thinking = resolve_thinking(self.ar, effort=conversation.effort,
                                           budget_tokens=conversation.thinking_budget, enabled=conversation.enable_thinking)
            options = dict(ar["prompt"]["chat_template_kwargs"])
            if conversation.tools:
                options["tools"] = conversation.tools
            messages = list(conversation.messages)
            if ar["prompt"]["system"] and not any(message["role"] == "system" for message in messages):
                messages.insert(0, {"role": "system", "content": ar["prompt"]["system"]})
            rendered = render_prompt(self.raw.tokenizer, messages,
                                     {**ar, "prompt": {**ar["prompt"], "chat_template_kwargs": options}})
            prompt = tuple(self.raw.encode(rendered, add_special_tokens=False))
            configured = ar["max_new_tokens"]
            limits = [value for value in (configured, conversation.max_tokens) if value is not None]
            requested = min(limits) if limits else None
            budget = generation_budget(requested, len(prompt), [self.raw],
                                       context_window=ar["engine"]["context_window"], thinking=thinking)
            budget.update(configured_max_new_tokens=configured, client_max_tokens=conversation.max_tokens)
            maximum = int(budget["effective_max_new_tokens"])
            # Validate support before starting the HTTP response or generation.
            budgeted_backend(self.raw, prompt, thinking_format_from_backend(self.raw, output_settings_from_config(ar)), budget)
        except ValueError as error:
            raise ProtocolError(str(error)) from error
        return Prepared(prompt, thinking["reasoning_effort"], budget, maximum, conversation.tools, ar)

    def generate(self, prepared: Prepared) -> Reply:
        with self._seed_lock:
            seeds = SeedStream(self.seed).derive("request", next(self._seeds))
        prompt, maximum, raw, ar = prepared.prompt, prepared.maximum, self.raw, prepared.ar
        sampling = ar["sampling"]
        policy = SamplingConfig(temperature=float(sampling["temperature"]), top_p=float(sampling["top_p"]),
                                top_k=sampling["top_k"], eos_token_id=raw.tokenizer.eos_token_id)
        thinking_format = thinking_format_from_backend(raw, output_settings_from_config(ar))
        bounded = budgeted_backend(raw, prompt, thinking_format, dict(prepared.budget))
        scope = SamplingScope.from_config(raw, ar).for_prompt(prompt)
        if scope.scope == "thinking" and self.reward_settings["scope"] == "full":
            scope = scope.full_fallback("reward_uses_full_sequence")
        backend = StatisticRecorder(scope.wrap(bounded, prompt), confidence_statistic("consilience", self.reward_settings))
        reward = confidence_reward("consilience", self.reward_settings, backend, thinking_format)
        config = conditional_is_config(ar["algorithms"]["is"], reward, total_length=maximum, backend=raw)
        start = time.monotonic()
        limit = self.serve["max_seconds"]
        deadline = None if limit is None else start + float(limit)
        result = run_conditional_is(backend, prompt, config, reward.batch, SeedStream(seeds),
                                    sampling=policy, deadline=deadline, prefix_reward=reward.model.cached_prefix)
        tokens, info = scope.finish(bounded, prompt, result.token_ids, max_new_tokens=maximum, sampling=policy,
                                    seed=SeedStream(seeds).derive("final-content"))
        # An unfinished thought has no answer; it is returned as the reasoning.
        thinking, content = info["thinking_text"], info["content_text"]
        if info["thinking_status"] == "incomplete":
            thinking, content = content, ""
        content, calls = parse_tool_calls(content, prepared.tools) if prepared.tools else (content.strip(), ())
        finish: Literal["stop", "tool_calls", "length"] = (
            "length" if not info["ended_by_eos"] else "tool_calls" if calls else "stop")
        return Reply(thinking.strip(), content, calls, finish, len(prompt), len(tokens), {
            "effort": prepared.effort, "steps": len(result.steps), "stopping_reason": result.stopping_reason,
            "generation_budget": dict(prepared.budget), "max_seconds": limit,
            "candidate_count": config.candidate_count, "rollout_count": config.rollout_count,
            "block_size": config.block_size,
            "sibling_replay": config.sibling_replay,
            "completion_fraction": config.completion_fraction,
            "rollout_race": config.rollout_race,
            "exploration_schedule": config.exploration_schedule,
            "final_pool_limit": config.final_pool_limit,
            "last_rollout_filter": config.last_rollout_filter,
            "weighted_completion_quota": config.weighted_completion_quota,
            "selection_ready": config.selection_ready,
            "sampling_plans": [step.sampling_plan for step in result.steps],
            "stopping_policy": "logical" if deadline is None else "wall_clock",
            "rollout_scheduler": "streaming" if getattr(backend, "supports_streaming_generation", False) else "batch",
            "engine_execution": getattr(raw, "execution_config", lambda: {})(),
            "sampling_scope": info["sampling_scope"], "seconds": round(time.monotonic() - start, 3)})


__all__ = ["Prepared", "Reasoner"]

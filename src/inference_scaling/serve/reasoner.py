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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from inference_scaling.app.ar import conditional_is_config, confidence_reward, confidence_statistic
from inference_scaling.app.records import Meter, RequestMeter
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


class _ThoughtPublisher:
    """Decode committed thinking only; incomplete byte/marker suffixes wait.

    A prefilled thinking channel remains reasoning on a malformed/incomplete
    block. Reward segmentation and its full-sequence fallback stay unchanged.
    Unknown/non-prefilled formats are buffered until the selected thought ends.
    """

    def __init__(self, raw, prompt, parser, enabled, callback):
        self.raw, self.prompt, self.parser, self.callback = raw, prompt, parser, callback
        formats: tuple[Any, ...] = getattr(parser, "formats", ())
        self.prefilled = enabled and len(formats) == 1 and formats[0].prompt_is_thinking(prompt)
        self.marker = formats[0].end_token_ids if self.prefilled else ()
        self.sent = ""

    def text(self, tokens, *, final=False):
        segments = self.parser.split(self.prompt, tokens, eos_token_id=self.raw.tokenizer.eos_token_id)
        if segments.has_complete_thinking:
            span = segments.thinking_token_ids
        elif self.prefilled and segments.status in {"incomplete", "malformed"}:
            span = tokens[:tokens.index(self.raw.tokenizer.eos_token_id)] if self.raw.tokenizer.eos_token_id in tokens else tokens
            if not final:
                for count in range(min(len(span), len(self.marker) - 1), 0, -1):
                    if span[-count:] == self.marker[:count]:
                        span = span[:-count]
                        break
        else:
            return ""
        text = self.raw.decode(span).strip()
        # An incomplete UTF-8 byte sequence can be replaced by the next token.
        if not final:
            text = text.rstrip("\ufffd").rstrip()
        return text

    def publish(self, tokens, *, final=False):
        if self.callback is None or (not self.prefilled and not final):
            return
        text = self.text(tokens, final=final)
        if not text.startswith(self.sent):
            raise RuntimeError("committed thinking text changed during incremental decoding")
        delta = text[len(self.sent):]
        if delta:
            self.sent = text
            self.callback(delta)


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

    def generate(self, prepared: Prepared, *, on_thinking: Callable[[str], None] | None = None) -> Reply:
        with self._seed_lock:
            seeds = SeedStream(self.seed).derive("request", next(self._seeds))
        prompt, maximum, raw, ar = prepared.prompt, prepared.maximum, self.raw, prepared.ar
        sampling = ar["sampling"]
        policy = SamplingConfig(temperature=float(sampling["temperature"]), top_p=float(sampling["top_p"]),
                                top_k=sampling["top_k"], eos_token_id=raw.tokenizer.eos_token_id)
        thinking_format = thinking_format_from_backend(raw, output_settings_from_config(ar))
        meter = Meter({})  # Engine-wide deltas would mix concurrent clients.
        measured = RequestMeter(raw, meter, raw.parameter_count)
        bounded = budgeted_backend(measured, prompt, thinking_format, dict(prepared.budget))
        scope = SamplingScope.from_config(raw, ar).for_prompt(prompt)
        if scope.scope == "thinking" and self.reward_settings["scope"] == "full":
            scope = scope.full_fallback("reward_uses_full_sequence")
        backend = StatisticRecorder(scope.wrap(bounded, prompt), confidence_statistic("consilience", self.reward_settings))
        reward = confidence_reward("consilience", self.reward_settings, backend, thinking_format)
        config = conditional_is_config(ar["algorithms"]["is"], reward, total_length=maximum, backend=raw)
        start = time.monotonic()
        clock_start = time.perf_counter()
        snapshot = getattr(raw, "snapshot", lambda: None)
        preemptions = getattr(snapshot(), "preemptions", 0)
        first_commit = first_publish = None

        def emit(text):
            nonlocal first_publish
            if first_publish is None:
                first_publish = time.perf_counter() - clock_start
            assert on_thinking is not None
            on_thinking(text)

        publisher = _ThoughtPublisher(raw, prompt, thinking_format, scope.scope == "thinking",
                                      emit if on_thinking is not None else None)

        def committed(tokens):
            nonlocal first_commit
            if tokens and first_commit is None:
                first_commit = time.perf_counter() - clock_start
            publisher.publish(tokens)

        limit = self.serve["max_seconds"]
        deadline = None if limit is None else start + float(limit)
        with meter.phase("search"):
            result = run_conditional_is(backend, prompt, config, reward.batch, SeedStream(seeds),
                                        sampling=policy, deadline=deadline, prefix_reward=reward.model.cached_prefix,
                                        on_commit=committed)
        publisher.publish(result.token_ids, final=True)
        thought_ready = time.perf_counter() - clock_start
        with meter.phase("finish"):
            tokens, info = scope.finish(bounded, prompt, result.token_ids, max_new_tokens=maximum, sampling=policy,
                                        seed=SeedStream(seeds).derive("final-content"))
        # Recomputed preempted tokens are not attributable to an individual
        # client. Conservatively suppress its cost if any preemption overlapped.
        if getattr(snapshot(), "preemptions", 0) != preemptions:
            meter.attribute(None)
        cost = meter.cost(
            lambda delta: delta.get("generation_forward_token_slots", 0) + delta.get("score_forward_token_slots", 0),
            lambda _role, delta: delta.get("estimated_dense_forward_flops", 0),
        )
        # An unfinished thought has no answer; it is returned as the reasoning.
        thinking, content = info["thinking_text"], info["content_text"]
        if info["thinking_status"] == "incomplete":
            thinking, content = content, ""
        elif publisher.prefilled and info["thinking_status"] == "malformed":
            thinking, content = publisher.text(result.token_ids, final=True), ""
        content, calls = parse_tool_calls(content, prepared.tools) if prepared.tools else (content.strip(), ())
        finish: Literal["stop", "tool_calls", "length"] = (
            "length" if not info["ended_by_eos"] else "tool_calls" if calls else "stop")
        return Reply(thinking.strip(), content, calls, finish, len(prompt), len(tokens), {
            "effort": prepared.effort, "steps": len(result.steps), "stopping_reason": result.stopping_reason,
            "generation_budget": dict(prepared.budget), "max_seconds": limit,
            "first_commit_seconds": first_commit, "thought_ready_seconds": thought_ready,
            "first_thinking_publish_seconds": first_publish,
            "sampling_seed": seeds, "cost": cost,
            "thinking_status": info["thinking_status"],
            "thinking_tokens": len(info["thinking_token_ids"]),
            "content_tokens": len(info["content_token_ids"]),
            "generated_tokens": (sum(phase.get("base", {}).get("generated_tokens", 0) for phase in meter.phases.values())
                                 if cost is not None else None),
            "tensor_parallel_size": ar["engine"]["vllm"]["tensor_parallel_size"],
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
            "sampling_scope": info["sampling_scope"], "seconds": time.perf_counter() - clock_start})


__all__ = ["Prepared", "Reasoner"]

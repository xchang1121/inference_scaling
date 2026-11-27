"""Thin serve-side adapter translating protocol Conversations into :class:`ScalingPipeline` calls.

All algorithm wiring (scope, thinking budget, statistic recording, intrinsic
reward, Conditional IS, final-content decoding, and cost attribution) is
converged inside :mod:`inference_scaling.arllm.pipeline`. This module handles
only prompt rendering, incremental SSE text publishing, tool-call parsing, and
optional per-request ``(M, K, B)`` scaling overrides.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.pipeline import (
    CONFIDENCE_REWARDS,
    ScalingOverrides,
    ScalingPipeline,
    ScalingRequest,
)
from inference_scaling.serve.protocol import Conversation, ProtocolError, Reply, parse_tool_calls
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
    seed: int | None = None
    scaling_overrides: ScalingOverrides | None = None


class _TextPublisher:
    def __init__(self, raw, callback):
        self.raw, self.callback, self.sent = raw, callback, ""

    def text(self, tokens, *, final=False):
        text = self.raw.decode(tokens).strip()
        return text if final else text.rstrip("\ufffd").rstrip()

    def publish(self, tokens, *, final=False):
        if self.callback is None:
            return
        text = self.text(tokens, final=final)
        if not text.startswith(self.sent):
            raise RuntimeError("committed text changed during incremental decoding")
        delta = text[len(self.sent):]
        if delta:
            self.sent = text
            self.callback(delta)


class _ThoughtPublisher(_TextPublisher):
    """Decode committed thinking only; incomplete byte/marker suffixes wait."""

    def __init__(self, raw, prompt, parser, enabled, callback):
        super().__init__(raw, callback)
        self.prompt, self.parser = prompt, parser
        formats: tuple[Any, ...] = getattr(parser, "formats", ())
        self.prefilled = enabled and len(formats) == 1 and formats[0].prompt_is_thinking(prompt)
        self.marker = formats[0].end_token_ids if self.prefilled else ()

    def text(self, tokens, *, final=False):
        segments = self.parser.split(self.prompt, tokens, eos_token_id=self.raw.tokenizer.eos_token_id)
        if segments.has_complete_thinking:
            span = segments.thinking_token_ids
        elif self.prefilled and segments.status in {"incomplete", "malformed"}:
            eos = self.raw.tokenizer.eos_token_id
            span = tokens[:tokens.index(eos)] if eos in tokens else tokens
            if not final:
                for count in range(min(len(span), len(self.marker) - 1), 0, -1):
                    if span[-count:] == self.marker[:count]:
                        span = span[:-count]
                        break
        else:
            return ""
        text = self.raw.decode(span).strip()
        if not final:
            text = text.rstrip("\ufffd").rstrip()
        return text

    def publish(self, tokens, *, final=False):
        if not self.prefilled and not final:
            return
        super().publish(tokens, final=final)


class Reasoner:
    """Protocol-facing reasoner backed by any :class:`AutoregressiveBackend` via :class:`ScalingPipeline`."""

    def __init__(self, settings: Mapping[str, Any], backend: Any | None = None) -> None:
        self.reward_kind = settings["run"]["reward"]
        if self.reward_kind not in CONFIDENCE_REWARDS:
            raise ValueError("the service requires run.reward=self_certainty or consilience; verifier needs a dataset")
        self.ar, self.reward_settings, self.serve = settings["ar"], settings["rewards"][self.reward_kind], settings["serve"]
        self.seed = int(settings["run"]["seed"])
        self.raw: Any = backend
        self.pipeline: ScalingPipeline | None = ScalingPipeline(backend) if backend is not None else None
        self._seeds = itertools.count()
        self._seed_lock = threading.Lock()

    def load(self) -> None:
        if self.raw is None:
            self.raw = load_backend(self.ar["model"], self.ar["engine"], seed=self.seed, logprobs=0)
        if self.pipeline is None:
            self.pipeline = ScalingPipeline(self.raw)

    def close(self) -> None:
        close_backend(self.raw)
        self.raw = self.pipeline = None

    def capabilities(self) -> dict[str, Any]:
        """Expose active engine capabilities and external IS defaults for health/introspection."""
        is_cfg = self.ar["algorithms"]["is"]
        m = int(is_cfg["candidate_count"])
        s = int(is_cfg["stages"])
        return {
            "engine": self.pipeline.capabilities().to_dict() if self.pipeline is not None else None,
            "reward": self.reward_kind,
            "selection": "block_metropolized",
            "beam_particles": 2 if s >= 3 and m >= 4 else 1,
            "defaults": {
                "candidate_count": m,
                "rollout_count": int(is_cfg["rollout_count"]),
                "stages": s,
            },
        }

    def prepare(self, conversation: Conversation) -> Prepared:
        """Render the prompt and compute the output budget before reasoning starts."""
        assert self.pipeline is not None and self.raw is not None
        try:
            ar, thinking = resolve_thinking(
                self.ar,
                effort=conversation.effort,
                budget_tokens=conversation.thinking_budget,
                enabled=conversation.enable_thinking,
            )
            options = dict(ar["prompt"]["chat_template_kwargs"])
            if conversation.tools:
                options["tools"] = conversation.tools
            messages = list(conversation.messages)
            if ar["prompt"]["system"] and not any(message["role"] == "system" for message in messages):
                messages.insert(0, {"role": "system", "content": ar["prompt"]["system"]})
            rendered = render_prompt(
                self.raw.tokenizer,
                messages,
                {**ar, "prompt": {**ar["prompt"], "chat_template_kwargs": options}},
            )
            prompt = tuple(self.raw.encode(rendered, add_special_tokens=False))
            budget, maximum = self.pipeline.prepare_budget(
                prompt, ar, thinking, client_max_tokens=conversation.max_tokens, validate_budget_backend=True,
            )
            overrides = ScalingOverrides.from_mapping(conversation.scaling)
        except ValueError as error:
            raise ProtocolError(str(error)) from error
        return Prepared(prompt, thinking["reasoning_effort"], budget, maximum, conversation.tools, ar,
                        conversation.seed, overrides)

    def generate(
        self,
        prepared: Prepared,
        *,
        on_thinking: Callable[[str], None] | None = None,
        on_content: Callable[[str], None] | None = None,
    ) -> Reply:
        assert self.pipeline is not None and self.raw is not None
        seeds = prepared.seed
        if seeds is None:
            with self._seed_lock:
                seeds = SeedStream(self.seed).derive("request", next(self._seeds))
        prompt, raw, ar = prepared.prompt, self.raw, prepared.ar
        thinking_format = thinking_format_from_backend(raw, output_settings_from_config(ar))
        scope = self.pipeline.build_scope(
            prompt, ar, active=True, reward_kind=self.reward_kind, reward_settings=self.reward_settings,
        )

        start = time.monotonic()
        clock_start = time.perf_counter()
        first_commit = first_publish = first_content = thought_ready = None

        def emit_thinking(text: str) -> None:
            nonlocal first_publish
            if first_publish is None:
                first_publish = time.perf_counter() - clock_start
            assert on_thinking is not None
            on_thinking(text)

        publisher = _ThoughtPublisher(
            raw, prompt, thinking_format, scope.scope == "thinking", emit_thinking if on_thinking is not None else None,
        )

        def emit_content(text: str) -> None:
            nonlocal first_content
            if first_content is None:
                first_content = time.perf_counter() - clock_start
            assert on_content is not None
            on_content(text)

        stream_content = on_content is not None and not prepared.tools and scope.scope == "thinking"
        content_publisher = _TextPublisher(raw, emit_content if stream_content else None)

        def committed(tokens: TokenSequence) -> None:
            nonlocal first_commit
            if tokens and first_commit is None:
                first_commit = time.perf_counter() - clock_start
            publisher.publish(tokens)

        def search_complete(search_tokens: TokenSequence) -> None:
            nonlocal thought_ready
            thought_ready = time.perf_counter() - clock_start
            publisher.publish(search_tokens, final=True)

        limit = self.serve["max_seconds"]
        deadline = None if limit is None else start + float(limit)
        outcome = self.pipeline.run_is(
            ScalingRequest(
                prompt=prompt,
                budget=prepared.budget,
                maximum=prepared.maximum,
                seed=seeds,
                ar=ar,
                reward_kind=self.reward_kind,
                reward_settings=self.reward_settings,
                deadline=deadline,
                overrides=prepared.scaling_overrides,
                on_commit=committed,
                on_search_complete=search_complete,
                on_content=content_publisher.publish if stream_content else None,
            ),
            track_preemptions=True,
        )
        info = outcome.info
        if info["thinking_status"] == "complete":
            content_publisher.publish(info["content_token_ids"], final=True)

        # An unfinished thought has no answer; it is returned as the reasoning.
        thinking, content = info["thinking_text"], info["content_text"]
        if info["thinking_status"] == "incomplete":
            thinking, content = content, ""
        elif publisher.prefilled and info["thinking_status"] == "malformed":
            thinking, content = publisher.text(outcome.search_tokens, final=True), ""
        content, calls = parse_tool_calls(content, prepared.tools) if prepared.tools else (content.strip(), ())
        finish: Literal["stop", "tool_calls", "length"] = (
            "length" if not info["ended_by_eos"] else "tool_calls" if calls else "stop"
        )
        config = outcome.config
        return Reply(thinking.strip(), content, calls, finish, len(prompt), len(outcome.tokens), {
            "effort": prepared.effort, "steps": len(outcome.steps), "stopping_reason": outcome.stopping_reason,
            "generation_budget": dict(prepared.budget), "max_seconds": limit,
            "first_commit_seconds": first_commit,
            "thought_ready_seconds": outcome.thought_ready_seconds if thought_ready is None else thought_ready,
            "first_thinking_publish_seconds": first_publish,
            "first_content_publish_seconds": first_content,
            "sampling_seed": seeds, "cost": outcome.cost,
            "thinking_status": info["thinking_status"],
            "thinking_tokens": len(info["thinking_token_ids"]),
            "content_tokens": len(info["content_token_ids"]),
            "generated_tokens": outcome.generated_tokens,
            "tensor_parallel_size": ar["engine"]["vllm"]["tensor_parallel_size"],
            "candidate_count": config.candidate_count, "rollout_count": config.rollout_count,
            "stages": config.stages, "beam_particles": config.beam_particles,
            "stage0_block_size": config.stage0_block_size,
            "reward": self.reward_kind, "reward_config": dict(self.reward_settings),
            "selection": "block_metropolized",
            "final_pool_limit": config.final_pool_limit,
            "sampling_plans": [step.sampling_plan for step in outcome.steps],
            "stopping_policy": "logical" if deadline is None else "wall_clock",
            "rollout_scheduler": outcome.rollout_scheduler,
            "engine_execution": getattr(raw, "execution_config", lambda: {})(),
            "engine_capabilities": self.pipeline.capabilities().to_dict(),
            "sampling_scope": info["sampling_scope"], "seconds": time.perf_counter() - clock_start,
        })


__all__ = ["Prepared", "Reasoner"]

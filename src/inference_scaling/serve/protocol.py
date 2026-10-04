"""OpenAI Chat Completions and Anthropic Messages, translated to the chat template's messages and back.

Requests become the template's messages: system, user, assistant (with
``reasoning_content`` and ``tool_calls`` whose arguments are objects) and tool.
Replies carry the chosen thought, the answer text and the tool calls parsed
from Qwen3.8's ``<tool_call><function=...><parameter=...>`` format.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


class ProtocolError(ValueError):
    """A request the service cannot serve; answered with HTTP 400."""


@dataclass(frozen=True)
class Conversation:
    messages: list[dict[str, Any]]
    # Function tools in the OpenAI format, which the chat template renders.
    tools: list[dict[str, Any]] | None
    effort: str | None
    max_tokens: int | None
    stream: bool


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass(frozen=True)
class Reply:
    thinking: str
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish: Literal["stop", "tool_calls", "length"]
    prompt_tokens: int
    completion_tokens: int
    # How the search went: effort, steps, forward tokens, stopping reason and seconds.
    scaling: dict[str, Any]


def _text(content: Any) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise ProtocolError("message content must be a string or a list of parts")
    texts = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "text":
            raise ProtocolError("only text content is supported")
        texts.append(str(part.get("text", "")))
    return "\n\n".join(texts)


def _arguments(value: Any) -> dict[str, Any]:
    """Tool-call arguments as the object the chat template iterates over."""

    if value in (None, ""):
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ProtocolError("tool-call arguments must be a JSON object")
    return parsed


def _with_system(system: list[str], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The template takes one leading system message."""

    text = "\n\n".join(part for part in system if part)
    return ([{"role": "system", "content": text}] if text else []) + messages


def openai_conversation(body: Mapping[str, Any]) -> Conversation:
    if body.get("n", 1) != 1:
        raise ProtocolError("n must be 1")
    choice = body.get("tool_choice", "auto")
    if choice not in (None, "auto", "none"):
        raise ProtocolError("tool_choice supports only auto or none; forced tool calls are not supported")
    system: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        if role in {"system", "developer"}:
            system.append(_text(message.get("content")))
        elif role in {"user", "tool"}:
            messages.append({"role": role, "content": _text(message.get("content"))})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": _text(message.get("content"))}
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if reasoning:
                entry["reasoning_content"] = str(reasoning)
            calls = [{"function": {"name": call["function"]["name"], "arguments": _arguments(call["function"].get("arguments"))}}
                     for call in message.get("tool_calls") or []]
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
        else:
            raise ProtocolError(f"unsupported message role {role!r}")
    tools = [tool for tool in body.get("tools") or [] if tool.get("type", "function") == "function"]
    return Conversation(_with_system(system, messages), tools if tools and choice != "none" else None,
                        body.get("reasoning_effort"), body.get("max_completion_tokens") or body.get("max_tokens"),
                        bool(body.get("stream")))


def anthropic_conversation(body: Mapping[str, Any]) -> Conversation:
    choice = body.get("tool_choice")
    if choice is not None and (not isinstance(choice, Mapping) or choice.get("type") not in ("auto", "none")
                               or set(choice) != {"type"}):
        raise ProtocolError("tool_choice supports only {type: auto} or {type: none}; "
                            "forced tool calls and additional tool-choice options are not supported")
    messages: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        texts = [str(block.get("text", "")) for block in blocks if block.get("type") == "text"]
        if any(block.get("type") not in {"text", "tool_use", "tool_result", "thinking", "redacted_thinking"}
               for block in blocks):
            raise ProtocolError("only text, tool and thinking blocks are supported")
        if message.get("role") == "user":
            # Tool results answer the previous turn's calls, so they come before the user's text.
            messages += [{"role": "tool", "content": ("Error: " if block.get("is_error") else "")
                          + _text(block.get("content") if isinstance(block.get("content"), list)
                                  else block.get("content") or "")}
                         for block in blocks if block.get("type") == "tool_result"]
            if texts:
                messages.append({"role": "user", "content": "\n\n".join(texts)})
        elif message.get("role") == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(texts)}
            thinking = "\n\n".join(str(block.get("thinking", "")) for block in blocks if block.get("type") == "thinking")
            if thinking:
                entry["reasoning_content"] = thinking
            calls = [{"function": {"name": block["name"], "arguments": _arguments(block.get("input"))}}
                     for block in blocks if block.get("type") == "tool_use"]
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
        else:
            raise ProtocolError(f"unsupported message role {message.get('role')!r}")
    system = body.get("system")
    tools = [{"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                               "parameters": tool["input_schema"]}}
             for tool in body.get("tools") or [] if "input_schema" in tool]
    choice = choice or {}
    effort = (body.get("output_config") or {}).get("effort") or body.get("reasoning_effort")
    return Conversation(_with_system([_text(system)] if system else [], messages),
                        tools if tools and choice.get("type") != "none" else None, effort, body.get("max_tokens"),
                        bool(body.get("stream")))


_TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _typed(value: str, schema: Mapping[str, Any]) -> Any:
    """A parameter's text as its schema type: strings stay text, other types are read as JSON."""

    kinds = schema.get("type", "string")
    if kinds == "string" or (isinstance(kinds, list) and "string" in kinds):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_tool_calls(text: str, tools: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[ToolCall, ...]]:
    """The answer text before any tool call, and the calls in Qwen3.8's format."""

    properties = {tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
                  for tool in tools if "function" in tool}
    calls = tuple(ToolCall(name.strip(), {key.strip(): _typed(value, properties.get(name.strip(), {}).get(key.strip(), {}))
                                          for key, value in _PARAMETER.findall(body)})
                  for name, body in _TOOL_CALL.findall(text))
    return (text[: text.find("<tool_call>")] if calls else text).strip(), calls


def _usage(reply: Reply) -> dict[str, int]:
    return {"prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens}


def _openai_calls(reply: Reply) -> list[dict[str, Any]]:
    return [{"index": index, "id": call.id, "type": "function",
             "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
            for index, call in enumerate(reply.tool_calls)]


def openai_response(reply: Reply, model: str, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content or None, "reasoning_content": reply.thinking}
    if reply.tool_calls:
        message["tool_calls"] = [{key: value for key, value in call.items() if key != "index"} for call in _openai_calls(reply)]
    return {"id": request_id, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": reply.finish}], "usage": _usage(reply),
            "scaling": reply.scaling}


def _sse(payload: Mapping[str, Any], event: str | None = None) -> str:
    return (f"event: {event}\n" if event else "") + f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _openai_chunk(model: str, request_id: str, delta: Mapping[str, Any], finish: str | None = None, **extra: Any) -> str:
    return _sse({"id": request_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra})


OPENAI_KEEPALIVE = ": keep-alive\n\n"


def openai_stream_start(model: str, request_id: str) -> str:
    return _openai_chunk(model, request_id, {"role": "assistant", "content": ""})


def openai_stream_body(reply: Reply, model: str, request_id: str) -> Iterator[str]:
    if reply.thinking:
        yield _openai_chunk(model, request_id, {"reasoning_content": reply.thinking})
    if reply.content:
        yield _openai_chunk(model, request_id, {"content": reply.content})
    if reply.tool_calls:
        yield _openai_chunk(model, request_id, {"tool_calls": _openai_calls(reply)})
    yield _openai_chunk(model, request_id, {}, reply.finish, usage=_usage(reply), scaling=reply.scaling)
    yield "data: [DONE]\n\n"


_STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


def _anthropic_blocks(reply: Reply, thinking: bool) -> list[dict[str, Any]]:
    return ([{"type": "thinking", "thinking": reply.thinking, "signature": ""}] if thinking and reply.thinking else []) \
        + ([{"type": "text", "text": reply.content}] if reply.content else []) \
        + [{"type": "tool_use", "id": "toolu_" + call.id.removeprefix("call_"), "name": call.name, "input": call.arguments}
           for call in reply.tool_calls]


def anthropic_response(reply: Reply, model: str, request_id: str, thinking: bool) -> dict[str, Any]:
    return {"id": request_id, "type": "message", "role": "assistant", "model": model,
            "content": _anthropic_blocks(reply, thinking), "stop_reason": _STOP_REASONS[reply.finish],
            "stop_sequence": None, "usage": {"input_tokens": reply.prompt_tokens, "output_tokens": reply.completion_tokens},
            "scaling": reply.scaling}


ANTHROPIC_KEEPALIVE = _sse({"type": "ping"}, "ping")


def anthropic_stream_start(model: str, request_id: str, prompt_tokens: int) -> str:
    return _sse({"type": "message_start", "message": {
        "id": request_id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}}}, "message_start")


def anthropic_stream_body(reply: Reply, thinking: bool) -> Iterator[str]:
    # Each block opens empty and receives its whole value in one delta.
    empty: dict[str, dict[str, Any]] = {"thinking": {"thinking": ""}, "text": {"text": ""}, "tool_use": {"input": {}}}
    deltas = {"thinking": ("thinking_delta", "thinking"), "text": ("text_delta", "text"),
              "tool_use": ("input_json_delta", "partial_json")}
    for index, block in enumerate(_anthropic_blocks(reply, thinking)):
        yield _sse({"type": "content_block_start", "index": index, "content_block": {**block, **empty[block["type"]]}},
                   "content_block_start")
        kind, key = deltas[block["type"]]
        value = json.dumps(block["input"], ensure_ascii=False) if block["type"] == "tool_use" else block[key]
        yield _sse({"type": "content_block_delta", "index": index, "delta": {"type": kind, key: value}},
                   "content_block_delta")
        yield _sse({"type": "content_block_stop", "index": index}, "content_block_stop")
    yield _sse({"type": "message_delta", "delta": {"stop_reason": _STOP_REASONS[reply.finish], "stop_sequence": None},
                "usage": {"output_tokens": reply.completion_tokens}, "scaling": reply.scaling}, "message_delta")
    yield _sse({"type": "message_stop"}, "message_stop")


__all__ = ["ANTHROPIC_KEEPALIVE", "OPENAI_KEEPALIVE", "Conversation", "ProtocolError", "Reply", "ToolCall",
           "anthropic_conversation", "anthropic_response", "anthropic_stream_body", "anthropic_stream_start",
           "openai_conversation", "openai_response", "openai_stream_body", "openai_stream_start", "parse_tool_calls"]

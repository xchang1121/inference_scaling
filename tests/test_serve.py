from __future__ import annotations

import copy
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from inference_scaling.app.settings import load_settings
from inference_scaling.serve.protocol import (ProtocolError, Reply, ToolCall, anthropic_conversation, openai_conversation,
                                              parse_tool_calls)
from inference_scaling.serve.reasoner import Reasoner
from test_app import ThinkingBackend, ThinkingTokenizer

TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {"properties": {
    "path": {"type": "string"}, "line": {"type": "integer"}, "lines": {"type": "array"}}}}}]


def test_qwen_tool_calls_are_parsed_with_their_schema_types() -> None:
    text = ("I will edit it.\n\n<tool_call>\n<function=edit>\n<parameter=path>\n42\n</parameter>\n"
            "<parameter=line>\n7\n</parameter>\n<parameter=lines>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "I will edit it."
    assert [(call.name, call.arguments) for call in calls] == [("edit", {"path": "42", "line": 7, "lines": ["a", "b"]})]
    assert parse_tool_calls("no calls here", TOOLS) == ("no calls here", ())


def test_openai_requests_become_template_messages() -> None:
    conversation = openai_conversation({"messages": [
        {"role": "developer", "content": "Be brief."}, {"role": "system", "content": [{"type": "text", "text": "Code."}]},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": None, "reasoning_content": "look first",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "edit", "arguments": "{\"line\": 7}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "done"}],
        "tools": TOOLS, "reasoning_effort": "high", "max_tokens": 99, "stream": True})
    assert conversation.messages == [
        {"role": "system", "content": "Be brief.\n\nCode."}, {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "reasoning_content": "look first",
         "tool_calls": [{"function": {"name": "edit", "arguments": {"line": 7}}}]},
        {"role": "tool", "content": "done"}]
    assert (conversation.tools, conversation.effort, conversation.max_tokens, conversation.stream) == (TOOLS, "high", 99, True)
    assert openai_conversation({"messages": [], "tools": TOOLS, "tool_choice": "none"}).tools is None
    with pytest.raises(ProtocolError, match="text"):
        openai_conversation({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})


def test_anthropic_requests_become_template_messages() -> None:
    conversation = anthropic_conversation({"system": [{"type": "text", "text": "Code."}], "max_tokens": 50, "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "look", "signature": "x"},
                                          {"type": "tool_use", "id": "t", "name": "edit", "input": {"line": 7}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "done"},
                                     {"type": "text", "text": "thanks"}]}],
        "tools": [{"name": "edit", "description": "Edit.", "input_schema": {"type": "object"}}],
        "output_config": {"effort": "low"}})
    assert conversation.messages == [
        {"role": "system", "content": "Code."}, {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "reasoning_content": "look",
         "tool_calls": [{"function": {"name": "edit", "arguments": {"line": 7}}}]},
        {"role": "tool", "content": "done"}, {"role": "user", "content": "thanks"}]
    assert conversation.tools == [{"type": "function", "function": {
        "name": "edit", "description": "Edit.", "parameters": {"type": "object"}}}]
    assert (conversation.effort, conversation.max_tokens) == ("low", 50)


class _ChatTokenizer(ThinkingTokenizer):
    def __init__(self) -> None:
        self.options = []

    def apply_chat_template(self, messages, **options):
        self.options.append(options)
        return "prompt"


class _ChatBackend(ThinkingBackend):
    config = SimpleNamespace(max_position_embeddings=262144)
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = _ChatTokenizer()
        self.batches = []

    def sample_batch(self, requests):
        self.batches.append(list(requests))
        return super().sample_batch(requests)


def test_the_reasoner_chooses_a_thought_and_answers_from_it() -> None:
    backend = _ChatBackend()
    settings = load_settings()
    reasoner = Reasoner(settings, backend)
    conversation = openai_conversation({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "minimal"})
    prepared = reasoner.prepare(conversation)
    assert prepared.effort == "low" and backend.tokenizer.options[0]["reasoning_effort"] == "low"
    assert settings["ar"]["max_new_tokens"] is None
    assert prepared.maximum == 262144 - len(prepared.prompt)
    assert prepared.budget["requested_thinking_budget"] == 4096
    default = reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}]}))
    assert default.effort == "xhigh" and default.budget["requested_thinking_budget"] == 131072
    assert default.maximum == prepared.maximum
    reply = reasoner.generate(prepared)
    # The model thinks "7", closes the thought and answers "7".
    assert (reply.thinking, reply.content, reply.finish, reply.tool_calls) == ("#### 7", "#### 7", "stop", ())
    assert reply.scaling["effort"] == "low" and reply.scaling["steps"] >= 1 and reply.scaling["sampling_scope"] == "thinking"
    assert reply.scaling["stopping_reason"] == "stop"
    with pytest.raises(ProtocolError, match="effort"):
        reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "huge"}))
    for alias, effort, limit in [("none", None, 0), ("minimal", "low", 4096), ("low", "low", 4096),
                                 ("medium", "medium", 16384), ("high", "xhigh", 262144),
                                 ("xhigh", "xhigh", 262144), ("max", "xhigh", 262144)]:
        item = reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}],
                                                    "reasoning_effort": alias}))
        assert (item.effort, item.budget["requested_thinking_budget"]) == (effort, limit)
        assert item.budget["effective_thinking_budget"] == min(limit, item.maximum)
        assert backend.tokenizer.options[-1]["enable_thinking"] == (alias != "none")
    for limit, effort in [(0, "low"), (4096, "low"), (4097, "medium"), (16384, "medium"), (16385, "xhigh")]:
        item = reasoner.prepare(openai_conversation({"messages": [], "thinking_budget": limit}))
        assert item.effort == effort and item.budget["requested_thinking_budget"] == limit
    for body in ({"thinking_budget": -1}, {"thinking_budget": True}, {"thinking_budget": 262145},
                 {"reasoning_effort": "medium", "thinking_budget": 10},
                 {"reasoning_effort": "none", "enable_thinking": True}):
        with pytest.raises(ProtocolError):
            reasoner.prepare(openai_conversation(body))
    # The base model would keep thinking; the budget supplies the delimiter,
    # then the selected thought still gets an ordinary final answer.
    backend._probabilities[(3, 0)] = backend._probabilities[(3,)]
    short = reasoner.prepare(openai_conversation({"messages": [], "thinking_budget": 1, "max_completion_tokens": 6}))
    reply = reasoner.generate(short)
    assert (reply.thinking, reply.content, reply.finish) == ("#### 7", "#### 7", "stop")
    assert reply.scaling["generation_budget"]["thinking_budget_enforcement"] == "close_then_content"


@pytest.mark.parametrize("block_first", [False, True])
def test_reasoner_uses_fixed_candidates_and_completions_without_extra_sampling(block_first) -> None:
    settings = load_settings()
    settings["ar"]["algorithms"]["is"].update(candidate_count=3, rollout_count=2, block_size=1, block_first=block_first)
    backend = _ChatBackend()
    reasoner = Reasoner(settings, backend)
    reply = reasoner.generate(reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}]})))
    assert (reply.thinking, reply.content, reply.finish) == ("#### 7", "#### 7", "stop")
    assert (reply.scaling["steps"], reply.scaling["candidate_count"], reply.scaling["rollout_count"],
            reply.scaling["block_size"]) == (2, 3, 2, 1)
    # Candidate generation, completions, the two fresh terminal candidates, and one final answer.
    assert [len(batch) for batch in backend.batches] == [3, 6 if block_first else 3, 2, 1]


def test_reasoner_time_limit_keeps_a_complete_thought_after_a_full_is_step(monkeypatch) -> None:
    settings = load_settings()
    settings["ar"]["algorithms"]["is"].update(candidate_count=3, rollout_count=2, block_size=1)
    settings["serve"]["max_seconds"] = 0.5
    backend = _ChatBackend()
    reasoner = Reasoner(settings, backend)
    prepared = reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}]}))
    clock = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr("inference_scaling.serve.reasoner.time.monotonic", lambda: next(clock))
    reply = reasoner.generate(prepared)
    assert reply.scaling["stopping_reason"] == "deadline" and reply.scaling["steps"] == 1
    assert (reply.thinking, reply.content, reply.finish) == ("#### 7", "#### 7", "stop")
    assert [len(batch) for batch in backend.batches] == [3, 3, 1]


def test_reasoner_default_uses_logical_stopping_even_when_time_passes(monkeypatch) -> None:
    settings = load_settings()
    settings["ar"]["algorithms"]["is"].update(candidate_count=3, rollout_count=2, block_size=1)
    assert settings["serve"]["max_seconds"] is None
    reasoner = Reasoner(settings, _ChatBackend())
    prepared = reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}]}))
    clock = iter([0.0, 10000.0])
    monkeypatch.setattr("inference_scaling.serve.reasoner.time.monotonic", lambda: next(clock))
    reply = reasoner.generate(prepared)
    assert reply.scaling["steps"] == 2 and reply.scaling["stopping_policy"] == "logical"
    assert reply.scaling["stopping_reason"] == "stop"


def test_reasoner_clamps_the_fixed_block_to_the_requested_output_limit() -> None:
    backend = _ChatBackend()
    reasoner = Reasoner(load_settings(), backend)
    prepared = reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}], "max_tokens": 1}))
    reply = reasoner.generate(prepared)
    assert reply.scaling["block_size"] == 1 and reply.scaling["stopping_reason"] == "length"
    assert reply.completion_tokens == 1 and reply.finish == "length" and not reply.content
    assert reply.scaling["generation_budget"]["client_max_tokens"] == 1
    assert reply.scaling["generation_budget"]["effective_max_new_tokens"] == 1
    for value in (0, -1, True, "10", 1.5):
        with pytest.raises(ProtocolError, match="positive integer"):
            openai_conversation({"messages": [{"role": "user", "content": "q"}], "max_completion_tokens": value})
    with pytest.raises(ProtocolError, match="must agree"):
        openai_conversation({"max_tokens": 10, "max_completion_tokens": 20})


class _StubReasoner:
    def prepare(self, conversation):
        return SimpleNamespace(prompt=(1, 2, 3), conversation=conversation)

    def generate(self, prepared):
        time.sleep(0.05)
        return Reply("think", "Editing.", (ToolCall("edit", {"line": 7}, "call_1"),), "tool_calls", 3, 5, {"effort": "medium"})


@pytest.fixture
def client():
    testclient = pytest.importorskip("fastapi.testclient")
    from inference_scaling.serve.server import create_app

    serve = copy.deepcopy(load_settings()["serve"])
    serve["server"]["keepalive_seconds"] = 0.01
    return testclient.TestClient(create_app(_StubReasoner(), serve))


def test_openai_chat_completions_answer_and_stream(client) -> None:
    body = {"model": "any", "messages": [{"role": "user", "content": "q"}], "tools": TOOLS}
    reply = client.post("/v1/chat/completions", json=body).json()
    message = reply["choices"][0]["message"]
    assert (message["content"], message["reasoning_content"], reply["choices"][0]["finish_reason"]) == (
        "Editing.", "think", "tool_calls")
    assert message["tool_calls"][0]["function"] == {"name": "edit", "arguments": "{\"line\": 7}"}
    assert reply["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    with client.stream("POST", "/v1/chat/completions", json=body | {"stream": True}) as response:
        text = "".join(response.iter_text())
    assert ": keep-alive" in text and text.endswith("data: [DONE]\n\n")
    chunks = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks][-1] == "tool_calls"
    assert client.get("/v1/models").json()["data"][0]["id"] == "qwen3.8-27b-is"
    for choice in ("required", {"type": "function", "function": {"name": "edit"}}):
        error = client.post("/v1/chat/completions", json=body | {"tool_choice": choice})
        assert error.status_code == 400 and "tool_choice" in error.json()["error"]["message"]


def test_anthropic_messages_answer_and_stream(client) -> None:
    body = {"model": "any", "max_tokens": 64, "messages": [{"role": "user", "content": "q"}],
            "thinking": {"type": "enabled"}}
    reply = client.post("/v1/messages", json=body).json()
    assert [block["type"] for block in reply["content"]] == ["thinking", "text", "tool_use"]
    assert reply["content"][2]["input"] == {"line": 7} and reply["stop_reason"] == "tool_use"
    with client.stream("POST", "/v1/messages", json=body | {"stream": True}) as response:
        events = [line[7:] for line in "".join(response.iter_text()).splitlines() if line.startswith("event: ")]
    assert events[0] == "message_start" and "ping" in events and events[-2:] == ["message_delta", "message_stop"]
    assert events.count("content_block_start") == 3
    assert client.post("/v1/messages/count_tokens", json=body).json() == {"input_tokens": 3}
    configured = anthropic_conversation(body | {"thinking": {"type": "enabled", "budget_tokens": 32}})
    assert configured.thinking_budget == 32 and configured.enable_thinking is True
    assert client.post("/v1/messages", json=body | {"thinking": {"type": "enabled", "budget_tokens": 32}}).status_code == 200
    error = client.post("/v1/messages", json=body | {"messages": [{"role": "user", "content": [{"type": "image"}]}]})
    assert error.status_code == 400
    for choice in ({"type": "any"}, {"type": "tool", "name": "edit"}):
        error = client.post("/v1/messages", json=body | {"tool_choice": choice})
        assert error.status_code == 400 and "tool_choice" in error.json()["error"]["message"]


def test_cancelled_response_holds_slot_until_worker_finishes():
    pytest.importorskip("fastapi")
    from inference_scaling.serve.server import create_app

    class BlockingReasoner:
        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()
            self.calls = self.active = self.maximum = 0
            self.lock = threading.Lock()

        def prepare(self, conversation):
            return SimpleNamespace(prompt=(1,))

        def generate(self, prepared):
            with self.lock:
                self.calls += 1
                index = self.calls
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            try:
                self.started.set()
                assert self.release.wait(3)
                if index == 1:
                    raise RuntimeError("detached request failed")
                return Reply("", "ok", (), "stop", 1, 1, {})
            finally:
                with self.lock:
                    self.active -= 1

    class Request:
        async def json(self):
            return {"messages": [{"role": "user", "content": "q"}]}

    async def check():
        reasoner = BlockingReasoner()
        serve = load_settings()["serve"]
        serve["server"]["max_concurrent_requests"] = 1
        app = create_app(reasoner, serve)
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/chat/completions")
        async with app.router.lifespan_context(app):
            first = asyncio.create_task(endpoint(Request()))
            second = None
            try:
                assert await asyncio.to_thread(reasoner.started.wait, 2)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(endpoint(Request()))
                await asyncio.sleep(0.05)
                assert reasoner.calls == reasoner.active == 1
            finally:
                reasoner.release.set()
                await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
            assert second is not None and second.result()
        assert reasoner.maximum == 1 and reasoner.active == 0 and reasoner.calls == 2

    asyncio.run(check())

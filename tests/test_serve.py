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
from inference_scaling.serve.settings import load_serve_settings
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
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = _ChatTokenizer()


def test_the_reasoner_chooses_a_thought_and_answers_from_it() -> None:
    backend = _ChatBackend()
    reasoner = Reasoner(load_settings(), load_serve_settings(), backend)
    conversation = openai_conversation({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "minimal"})
    prepared = reasoner.prepare(conversation)
    assert prepared.effort == "low" and backend.tokenizer.options[0]["reasoning_effort"] == "low"
    reply = reasoner.generate(prepared)
    # The model thinks "7", closes the thought and answers "7".
    assert (reply.thinking, reply.content, reply.finish, reply.tool_calls) == ("#### 7", "#### 7", "stop", ())
    assert reply.scaling["effort"] == "low" and reply.scaling["steps"] >= 1 and reply.scaling["sampling_scope"] == "thinking"
    with pytest.raises(ProtocolError, match="effort"):
        reasoner.prepare(openai_conversation({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "huge"}))


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

    serve = copy.deepcopy(load_serve_settings())
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
            "thinking": {"type": "enabled", "budget_tokens": 32}}
    reply = client.post("/v1/messages", json=body).json()
    assert [block["type"] for block in reply["content"]] == ["thinking", "text", "tool_use"]
    assert reply["content"][2]["input"] == {"line": 7} and reply["stop_reason"] == "tool_use"
    with client.stream("POST", "/v1/messages", json=body | {"stream": True}) as response:
        events = [line[7:] for line in "".join(response.iter_text()).splitlines() if line.startswith("event: ")]
    assert events[0] == "message_start" and "ping" in events and events[-2:] == ["message_delta", "message_stop"]
    assert events.count("content_block_start") == 3
    assert client.post("/v1/messages/count_tokens", json=body).json() == {"input_tokens": 3}
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
        serve = load_serve_settings()
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

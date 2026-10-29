"""The HTTP endpoints: ``/v1/chat/completions``, ``/v1/messages``, ``/v1/models`` and ``/health``.

A request reasons in a worker thread, at most ``max_concurrent_requests`` at a
time, all sharing one engine. Streaming sends irrevocable thinking prefixes
after each IS selection; unfinished/unselected rollouts never reach clients.
Authentication and rate limits belong to the gateway.

FastAPI reads the handlers' annotations at runtime, so they are not postponed.
"""

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from inference_scaling.serve.protocol import (
    ANTHROPIC_KEEPALIVE,
    OPENAI_KEEPALIVE,
    Conversation,
    ProtocolError,
    Reply,
    anthropic_conversation,
    anthropic_response,
    anthropic_stream_body,
    anthropic_stream_start,
    anthropic_thinking_delta,
    openai_conversation,
    openai_response,
    openai_stream_body,
    openai_stream_start,
    openai_thinking_delta,
)


def create_app(reasoner: Any, serve: dict[str, Any]) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    server = serve["server"]
    model, keepalive = str(server["served_model_name"]), float(server["keepalive_seconds"])
    slots = asyncio.Semaphore(int(server["max_concurrent_requests"]))
    active: set[asyncio.Task[Reply]] = set()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            # The engine must stay alive until detached requests finish too.
            if active:
                await asyncio.gather(*active, return_exceptions=True)

    app = FastAPI(title="inference-scaling", lifespan=lifespan)

    @app.exception_handler(ProtocolError)
    async def invalid(_request: Request, error: ProtocolError) -> JSONResponse:
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(error)}}, 400)

    async def reason(conversation: Conversation, updates: "asyncio.Queue[str] | None" = None,
                     disconnected: threading.Event | None = None) -> tuple[Any, "asyncio.Future[Reply]"]:
        prepared = reasoner.prepare(conversation)
        await slots.acquire()
        loop = asyncio.get_running_loop()

        def publish(text: str) -> None:
            if updates is not None and (disconnected is None or not disconnected.is_set()):
                loop.call_soon_threadsafe(updates.put_nowait, text)

        def generate() -> Reply:
            try:
                return (reasoner.generate(prepared, on_thinking=publish) if updates is not None
                        else reasoner.generate(prepared))
            finally:
                # Cancelling an asyncio waiter cannot stop a running thread.
                loop.call_soon_threadsafe(slots.release)

        def finished(task: asyncio.Task[Reply]) -> None:
            active.discard(task)
            if not task.cancelled():
                task.exception()  # Observe failures even after a client disconnects.

        task = asyncio.create_task(asyncio.to_thread(generate))
        active.add(task)
        task.add_done_callback(finished)
        return prepared, task

    async def stream(start: str, task: "asyncio.Future[Reply]", ping: str,
                     body: Callable[[Reply, str], Iterator[str]], updates: "asyncio.Queue[str]",
                     delta: Callable[..., Iterator[str]], disconnected: threading.Event) -> AsyncIterator[str]:
        chunks: list[str] = []
        pending = asyncio.create_task(updates.get())
        try:
            yield start
            while True:
                waiters: set[asyncio.Future[Any]] = {task, pending}
                ready, _ = await asyncio.wait(waiters, timeout=keepalive)
                if pending in ready:
                    text = pending.result()
                    for event in delta(text, first=not chunks):
                        yield event
                    chunks.append(text)
                    pending = asyncio.create_task(updates.get())
                if task in ready:
                    # All worker callbacks precede task completion. Drain the
                    # last queued update before closing the thinking block.
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    while not updates.empty():
                        text = updates.get_nowait()
                        for event in delta(text, first=not chunks):
                            yield event
                        chunks.append(text)
                    break
                if not ready:
                    yield ping
            for event in body(task.result(), "".join(chunks)):
                yield event
        finally:
            disconnected.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "inference-scaling"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        conversation = openai_conversation(await request.json())
        request_id = "chatcmpl-" + uuid.uuid4().hex
        updates: asyncio.Queue[str] = asyncio.Queue()
        disconnected = threading.Event()
        _, task = await reason(conversation, updates if conversation.stream else None, disconnected)
        if not conversation.stream:
            return openai_response(await asyncio.shield(task), model, request_id)
        return StreamingResponse(stream(openai_stream_start(model, request_id), task, OPENAI_KEEPALIVE,
                                        lambda reply, prefix: openai_stream_body(reply, model, request_id, prefix),
                                        updates, lambda text, **kw: openai_thinking_delta(text, model, request_id, **kw), disconnected),
                                 media_type="text/event-stream")

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        body = await request.json()
        conversation, request_id = anthropic_conversation(body), "msg_" + uuid.uuid4().hex
        # Anthropic clients see the thought when they enable thinking.
        thinking = (body.get("thinking") or {}).get("type") == "enabled"
        updates: asyncio.Queue[str] = asyncio.Queue()
        disconnected = threading.Event()
        prepared, task = await reason(conversation, updates if conversation.stream and thinking else None, disconnected)
        if not conversation.stream:
            return anthropic_response(await asyncio.shield(task), model, request_id, thinking)
        return StreamingResponse(stream(anthropic_stream_start(model, request_id, len(prepared.prompt)), task,
                                        ANTHROPIC_KEEPALIVE, lambda reply, prefix: anthropic_stream_body(reply, thinking, prefix),
                                        updates, anthropic_thinking_delta, disconnected),
                                 media_type="text/event-stream")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> dict[str, int]:
        return {"input_tokens": len(reasoner.prepare(anthropic_conversation(await request.json())).prompt)}

    return app


def main() -> None:
    import uvicorn

    from inference_scaling.app.settings import load_settings
    from inference_scaling.serve.reasoner import Reasoner
    settings = load_settings()
    serve = settings["serve"]
    reasoner = Reasoner(settings)
    reasoner.load()
    try:
        uvicorn.run(create_app(reasoner, serve), host=serve["server"]["host"], port=int(serve["server"]["port"]))
    finally:
        reasoner.close()


__all__ = ["create_app", "main"]

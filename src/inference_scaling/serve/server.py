"""The HTTP endpoints: ``/v1/chat/completions``, ``/v1/messages``, ``/v1/models`` and ``/health``.

A request reasons in a worker thread, at most ``max_concurrent_requests`` at a
time, all sharing one engine. The thought is chosen before any of it can be
sent, so a stream sends its opening event, keep-alives while the search runs,
then the reply. Authentication and rate limits belong to the gateway.

FastAPI reads the handlers' annotations at runtime, so they are not postponed.
"""

import asyncio
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
    openai_conversation,
    openai_response,
    openai_stream_body,
    openai_stream_start,
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

    async def reason(conversation: Conversation) -> tuple[Any, "asyncio.Future[Reply]"]:
        prepared = reasoner.prepare(conversation)
        await slots.acquire()
        loop = asyncio.get_running_loop()

        def generate() -> Reply:
            try:
                return reasoner.generate(prepared)
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

    async def stream(start: str, task: "asyncio.Future[Reply]", ping: str, body: Callable[[Reply], Iterator[str]]
                     ) -> AsyncIterator[str]:
        yield start
        while not task.done():
            await asyncio.wait({task}, timeout=keepalive)
            if not task.done():
                yield ping
        for event in body(task.result()):
            yield event

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
        _, task = await reason(conversation)
        if not conversation.stream:
            return openai_response(await asyncio.shield(task), model, request_id)
        return StreamingResponse(stream(openai_stream_start(model, request_id), task, OPENAI_KEEPALIVE,
                                        lambda reply: openai_stream_body(reply, model, request_id)),
                                 media_type="text/event-stream")

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        body = await request.json()
        conversation, request_id = anthropic_conversation(body), "msg_" + uuid.uuid4().hex
        # Anthropic clients see the thought when they enable thinking.
        thinking = (body.get("thinking") or {}).get("type") == "enabled"
        prepared, task = await reason(conversation)
        if not conversation.stream:
            return anthropic_response(await asyncio.shield(task), model, request_id, thinking)
        return StreamingResponse(stream(anthropic_stream_start(model, request_id, len(prepared.prompt)), task,
                                        ANTHROPIC_KEEPALIVE, lambda reply: anthropic_stream_body(reply, thinking)),
                                 media_type="text/event-stream")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> dict[str, int]:
        return {"input_tokens": len(reasoner.prepare(anthropic_conversation(await request.json())).prompt)}

    return app


def main() -> None:
    import uvicorn

    from inference_scaling.app.settings import load_settings
    from inference_scaling.serve.reasoner import Reasoner
    from inference_scaling.serve.settings import load_serve_settings

    serve = load_serve_settings()
    reasoner = Reasoner(load_settings(), serve)
    reasoner.load()
    try:
        uvicorn.run(create_app(reasoner, serve), host=serve["server"]["host"], port=int(serve["server"]["port"]))
    finally:
        reasoner.close()


__all__ = ["create_app", "main"]

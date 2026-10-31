"""POST /v1/chat/completions — OpenAI-compatible chat completions.

Session routing
---------------
No session header              → stateless (one-shot per request)
Header present, known ID       → stateful (existing agent.send())
Header present, unknown ID     → stateful (create new agent, return ID in response header)

Accepted session headers, first match wins:
  X-Cursor-Session-ID, then OpenClaw's session_id, then x-session-affinity.

The first turn of a session receives the flattened transcript. Later turns send
the newest user question. A trailing OpenClaw runtime-context message is kept
with that question; it is not sent alone. The SDK Agent holds the earlier history.

Thinking / cursor_params
------------------------
Per-request cursor_params (via extra_body) take priority over the global
CURSORPIPE_THINKING_LEVEL. When either is active, thinking chunks arrive as
delta.reasoning_content (streaming) or cursor_metadata.thinking (non-streaming).
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from cursorpipe._client import (
    CompletionResult,
    StreamChunk,
    complete,
    complete_stateful,
    stream_complete,
    stream_complete_stateful,
)
from cursorpipe._config import settings
from cursorpipe_server.schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    CursorMetadata,
    DeltaMessage,
    StreamChoice,
)
from cursorpipe_server.usage import token_usage_to_openai

router = APIRouter()
logger = logging.getLogger("cursorpipe.completions")

SESSION_HEADER = "X-Cursor-Session-ID"
# OpenClaw sends these when compat.sendSessionAffinityHeaders is true.
# x-client-request-id is omitted: other clients use it as a per-request id.
AFFINITY_SESSION_HEADERS = ("session_id", "x-session-affinity")
USER_ID_HEADER = "X-User-Id"
EXPOSE_THINKING_HEADER = "X-Cursorpipe-Expose-Thinking"


def _incoming_session_id(request: Request) -> str:
    """Resolve the session id from cursorpipe or OpenClaw affinity headers."""
    explicit = request.headers.get(SESSION_HEADER, "").strip()
    if explicit:
        return explicit
    for name in AFFINITY_SESSION_HEADERS:
        value = request.headers.get(name, "").strip()
        if value:
            return value
    return ""


def _stateful_prompt(messages: list, *, initial: bool) -> str:
    """Seed a new agent with the transcript; later turns send the new question."""
    if not initial:
        return _followup_prompt(messages)
    from cursorpipe._client import _flatten_messages

    prompt, _system = _flatten_messages(_messages_as_dicts(messages))
    return prompt or _followup_prompt(messages)


def _message_text(message) -> str:
    content = message.content or ""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return content


def _is_runtime_context(text: str) -> bool:
    """OpenClaw appends this as a user message after the real question."""
    return "OpenClaw runtime context:" in text


def _followup_prompt(messages: list) -> str:
    """Send the newest real user question, with any runtime context after it.

    The runtime-context carrier is itself role=user and is often last. Sending
    only that carrier drops the question for this turn, so the agent answers
    the previous one.
    """
    user_texts = [
        _message_text(message) for message in messages if message.role == "user"
    ]
    if not user_texts:
        return ""
    last_real = None
    for index in range(len(user_texts) - 1, -1, -1):
        if not _is_runtime_context(user_texts[index]):
            last_real = index
            break
    if last_real is None:
        return user_texts[-1]
    context = [
        text
        for text in user_texts[last_real + 1 :]
        if _is_runtime_context(text) and text.strip()
    ]
    question = user_texts[last_real]
    if not context:
        return question
    return "\n\n".join([*context, question])


def _log_stateful_routing_debug(
    session_id: str,
    x_user_id_raw: str | None,
    user_id_value: str,
) -> None:
    from cursorpipe._config import settings as app_settings

    if not app_settings.effective_log_request_headers:
        return
    if not logger.isEnabledFor(logging.INFO):
        return
    from cursorpipe._user_workspace import resolve_workspace_layout

    layout = resolve_workspace_layout(session_id, user_id_value)
    logger.info(
        "[stateful] session_id=%r  x_user_id_header=%r  resolved_user_id=%r  "
        "env_var=%s  agent_cwd=%s",
        session_id,
        x_user_id_raw,
        user_id_value,
        app_settings.session_user_id_env_var,
        layout.cwd,
    )


def _messages_as_dicts(messages: list) -> list[dict]:
    return [m.model_dump(exclude_none=True) for m in messages]


def _include_stream_usage(body: ChatCompletionRequest) -> bool:
    return bool(body.stream_options and body.stream_options.include_usage)


def _expose_thinking(request: Request) -> bool:
    return request.headers.get(EXPOSE_THINKING_HEADER, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _openai_usage(result: CompletionResult) -> CompletionUsage | None:
    if result.usage is None:
        return None
    return token_usage_to_openai(result.usage)


def _cursor_metadata(result: CompletionResult, session_id: str | None = None) -> CursorMetadata:
    usage = result.usage
    return CursorMetadata(
        duration_ms=result.duration_ms,
        run_id=result.run_id,
        agent_id=result.agent_id,
        session_id=session_id,
        thinking=result.thinking,
        thinking_duration_ms=result.thinking_duration_ms,
        cache_read_tokens=usage.cache_read_tokens if usage else 0,
        cache_write_tokens=usage.cache_write_tokens if usage else 0,
    )


# ── Main endpoint ────────────────────────────────────────────────────────────


@router.post("/v1/chat/completions", tags=["completions"])
async def chat_completions(request: Request, body: ChatCompletionRequest):
    cursor_client = request.app.state.cursor_client
    session_store = request.app.state.session_store

    incoming_session_id = _incoming_session_id(request)
    is_stateful = bool(incoming_session_id)
    model = body.model or settings.model
    expose_thinking = _expose_thinking(request)

    if is_stateful:
        from cursorpipe._session_env import resolve_user_id_value

        x_user_id_raw = request.headers.get(USER_ID_HEADER)
        user_id_value = resolve_user_id_value(
            incoming_session_id,
            x_user_id_raw,
        )
        _log_stateful_routing_debug(
            incoming_session_id, x_user_id_raw, user_id_value
        )
        return await _handle_stateful(
            body=body,
            model=model,
            incoming_session_id=incoming_session_id,
            user_id_value=user_id_value,
            cursor_client=cursor_client,
            session_store=session_store,
            expose_thinking=expose_thinking,
        )
    return await _handle_stateless(
        body=body,
        model=model,
        cursor_client=cursor_client,
        expose_thinking=expose_thinking,
    )


# ── Stateless ────────────────────────────────────────────────────────────────


async def _handle_stateless(
    body: ChatCompletionRequest,
    model: str,
    cursor_client,
    expose_thinking: bool,
):
    messages = _messages_as_dicts(body.messages)
    cursor_params = body.cursor_params or None

    if body.stream:
        return EventSourceResponse(
            _stateless_stream_generator(
                messages,
                model,
                cursor_client,
                cursor_params,
                expose_thinking=expose_thinking,
                include_usage=_include_stream_usage(body),
            ),
            media_type="text/event-stream",
        )

    result = await complete(messages, model, cursor_client, cursor_params)
    return JSONResponse(
        ChatCompletionResponse(
            model=result.actual_model or model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(
                        content=result.text,
                        reasoning_content=result.thinking or None,
                    ),
                    finish_reason=result.finish_reason,  # type: ignore[arg-type]
                )
            ],
            usage=_openai_usage(result),
            cursor_metadata=_cursor_metadata(result),
        ).model_dump(exclude_none=True)
    )


async def _stateless_stream_generator(
    messages: list[dict],
    model: str,
    cursor_client,
    cursor_params: dict[str, str] | None = None,
    *,
    expose_thinking: bool,
    include_usage: bool = False,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    pending_usage = None

    # Opening chunk with role
    yield {"data": ChatCompletionChunk(
        id=completion_id, created=created, model=model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    ).model_dump_json(exclude_none=True)}

    async for chunk in stream_complete(
        messages,
        model,
        cursor_client,
        cursor_params,
        expose_thinking=expose_thinking,
    ):
        if chunk.type == "usage":
            pending_usage = chunk.usage
            continue
        yield {"data": _chunk_to_sse(chunk, completion_id, created, model)}

    # Final stop chunk
    yield {"data": ChatCompletionChunk(
        id=completion_id, created=created, model=model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    ).model_dump_json(exclude_none=True)}

    if include_usage and pending_usage is not None:
        yield {"data": ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[],
            usage=token_usage_to_openai(pending_usage),
        ).model_dump_json(exclude_none=True)}

    yield {"data": "[DONE]"}


# ── Stateful ─────────────────────────────────────────────────────────────────


async def _handle_stateful(
    body: ChatCompletionRequest,
    model: str,
    incoming_session_id: str,
    user_id_value: str,
    cursor_client,
    session_store,
    expose_thinking: bool,
):
    cursor_params = body.cursor_params or None

    from cursorpipe._config import settings as app_settings
    from cursorpipe._session_env import session_env_vars, write_session_env_file

    if app_settings.session_user_id_env:
        write_session_env_file(
            app_settings.workspace,
            incoming_session_id,
            session_env_vars(user_id_value),
        )

    entry = await session_store.get(incoming_session_id)
    if entry is None:
        entry = await session_store.get_or_create(
            incoming_session_id,
            model,
            cursor_client,
            cursor_params,
            user_id_value=user_id_value,
        )

    response_headers = {SESSION_HEADER: entry.session_id}
    prompt = _stateful_prompt(
        body.messages,
        initial=await session_store.consume_initial_prompt(entry.session_id),
    )

    if body.stream:
        return EventSourceResponse(
            _stateful_stream_generator(
                entry,
                prompt,
                model,
                expose_thinking=expose_thinking,
                include_usage=_include_stream_usage(body),
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )

    result = await complete_stateful(entry, prompt)
    return JSONResponse(
        content=ChatCompletionResponse(
            model=result.actual_model or model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(
                        content=result.text,
                        reasoning_content=result.thinking or None,
                    ),
                    finish_reason=result.finish_reason,  # type: ignore[arg-type]
                )
            ],
            usage=_openai_usage(result),
            cursor_metadata=_cursor_metadata(result, session_id=entry.session_id),
        ).model_dump(exclude_none=True),
        headers=response_headers,
    )


async def _stateful_stream_generator(
    entry,
    last_user_message: str,
    model: str,
    *,
    expose_thinking: bool,
    include_usage: bool = False,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    pending_usage = None

    yield {"data": ChatCompletionChunk(
        id=completion_id, created=created, model=model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    ).model_dump_json(exclude_none=True)}

    async for chunk in stream_complete_stateful(
        entry,
        last_user_message,
        expose_thinking=expose_thinking,
    ):
        if chunk.type == "usage":
            pending_usage = chunk.usage
            continue
        yield {"data": _chunk_to_sse(chunk, completion_id, created, model)}

    yield {"data": ChatCompletionChunk(
        id=completion_id, created=created, model=model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    ).model_dump_json(exclude_none=True)}

    if include_usage and pending_usage is not None:
        yield {"data": ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[],
            usage=token_usage_to_openai(pending_usage),
        ).model_dump_json(exclude_none=True)}

    yield {"data": "[DONE]"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chunk_to_sse(chunk: StreamChunk, completion_id: str, created: int, model: str) -> str:
    """Convert a StreamChunk to an SSE data payload string."""
    if chunk.type == "thinking":
        delta = DeltaMessage(reasoning_content=chunk.text)
    else:
        delta = DeltaMessage(content=chunk.text)

    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=delta)],
    ).model_dump_json(exclude_none=True)

"""OpenAI-compatible request and response Pydantic models."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Request ─────────────────────────────────────────────────────────────────────

def normalize_chat_message_role(role: object) -> object:
    """Map OpenAI-compatible alias roles before strict validation.

    GPT-5 / OpenClaw often send ``developer`` instead of ``system``.
    Legacy clients may send ``function`` instead of ``tool``.
    """
    if not isinstance(role, str):
        return role
    normalized = role.strip().lower()
    if normalized == "developer":
        return "system"
    if normalized == "function":
        return "tool"
    return normalized


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, role: object) -> object:
        return normalize_chat_message_role(role)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    # Accept and silently ignore unknown OpenAI fields (logit_bias, top_p,
    # frequency_penalty, etc.) so clients never get 422s for fields we
    # don't implement.
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="composer-2.5")
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # cursorpipe extension: per-request Cursor SDK model parameters.
    # Pass via OpenAI SDK as extra_body={"cursor_params": {"reasoning": "medium"}}.
    # Keys/values must match those listed in GET /v1/models → cursor_parameters.
    # Takes priority over the global CURSORPIPE_THINKING_LEVEL setting.
    cursor_params: dict[str, str] | None = None


# ── Non-streaming response ───────────────────────────────────────────────────────


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    reasoning_content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"] = "stop"


class CursorMetadata(BaseModel):
    duration_ms: int = 0
    run_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    thinking: str | None = None
    thinking_duration_ms: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None
    cursor_metadata: CursorMetadata = Field(default_factory=CursorMetadata)


# ── Streaming response (SSE chunks) ─────────────────────────────────────────────


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    usage: CompletionUsage | None = None


# ── Models endpoint ──────────────────────────────────────────────────────────────


class ModelParamValueDef(BaseModel):
    """One accepted value for a model parameter (e.g. value="high")."""

    value: str
    display_name: str = ""


class ModelParamDef(BaseModel):
    """A per-model parameter definition exposed by the Cursor SDK (e.g. thinking)."""

    id: str
    display_name: str = ""
    values: list[ModelParamValueDef] = Field(default_factory=list)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "cursor"
    # cursorpipe extension: per-model parameters from the SDK (e.g. thinking=low|high).
    # Standard OpenAI clients will ignore this field.
    cursor_parameters: list[ModelParamDef] = Field(default_factory=list)


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# ── Sessions endpoint ────────────────────────────────────────────────────────────


class SessionInfo(BaseModel):
    id: str
    model: str
    created_at: str
    last_used_at: str


class SessionList(BaseModel):
    object: Literal["list"] = "list"
    data: list[SessionInfo]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(default="composer-2.5")
    # cursorpipe extension: Cursor SDK model parameters to apply for this session.
    cursor_params: dict[str, str] | None = None

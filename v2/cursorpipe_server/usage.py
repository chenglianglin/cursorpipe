"""Map cursor-sdk TokenUsage to OpenAI-compatible usage objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cursorpipe_server.schemas import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

if TYPE_CHECKING:
    from cursor_sdk.types import TokenUsage


def token_usage_to_openai(usage: TokenUsage) -> CompletionUsage:
    """Convert SDK token counts to an OpenAI ``usage`` object."""
    completion_details = None
    if usage.reasoning_tokens is not None:
        completion_details = CompletionTokensDetails(reasoning_tokens=usage.reasoning_tokens)

    prompt_details = None
    if usage.cache_read_tokens:
        prompt_details = PromptTokensDetails(cached_tokens=usage.cache_read_tokens)

    total = usage.total_tokens
    if not total:
        total = usage.input_tokens + usage.output_tokens

    return CompletionUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=total,
        completion_tokens_details=completion_details,
        prompt_tokens_details=prompt_details,
    )

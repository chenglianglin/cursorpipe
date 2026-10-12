"""Sanitized HTTP request header logging for debugging."""

from __future__ import annotations

import logging
from typing import Mapping

from starlette.requests import Request

logger = logging.getLogger("cursorpipe.access")

# Headers omitted or redacted in logs (case-insensitive match on name).
_REDACT_EXACT = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "proxy-authorization",
    }
)
_REDACT_PREFIXES = ("x-cursor-api-key",)

# Logged at debug when header logging is enabled (easy to grep).
_SESSION_HEADERS = ("x-cursor-session-id", "x-user-id")


def _redact_header_value(name: str, value: str) -> str:
    lower = name.lower()
    if lower in _REDACT_EXACT:
        return "<redacted>"
    for prefix in _REDACT_PREFIXES:
        if lower.startswith(prefix):
            return "<redacted>"
    if len(value) > 500:
        return value[:500] + "…"
    return value


def headers_for_log(request: Request) -> dict[str, str]:
    """Return incoming headers with secrets redacted."""
    out: dict[str, str] = {}
    for name, value in request.headers.items():
        out[name] = _redact_header_value(name, value)
    return out


def session_headers_for_log(request: Request) -> dict[str, str]:
    """Subset of headers used for cursorpipe stateful / user env routing."""
    found: dict[str, str] = {}
    for name, value in request.headers.items():
        lower = name.lower()
        if lower in _SESSION_HEADERS:
            found[lower] = _redact_header_value(name, value)
    return found


def _header_log_level(explicit_override: bool | None) -> int:
    """INFO when headers forced on; DEBUG when following log_level=debug."""
    if explicit_override is True:
        return logging.INFO
    return logging.DEBUG


def log_incoming_headers(
    request: Request,
    *,
    request_id: str,
    enabled: bool,
    explicit_override: bool | None = None,
) -> None:
    """Emit sanitized header lines when header logging is enabled."""
    if not enabled:
        return

    level = _header_log_level(explicit_override)
    if not logger.isEnabledFor(level):
        return

    session = session_headers_for_log(request)
    if session:
        logger.log(
            level,
            "[HEADERS] request_id=%s  session=%s",
            request_id,
            session,
        )
    else:
        logger.log(
            level,
            "[HEADERS] request_id=%s  session=<none>  "
            "(no X-Cursor-Session-ID / X-User-Id)",
            request_id,
        )

    logger.log(
        level,
        "[HEADERS] request_id=%s  all=%s",
        request_id,
        headers_for_log(request),
    )

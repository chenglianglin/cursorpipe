"""Sanitized HTTP request header logging for debugging."""

from __future__ import annotations

import logging

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
_SESSION_HEADERS = (
    "x-cursor-session-id",
    "session_id",
    "x-session-affinity",
    "x-user-id",
)


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


def log_incoming_headers(
    request: Request,
    *,
    request_id: str,
    enabled: bool,
) -> None:
    """Emit sanitized header lines when header logging is enabled."""
    if not enabled or not logger.isEnabledFor(logging.INFO):
        return

    session = session_headers_for_log(request)
    if session:
        logger.info(
            "[HEADERS] request_id=%s  session=%s",
            request_id,
            session,
        )
    else:
        logger.info(
            "[HEADERS] request_id=%s  session=<none>  "
            "(no session id / X-User-Id)",
            request_id,
        )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[HEADERS] request_id=%s  all=%s",
            request_id,
            headers_for_log(request),
        )

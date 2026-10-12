"""Apply CURSORPIPE_LOG_LEVEL and optional header-logging visibility."""

from __future__ import annotations

import logging

from cursorpipe._config import settings

_ACCESS_LOGGERS = ("cursorpipe.access", "cursorpipe.completions")


def configure_logging() -> None:
    """Configure root logging and bump access loggers when header debug is on."""
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    if settings.effective_log_request_headers:
        # Session / stateful header lines use INFO so they appear even when the
        # global level is error/warning (common in Docker production compose).
        for name in _ACCESS_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)


def header_logging_status_message() -> str:
    if not settings.effective_log_request_headers:
        reason = "disabled"
        if settings.log_request_headers is False:
            reason = "CURSORPIPE_LOG_REQUEST_HEADERS=false"
        else:
            reason = f"CURSORPIPE_LOG_LEVEL={settings.log_level} (need debug or LOG_REQUEST_HEADERS=true)"
        return f"Request header logging: off ({reason})"

    if settings.log_request_headers is True:
        return "Request header logging: on (CURSORPIPE_LOG_REQUEST_HEADERS=true, session headers at INFO)"
    return "Request header logging: on (log_level=debug, session headers at INFO)"

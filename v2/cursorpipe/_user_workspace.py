"""Per-user workspace directories under the project root."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from cursorpipe._session_env import resolve_user_id_value

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9._-]+")
_AGENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# openclaw-webchat-<agentId>-<hash>. The hash may be shorter than 32 when the
# 64-character affinity header truncates the end. Agent ids may contain hyphens.
_WEBCHAT_AGENT_SESSION = re.compile(
    r"^openclaw-webchat-(?P<agent>.+)-(?P<hash>[0-9a-fA-F]+)$"
)
# agent:<agentId>:<rest>, including Open WebUI keys such as
# agent:main:openresponses-user:openwebui:<user>:chat:<chat>:<boundary>.
# The affinity header keeps only the first 64 code points, so the rest may be cut.
_AGENT_KEY_SESSION = re.compile(
    r"^agent:(?P<agent>[a-z0-9][a-z0-9_-]{0,63}):"
)
# Gateway prompt-cache key for Open WebUI: <transcript uuid>:<boundaryCount>.
_TRANSCRIPT_SESSION = re.compile(
    r"^(?P<transcript>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?::\d+)?$"
)


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved local agent cwd/dirs and where to write session env files."""

    cwd: Path
    dirs: tuple[Path, ...]
    env_files_root: Path


def safe_user_segment(user_id: str) -> str:
    cleaned = _SAFE_SEGMENT.sub("_", user_id.strip())[:200]
    return cleaned or "anonymous"


def _validated_agent_id(agent_id: str) -> str | None:
    lowered = agent_id.lower()
    if _AGENT_ID.fullmatch(lowered) is None:
        return None
    return lowered


def openclaw_agent_id_from_session(session_id: str) -> str | None:
    """Return the OpenClaw agent id embedded in an affinity key."""
    text = session_id.strip()
    match = _WEBCHAT_AGENT_SESSION.fullmatch(text)
    if match is None:
        match = _AGENT_KEY_SESSION.match(text)
    if match is None:
        return None
    return _validated_agent_id(match.group("agent"))


def _agent_id_from_session_key(session_key: str) -> str | None:
    match = _AGENT_KEY_SESSION.match(session_key.strip())
    if match is None:
        return None
    return _validated_agent_id(match.group("agent"))


def _lookup_session_key(db_path: Path, transcript_id: str) -> str | None:
    """Read the session key for a transcript id. Missing schema returns None."""
    uri = f"file:{quote(db_path.as_posix())}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        for sql in (
            "SELECT session_key FROM session_nodes WHERE current_session_id = ? LIMIT 1",
            "SELECT session_key FROM session_windows WHERE session_id = ? LIMIT 1",
        ):
            try:
                row = connection.execute(sql, (transcript_id,)).fetchone()
            except sqlite3.Error:
                continue
            if row is not None and row[0]:
                return str(row[0])
    finally:
        connection.close()
    return None


def openclaw_agent_id_from_transcript(project_root: Path, session_id: str) -> str | None:
    """Resolve ``<uuid>:<boundary>`` via each agent's session store.

    The store is ``<project_root>/agents/<agentId>/agent/openclaw-agent.sqlite``.
    ``session_nodes.current_session_id`` is the transcript id; the session key
    starts with ``agent:<agentId>:``.
    """
    match = _TRANSCRIPT_SESSION.fullmatch(session_id.strip())
    if match is None:
        return None
    transcript_id = match.group("transcript").lower()
    agents_dir = project_root / "agents"
    if not agents_dir.is_dir():
        return None
    for agent_dir in sorted(path for path in agents_dir.iterdir() if path.is_dir()):
        db_path = agent_dir / "agent" / "openclaw-agent.sqlite"
        if not db_path.is_file():
            continue
        session_key = _lookup_session_key(db_path, transcript_id)
        if session_key is None:
            continue
        agent_id = _agent_id_from_session_key(session_key)
        if agent_id is not None:
            return agent_id
    return None


def agent_workspace_dir(project_root: Path, agent_id: str) -> Path:
    """Map an OpenClaw agent id to its Cursor project directory."""
    if agent_id == "main":
        return project_root / "workspace"
    return project_root / "workspaces" / agent_id


def resolve_workspace_layout(
    session_id: str,
    user_id_value: str | None = None,
) -> WorkspaceLayout:
    """Pick agent cwd (and optional multi-root dirs) for this session/user."""
    from cursorpipe._config import settings

    project_root = Path(settings.workspace).expanduser().resolve()
    agent_id = openclaw_agent_id_from_session(session_id)
    if agent_id is None:
        agent_id = openclaw_agent_id_from_transcript(project_root, session_id)
    if agent_id is not None:
        return WorkspaceLayout(
            cwd=agent_workspace_dir(project_root, agent_id),
            dirs=(),
            env_files_root=project_root,
        )

    user_id = user_id_value if user_id_value is not None else resolve_user_id_value(
        session_id
    )

    mode = (settings.user_workspace_mode or "off").lower().strip()
    if mode != "subdir":
        return WorkspaceLayout(cwd=project_root, dirs=(), env_files_root=project_root)

    rel = settings.user_workspace_subdir.strip().strip("/") or ".cursorpipe/users"
    user_dir = project_root / rel / safe_user_segment(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    dirs: tuple[Path, ...] = ()
    if settings.user_workspace_mount_project_root:
        dirs = (project_root,)

    return WorkspaceLayout(cwd=user_dir, dirs=dirs, env_files_root=user_dir)

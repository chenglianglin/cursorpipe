"""Per-user workspace directories under the project root."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from cursorpipe._session_env import resolve_user_id_value

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9._-]+")
_AGENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# openclaw-webchat-<agentId>-<hash>. The hash may be shorter than 32 when the
# 64-character affinity header truncates the end. Agent ids may contain hyphens.
_WEBCHAT_AGENT_SESSION = re.compile(
    r"^openclaw-webchat-(?P<agent>.+)-(?P<hash>[0-9a-fA-F]+)$"
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


def openclaw_agent_id_from_session(session_id: str) -> str | None:
    """Return the OpenClaw agent id embedded in a webchat affinity key."""
    match = _WEBCHAT_AGENT_SESSION.fullmatch(session_id.strip())
    if match is None:
        return None
    agent_id = match.group("agent").lower()
    if _AGENT_ID.fullmatch(agent_id) is None:
        return None
    return agent_id


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

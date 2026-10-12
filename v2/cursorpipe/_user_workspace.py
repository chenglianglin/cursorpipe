"""Per-user workspace directories under the project root."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from cursorpipe._session_env import resolve_user_id_value

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved local agent cwd/dirs and where to write session env files."""

    cwd: Path
    dirs: tuple[Path, ...]
    env_files_root: Path


def safe_user_segment(user_id: str) -> str:
    cleaned = _SAFE_SEGMENT.sub("_", user_id.strip())[:200]
    return cleaned or "anonymous"


def resolve_workspace_layout(
    session_id: str,
    user_id_value: str | None = None,
) -> WorkspaceLayout:
    """Pick agent cwd (and optional multi-root dirs) for this session/user."""
    from cursorpipe._config import settings

    project_root = Path(settings.workspace).expanduser().resolve()
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

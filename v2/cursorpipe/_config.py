import os

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CURSORPIPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # ── Cursor auth ────────────────────────────────────────────────────────────
    # CURSOR_API_KEY is primary (matches the SDK's own env var name).
    # CURSORPIPE_API_KEY is accepted as an alias for consistency with v1,
    # so users can reuse a v1 .env file without changes.
    cursor_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CURSOR_API_KEY", "CURSORPIPE_API_KEY"),
    )

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)

    # Optional bearer token clients must send in Authorization header.
    # Empty string disables auth.
    bearer_token: str = Field(default="")

    # Comma-separated allowed CORS origins. "*" allows all.
    cors_origins: str = Field(default="*")

    # ── Agent ──────────────────────────────────────────────────────────────────
    model: str = Field(default="composer-2.5")

    # Working directory passed to LocalAgentOptions.
    workspace: str = Field(default=".")

    # ── Stateful sessions ──────────────────────────────────────────────────────
    session_ttl_minutes: int = Field(default=30)

    # When true, stateful requests (X-Cursor-Session-ID) inject a shell env var
    # for the SDK local agent — default name X_USER_ID — so curl/scripts can pass
    # open-terminal OpenAPI header x-user-id. Value: X-User-Id header if set,
    # else the session id string.
    session_user_id_env: bool = Field(default=True)

    # Shell/env name forwarded to the agent (curl: -H "x-user-id: $X_USER_ID").
    session_user_id_env_var: str = Field(default="X_USER_ID")

    # Per-user agent working directory under CURSORPIPE_WORKSPACE (stateful).
    # off    — all sessions use workspace root (default)
    # subdir — cwd = workspace/<user_workspace_subdir>/<sanitized X-User-Id or session id>
    user_workspace_mode: str = Field(default="off")
    user_workspace_subdir: str = Field(default=".cursorpipe/users")
    # When subdir mode: also load project rules/skills from workspace root via dirs.
    user_workspace_mount_project_root: bool = Field(default=True)

    # ── Thinking / reasoning ───────────────────────────────────────────────────
    # Controls whether thinking is requested from the SDK AND surfaced in
    # responses (reasoning_content for streaming, cursor_metadata.thinking for
    # non-streaming).
    #
    # Accepted values: "off" | "low" | "high"
    #   off  — do not request thinking; discard any thinking tokens (default)
    #   low  — request thinking=low from the SDK; surface in response
    #   high — request thinking=high from the SDK; surface in response
    #
    # Backward-compat: CURSORPIPE_EXPOSE_THINKING=true maps to thinking_level="high".
    # CURSORPIPE_EXPOSE_THINKING=false (or unset) leaves thinking_level unchanged.
    thinking_level: str = Field(default="off")

    @model_validator(mode="before")
    @classmethod
    def _upgrade_expose_thinking(cls, values: dict) -> dict:
        """Convert legacy CURSORPIPE_EXPOSE_THINKING=true → thinking_level=high.

        pydantic-settings drops unknown env vars before this validator runs, so
        we read CURSORPIPE_EXPOSE_THINKING directly from os.environ. The env
        var set via monkeypatch.setenv() is visible there too.
        """
        import os

        expose = os.environ.get("CURSORPIPE_EXPOSE_THINKING", "").lower()
        if expose in ("true", "1", "yes"):
            # Only upgrade if thinking_level is not explicitly provided
            existing = values.get("CURSORPIPE_THINKING_LEVEL") or values.get("thinking_level") or ""
            if not existing or existing == "off":
                values["thinking_level"] = "high"
        return values

    # ── Logging ────────────────────────────────────────────────────────────────
    # Accepts: debug, info, warning, error, critical
    log_level: str = Field(default="info")

    # Log sanitized incoming HTTP headers on each request.
    # unset (None): enable only when CURSORPIPE_LOG_LEVEL=debug
    # true: log session headers at INFO even when log level is error/warning
    # false: never log headers, even when log level is debug
    log_request_headers: bool | None = Field(default=None)

    @property
    def thinking_param(self) -> str | None:
        """Return the SDK thinking parameter value, or None if thinking is off."""
        level = self.thinking_level.lower().strip()
        return level if level in ("low", "high") else None

    def cors_origins_list(self) -> list[str]:
        """Parse CURSORPIPE_CORS_ORIGINS into a list for CORSMiddleware."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_log_request_headers(self) -> bool:
        """Whether middleware should emit sanitized request header debug lines."""
        if self.log_request_headers is not None:
            return self.log_request_headers
        return self.log_level.lower().strip() == "debug"


settings = Settings()


def local_agent_options():
    """Shared LocalAgentOptions for all SDK agents in this server."""
    from cursor_sdk import LocalAgentOptions

    return LocalAgentOptions(
        cwd=settings.workspace,
        setting_sources=["project"],
    )


def local_agent_options_wire() -> dict:
    """Local agent options as bridge wire JSON (for merging envVars)."""
    return local_agent_options().to_json()


def local_agent_options_for_session(
    session_id: str,
    *,
    user_id_value: str | None = None,
) -> dict | object:
    """Local options for a stateful session (optional per-user cwd + envVars)."""
    from cursor_sdk import LocalAgentOptions

    from cursorpipe._session_env import (
        merge_local_options_with_env,
        resolve_user_id_value,
        session_env_vars,
        write_session_env_file,
    )
    from cursorpipe._user_workspace import resolve_workspace_layout

    layout = resolve_workspace_layout(session_id, user_id_value)
    opts = LocalAgentOptions(
        cwd=os.fspath(layout.cwd),
        dirs=[os.fspath(p) for p in layout.dirs] if layout.dirs else None,
        setting_sources=["project"],
    )
    wire = opts.to_json()

    if not settings.session_user_id_env:
        return wire

    value = user_id_value if user_id_value is not None else resolve_user_id_value(
        session_id
    )
    env_vars = session_env_vars(value)
    env_file = write_session_env_file(layout.env_files_root, session_id, env_vars)
    return merge_local_options_with_env(wire, env_vars, env_file)

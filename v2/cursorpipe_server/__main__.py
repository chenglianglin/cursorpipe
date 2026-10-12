"""CLI entry point: `cursorpipe-server` or `python -m cursorpipe_server`."""

import uvicorn

from cursorpipe._config import settings
from cursorpipe._logging_setup import configure_logging


def main() -> None:
    log_level = settings.log_level.lower()
    configure_logging()

    uvicorn.run(
        "cursorpipe_server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()

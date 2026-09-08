import os
import sys

from loguru import logger

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or os.getenv("LOGURU_LEVEL") or "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=resolved_level)
    _CONFIGURED = True

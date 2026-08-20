"""Structured logging setup for the whole platform."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)

def setup_logging(level: str | None = None, stream: bool = True) -> None:
    """Idempotent logging configuration.

    Level resolution order: explicit arg > $MINILLM_LOG_LEVEL > INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("MINILLM_LOG_LEVEL", "INFO")).upper()
    handlers: list[logging.Handler] = []
    if stream:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=_FORMAT,
        handlers=handlers or None,
        force=True,
    )
    # Keep third-party logs (transformers/torch) readable.
    logging.getLogger("transformers").setLevel(os.environ.get("HF_LOG_LEVEL", "WARNING"))
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)

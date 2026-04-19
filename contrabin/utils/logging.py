"""Rich-aware logging setup."""

from __future__ import annotations

import logging
from typing import Any

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # pragma: no cover - rich is a declared dependency
    _HAS_RICH = False


def setup_logging(level: int | str = logging.INFO, **kwargs: Any) -> None:
    """Configure the root logger.

    Uses :class:`rich.logging.RichHandler` when available, otherwise a plain
    ``StreamHandler``.
    """
    if _HAS_RICH:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_time=True, show_path=False, **kwargs
        )
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=[handler], force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

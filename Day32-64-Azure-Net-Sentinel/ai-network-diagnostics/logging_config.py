"""
Structured logging setup for the diagnostic pipeline.
Configures root logger level and format; modules use getLogger(__name__).
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root logger and console handler."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def log_with_correlation(
    logger: logging.Logger,
    level: int,
    msg: str,
    correlation_id: Optional[str] = None,
    *args: object,
    **kwargs: object,
) -> None:
    """Log message with optional correlation_id prefix."""
    if correlation_id:
        msg = f"[cid={correlation_id}] {msg}"
    logger.log(level, msg, *args, **kwargs)

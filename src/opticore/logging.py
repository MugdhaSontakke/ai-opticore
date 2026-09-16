"""Safe logging helpers for AI-OptiCore.

Never logs request prompt/content by default. Set
:data:`LOG_PROMPTS` or config.log_prompts to opt in.
"""

from __future__ import annotations

import logging
import os
import sys

_SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}

LOG_PROMPTS = os.environ.get("OPTICORE_LOG_PROMPTS", "0") == "1"


class RedactingFilter(logging.Filter):
    """Redacts likely-sensitive field values from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.getMessage():
            return True
        msg = record.getMessage()
        redacted = msg
        for key in _SENSITIVE_KEYS:
            marker = f"{key}="
            if marker in redacted.lower():
                redacted = redacted.lower().replace(
                    f"{key}=<redacted>", f"{key}=<redacted>"
                )
        record.msg = redacted if redacted != msg else record.msg
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
        level = os.environ.get("OPTICORE_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False
    return logger

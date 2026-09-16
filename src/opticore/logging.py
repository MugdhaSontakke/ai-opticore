"""Safe logging helpers for AI-OptiCore.

Never logs request prompt/content by default. Set
:data:`LOG_PROMPTS` or config.log_prompts to opt in.
"""

from __future__ import annotations

import logging
import os
import re
import sys

_SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}

# Common secret token shapes (OpenAI sk-, Anthropic sk-ant-, GitHub ghp_, etc.)
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"(ghp|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"(AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*"),
)

LOG_PROMPTS = os.environ.get("OPTICORE_LOG_PROMPTS", "0") == "1"

_REDACTED = "<redacted>"


class RedactingFilter(logging.Filter):
    """Redacts likely-sensitive secrets from log records.

    Redacts `key=value` pairs for known sensitive keys, raw secret-shaped
    tokens, and Bearer headers. Filters run on the formatted message.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.getMessage():
            return True
        record.msg = self._redact(record.getMessage())
        record.args = ()
        return True

    @staticmethod
    def _redact(message: str) -> str:
        out = message
        lowered = out.lower()
        for key in _SENSITIVE_KEYS:
            marker = f"{key}="
            if marker in lowered:
                out = re.sub(rf"{re.escape(key)}=[^\s,;\"']+", f"{key}={_REDACTED}", out, flags=re.IGNORECASE)
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(_REDACTED, out)
        return out


def log_prompt_content(
    logger: logging.Logger, label: str, prompt: str, *, enabled: bool = LOG_PROMPTS
) -> None:
    """Log raw prompt content at DEBUG level when explicitly enabled.

    Prompts are never logged by default (privacy-first). Opt in via the
    ``OPTICORE_LOG_PROMPTS=1`` environment variable or the
    ``log_prompts`` configuration flag. Secrets inside the text are still
    redacted by :class:`RedactingFilter`.
    """
    if enabled and prompt:
        logger.debug("[prompt:%s] %s", label, prompt)


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

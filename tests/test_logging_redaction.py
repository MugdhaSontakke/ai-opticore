"""Tests for logging redaction and provider error mapping."""

from __future__ import annotations

import logging

from opticore.logging import RedactingFilter


def test_redact_sensitive_key_value() -> None:
    out = RedactingFilter._redact("OPENAI_API_KEY=sk-abcdef123456 config")
    assert "sk-abcdef123456" not in out
    assert "<redacted>" in out


def test_redact_raw_sdk_token() -> None:
    out = RedactingFilter._redact("key=sk-ant-abcdefghijklmnopqrstuvwxyz12345678")
    assert "sk-ant-abcdefghijklmnopqrstuvwxyz" not in out


def test_redact_bearer_header() -> None:
    out = RedactingFilter._redact("Authorization: Bearer abcdef.ghijkl.mnopqr")
    assert "Bearer abcdef" not in out
    assert "<redacted>" in out


def test_redact_leaves_plain_text_intact() -> None:
    message = "optimizer=prompt tokens=10->8"
    out = RedactingFilter._redact(message)
    assert out == message


def test_filter_applies_to_record() -> None:
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="api_key=sk-live-secret-value",
        args=(), exc_info=None,
    )
    assert RedactingFilter().filter(record) is True
    assert "sk-live-secret-value" not in record.getMessage()

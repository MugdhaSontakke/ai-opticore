"""Tests for SSRF/URL validation and secret redaction helpers."""

from __future__ import annotations

import pytest

from opticore.exceptions import ConfigurationError
from opticore.security import redact_text, validate_base_url


class TestValidateBaseUrl:
    def test_accepts_http_localhost_default(self) -> None:
        assert validate_base_url("http://localhost:11434") == "http://localhost:11434"

    def test_strips_trailing_slash(self) -> None:
        assert validate_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"

    def test_accepts_public_literal_ip(self) -> None:
        assert validate_base_url("http://8.8.8.8") == "http://8.8.8.8"

    @pytest.mark.parametrize("url", ["ftp://host/x", "file:///etc/passwd", "gopher://x"])
    def test_rejects_non_http_schemes(self, url: str) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url(url)

    def test_rejects_empty_url(self) -> None:
        for bad in ("", "   ", None):
            with pytest.raises(ConfigurationError):
                validate_base_url(bad)

    def test_rejects_url_without_host(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url("https:///path")

    def test_rejects_embedded_credentials(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url("https://user:secret@api.example.com/v1")

    def test_metadata_endpoint_always_blocked(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url("http://169.254.169.254/latest/meta-data")

    def test_private_network_allowed_by_default(self) -> None:
        assert validate_base_url("http://10.0.0.5:8080") == "http://10.0.0.5:8080"

    def test_private_network_blocked_when_disabled(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url("http://10.0.0.5", allow_private_networks=False)

    def test_allowed_hosts_overrides_network_checks(self) -> None:
        url = "http://10.0.0.5"
        assert (
            validate_base_url(
                url,
                allow_private_networks=False,
                allowed_hosts=["10.0.0.5"],
            )
            == url
        )

    def test_resolution_failure_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url("http://nonexistent-host.invalid/")

    def test_loopback_blocked_when_disallowed(self) -> None:
        with pytest.raises(ConfigurationError):
            validate_base_url("http://127.0.0.1", allow_loopback=False)


class TestRedactText:
    def test_redacts_sensitive_key_value(self) -> None:
        out = redact_text("OPENAI_API_KEY=sk-abcdef123456 config")
        assert "sk-abcdef123456" not in out
        assert "<redacted>" in out

    def test_redacts_raw_sdk_token(self) -> None:
        out = redact_text("key=sk-ant-abcdefghijklmnopqrstuvwxyz12345678")
        assert "sk-ant-abcdefghijklmnopqrstuvwxyz" not in out

    def test_redacts_bearer_header(self) -> None:
        out = redact_text("Authorization: Bearer abcdef.ghijkl.mnopqr")
        assert "Bearer abcdef" not in out
        assert "<redacted>" in out

    def test_leaves_plain_text_intact(self) -> None:
        message = "optimizer=prompt tokens=10->8"
        assert redact_text(message) == message

    def test_handles_empty_input(self) -> None:
        assert redact_text("") == ""

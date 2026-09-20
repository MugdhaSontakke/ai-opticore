"""Provider hardening tests: retries, error mapping, malformed responses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opticore.core.config import ProviderConfig
from opticore.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderUnavailableError,
)


class FakeRateLimitError(Exception):
    status_code = 429


class FakeServerError(Exception):
    status_code = 500


class FakeBadRequestError(Exception):
    status_code = 400


class _NoMatchError(Exception):
    """Sentinel that raised fake exceptions never subclass."""


def _install_fake_openai(create: Any, extras: dict[str, Any] | None = None) -> None:
    """Install a fake `openai` module whose client calls ``create(**kwargs)``."""
    import sys

    class FakeCompletions:
        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def create(self, **kwargs: Any) -> Any:
            return self._fn(**kwargs)

    class FakeChat:
        def __init__(self, fn: Any) -> None:
            self.completions = FakeCompletions(fn)

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = FakeChat(create)

    attrs = {
        "OpenAI": FakeClient,
        # Map SDK exception names to a class the raised fakes never inherit so
        # the isinstance() classifier cannot false-match in arbitrary order.
        "RateLimitError": _NoMatchError,
        "APIStatusError": _NoMatchError,
        "AuthenticationError": _NoMatchError,
        "APITimeoutError": _NoMatchError,
        "APIConnectionError": _NoMatchError,
    }
    if extras:
        attrs.update(extras)
    sys.modules["openai"] = SimpleNamespace(**attrs)


def _provider(**kwargs: Any) -> Any:
    from opticore.providers.openai import OpenAIProvider

    provider = OpenAIProvider()
    provider.config.api_key_env = "OPTICORE_TEST_FAKE_KEY"
    return provider


@pytest.fixture
def fake_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTICORE_TEST_FAKE_KEY", "sk-test")


def test_openai_retries_rate_limit_then_succeeds(fake_key) -> None:
    calls = {"n": 0}

    def create(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimitError("slow down")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            model="gpt-test",
            usage=None,
        )

    _install_fake_openai(create, extras={"RateLimitError": FakeRateLimitError})
    provider = _provider()
    result = provider.generate({"prompt": "hi"})
    assert result.content == "ok"
    assert calls["n"] == 3


def test_openai_non_retryable_status_surfaces_immediately(fake_key) -> None:
    calls = {"n": 0}

    def create(**kwargs: Any) -> Any:
        calls["n"] += 1
        raise FakeBadRequestError("bad request")

    _install_fake_openai(create)
    provider = _provider()
    with pytest.raises(ProviderError):
        provider.generate({"prompt": "hi"})
    assert calls["n"] == 1


def test_openai_server_error_retried_then_unavailable(fake_key) -> None:
    calls = {"n": 0}

    def create(**kwargs: Any) -> Any:
        calls["n"] += 1
        raise FakeServerError("server exploded")

    _install_fake_openai(create)
    provider = _provider()
    provider.config.max_attempts = 3  # initial + 2 retries
    with pytest.raises(ProviderUnavailableError):
        provider.generate({"prompt": "hi"})
    assert calls["n"] == 3  # initial + 2 retries


def test_openai_empty_choices_raises_response_error(fake_key) -> None:
    def create(**kwargs: Any) -> Any:
        return SimpleNamespace(choices=[], model="gpt-test", usage=None)

    _install_fake_openai(create)
    provider = _provider()
    with pytest.raises(ProviderResponseError):
        provider.generate({"prompt": "hi"})


def test_openai_connection_error_maps_to_unavailable(fake_key) -> None:
    class FakeConnectionError(Exception):
        pass

    def create(**kwargs: Any) -> Any:
        raise FakeConnectionError("connection refused")

    _install_fake_openai(
        create, extras={"APIConnectionError": FakeConnectionError, "RateLimitError": Exception}
    )
    provider = _provider()
    provider.config.max_retries = 1
    with pytest.raises(ProviderUnavailableError):
        provider.generate({"prompt": "hi"})


def test_openai_base_url_ssrf_guardrail() -> None:
    from opticore.providers.openai import OpenAIProvider

    with pytest.raises(ConfigurationError):
        OpenAIProvider(config=ProviderConfig(base_url="http://169.254.169.254/latest"))


def test_provider_config_backoff_constraints() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(backoff_max_seconds=0.1, backoff_base_seconds=2.0)
    with pytest.raises(ConfigurationError):
        ProviderConfig(backoff_base_seconds=-1)


def test_provider_config_max_attempts_derived() -> None:
    config = ProviderConfig(max_retries=4)
    assert config.max_attempts == 5


def test_ollama_http_non_json_body_maps_to_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import requests

    from opticore.providers.ollama import OllamaProvider

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> str:
            return "not a dict"

    # Make the preferred `ollama` client unavailable so the HTTP path runs.
    monkeypatch.setitem(sys.modules, "ollama", None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse())

    provider = OllamaProvider(base_url="http://127.0.0.1:9999")
    with pytest.raises(ProviderResponseError):
        provider.generate({"prompt": "hi"})


def test_redaction_in_error_messages(fake_key) -> None:
    from opticore.security import redact_text

    message = "OpenAI request failed: OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz12345"
    cleaned = redact_text(message)
    assert "sk-abcdefghijklmnopqrstuvwxyz12345" not in cleaned
    assert "<redacted>" in cleaned

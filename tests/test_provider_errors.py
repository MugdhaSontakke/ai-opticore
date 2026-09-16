"""Tests for provider error mapping using mocked SDKs (no external APIs)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opticore.exceptions import ProviderAuthError, ProviderTimeoutError
from opticore.providers.openai import OpenAIProvider


class FakeTimeoutError(Exception):
    pass


def _install_fake_openai(behavior: str) -> None:
    """Install a fake `openai` module with the given completion behavior."""
    import sys

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        @property
        def chat(self) -> FakeChat:
            return FakeChat(behavior)

    class FakeChat:
        def __init__(self, behavior: str) -> None:
            self.behavior = behavior
            self.completions = FakeCompletions(behavior)

    class FakeCompletions:
        def __init__(self, behavior: str) -> None:
            self.behavior = behavior

        def create(self, **kwargs: Any) -> Any:
            if self.behavior == "timeout":
                raise FakeTimeoutError("Request timed out")
            if self.behavior == "auth":
                raise RuntimeError("Incorrect API key provided")
            raise AssertionError("unexpected behavior")

    fake = SimpleNamespace(OpenAI=FakeClient, APITimeoutError=FakeTimeoutError)
    sys.modules["openai"] = fake


def _provider(**kwargs: Any) -> OpenAIProvider:
    provider = OpenAIProvider()
    provider.config.api_key_env = "OPTICORE_TEST_FAKE_KEY"
    return provider


def test_timeout_maps_to_provider_timeout(monkeypatch) -> None:
    _install_fake_openai("timeout")
    provider = _provider()
    with monkeypatch.context() as m:
        m.setenv("OPTICORE_TEST_FAKE_KEY", "sk-test")
        with pytest.raises(ProviderTimeoutError):
            provider.generate({"prompt": "hi"})


def test_auth_failure_maps_to_provider_auth(monkeypatch) -> None:
    _install_fake_openai("auth")
    provider = _provider()
    with monkeypatch.context() as m:
        m.setenv("OPTICORE_TEST_FAKE_KEY", "sk-test")
        with pytest.raises(ProviderAuthError):
            provider.generate({"prompt": "hi"})


def test_missing_api_key_raises_auth_error(monkeypatch) -> None:
    provider = _provider()
    with monkeypatch.context() as m:
        m.delenv("OPTICORE_TEST_FAKE_KEY", raising=False)
        with pytest.raises(ProviderAuthError):
            provider.generate({"prompt": "hi"})

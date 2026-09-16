"""Abstract model provider interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.logging import get_logger

logger = get_logger("opticore.providers")


class ProviderError(Exception):
    """Base exception for provider failures."""


class ProviderAuthError(ProviderError):
    """Raised when a provider reports missing/invalid credentials."""


class ProviderRateError(ProviderError):
    """Raised on provider-side rate limiting or quota exhaustion."""


@dataclass
class ProviderResponse:
    """Normalized response returned by every provider."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class BaseProvider(ABC):
    """Base class every model provider extends.

    Subclasses implement :meth:`generate` and expose their supported models
    via :meth:`models`. API keys are read from environment variables only.
    """

    name = "base"
    requires_api_key = False

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig(provider=self.name)
        self._api_key: str | None = None

    def _read_env_key(self, var: str | None) -> str | None:
        if not var:
            return None
        value = os.environ.get(var)
        if value:
            self._api_key = value
        return value

    def _check_api_key(self, required_var: str) -> str:
        token = self._api_key or os.environ.get(required_var)
        if not token:
            raise ProviderAuthError(
                f"Missing API key. Set {required_var} in your environment "
                "or .env file."
            )
        return token

    @abstractmethod
    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        """Generate a completion for the given request dict."""

    def models(self) -> list[str]:
        """List supported model identifiers (may be populated lazily)."""
        return [self.config.model] if self.config.model else []

    def health(self) -> dict[str, Any]:
        """Return a lightweight capability report."""
        return {
            "provider": self.name,
            "configured": self._api_key is not None or not self.requires_api_key,
            "models": self.models(),
        }

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Fill defaults for a provider request."""
        normalized = dict(request)
        normalized.setdefault("model", self.config.model)
        normalized.setdefault("max_tokens", None)
        return normalized


class FakeProviderMixin:
    """Mixin for deterministic fake providers used in tests and benchmarks.

    A fake provider returns content derived from the request so no external
    API is needed. It must not be used to fabricate production claims.
    """

    name = "fake"

    def __init__(self) -> None:
        self.config = ProviderConfig(provider=self.name)

    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        model = request.get("model") or (self.config.model or "fake-model")
        prompt = request.get("prompt", "")
        content = f"fake-response-for: {prompt[:40]}"
        return ProviderResponse(
            content=content,
            model=model,
            provider=f"fake:{self.name}",
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(content.split())),
            latency_ms=1.0,
            metadata={"deterministic": True},
        )

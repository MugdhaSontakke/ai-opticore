"""Abstract model provider interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from opticore.core.config import ProviderConfig
from opticore.core.model import AIResponse, ProviderResponse
from opticore.exceptions import ProviderAuthError, ProviderError, ProviderRateError
from opticore.logging import get_logger

if TYPE_CHECKING:
    from opticore.core.model import AIRequest

__all__ = [
    "AIResponse",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateError",
    "ProviderResponse",
    "BaseProvider",
    "FakeProviderMixin",
]

logger = get_logger("opticore.providers")


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
    def generate(self, request: dict[str, Any]) -> AIResponse:
        """Generate a completion for the given request dict."""

    def generate_from_request(self, request: AIRequest) -> AIResponse:
        """Generate directly from an :class:`AIRequest`."""
        return self.generate(request.to_dict())

    def models(self) -> list[str]:
        """List supported model identifiers (may be populated lazily)."""
        return [self.config.model] if self.config.model else []

    def health(self, model: str | None = None, probe: bool = False) -> dict[str, Any]:
        """Return a lightweight capability report.

        ``model`` (optional) restricts the report to that model and ``probe``
        requests a live connectivity check. Subclasses (e.g. Ollama) perform
        real probes; key-based providers only report configuration, so a wrong
        key is surfaced by :meth:`generate`, not here.
        """
        report: dict[str, Any] = {
            "provider": self.name,
            "configured": self._api_key is not None or not self.requires_api_key,
            "models": self.models(),
        }
        if model:
            report["requested_model"] = model
        return report

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

    def generate(self, request: dict[str, Any]) -> AIResponse:
        model = request.get("model") or (self.config.model or "fake-model")
        prompt = request.get("prompt", "")
        content = f"fake-response-for: {prompt[:40]}"
        return AIResponse(
            content=content,
            model=model,
            provider=f"fake:{self.name}",
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(content.split())),
            latency_ms=1.0,
            metadata={"deterministic": True},
        )

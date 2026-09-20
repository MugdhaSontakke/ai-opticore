"""OpenAI-compatible provider via the official SDK.

Also usable for any OpenAI-compatible endpoint (Azure, vLLM, llama.cpp
server, etc.) by overriding ``base_url``.

Production contract:
- The SDK's own retries are disabled (``max_retries=0``) so this library owns
  the retry policy, backoff, and jitter (see :mod:`opticore.providers.retry`).
- Errors are mapped to the typed hierarchy in :mod:`opticore.exceptions`.
- ``base_url`` passes through SSRF/credential validation
  (:func:`opticore.security.validate_base_url`).
- Exception messages are redacted so API keys never leak.
"""

from __future__ import annotations

import time
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from opticore.logging import get_logger
from opticore.providers.base import BaseProvider, ProviderResponse
from opticore.providers.retry import (
    NON_RETRYABLE_STATUS,
    RETRYABLE_STATUS,
    RetryPolicy,
    retry_call,
)
from opticore.security import redact_text, validate_base_url

logger = get_logger("opticore.providers.openai")

_VALID_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1",
    "o3-mini",
}


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI and OpenAI-compatible APIs.

    The OpenAI library is imported lazily so this module can be imported even
    when the package is not installed. The API key is read from the
    ``OPENAI_API_KEY`` environment variable (or ``api_key_env``).
    """

    name = "openai"
    requires_api_key = True

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(config or ProviderConfig(provider="openai"))
        if model:
            self.config.model = model
        if self.config.api_key_env is None:
            self.config.api_key_env = "OPENAI_API_KEY"
        if self.config.base_url:
            self.config.base_url = validate_base_url(
                self.config.base_url,
                allow_private_networks=self.config.allow_private_networks,
                allowed_hosts=self.config.allowed_hosts,
            )
        self._client: Any = None
        self._read_env_key(self.config.api_key_env)

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.config.max_attempts,
            base_delay_seconds=self.config.backoff_base_seconds,
            max_delay_seconds=self.config.backoff_max_seconds,
        )

    def _get_client(self) -> Any:
        """Return a lazily-created, reusable OpenAI client."""
        from openai import OpenAI

        if self._client is None:
            key = self._check_api_key(self.config.api_key_env or "OPENAI_API_KEY")
            self._client = OpenAI(
                api_key=key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                # We own retrying; the SDK must not.
                max_retries=0,
            )
        return self._client

    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        client = self._get_client()
        req = self._normalize_request(request)
        kwargs = self._build_kwargs(req)

        def call() -> Any:
            return client.chat.completions.create(**kwargs)

        started = time.monotonic()
        try:
            completion = retry_call(
                call,
                policy=self._retry_policy(),
                classify=_classify_openai_error,
                finalize=_finalize_openai_error,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak raw SDK exceptions
            raise ProviderError(
                f"OpenAI request failed: {redact_text(str(exc) or type(exc).__name__)}"
            ) from exc

        if not getattr(completion, "choices", None):
            raise ProviderResponseError(
                f"OpenAI response had no completion choices (model={req['model']!r}). "
                "The endpoint may be returning a non-chat response."
            )
        try:
            choice = completion.choices[0]
            usage = completion.usage
            content = choice.message.content or ""
            return ProviderResponse(
                content=content,
                model=completion.model or req["model"],
                provider=self.name,
                input_tokens=(
                    usage.prompt_tokens
                    if usage
                    else request.get("_input_tokens", 0)
                ),
                output_tokens=usage.completion_tokens if usage else 0,
                latency_ms=elapsed_ms,
                metadata={"finish_reason": choice.finish_reason or ""},
            )
        except ProviderResponseError:
            raise
        except Exception as exc:  # noqa: BLE001 - schema drift becomes typed error
            raise ProviderResponseError(
                f"OpenAI response could not be parsed: "
                f"{redact_text(str(exc) or type(exc).__name__)}"
            ) from exc

    def _build_kwargs(self, req: dict[str, Any]) -> dict[str, Any]:
        messages = self._build_messages(req)
        return {
            "model": req["model"] or (self.config.model or "gpt-4o-mini"),
            "messages": messages,
            "max_tokens": req.get("max_tokens"),
            "temperature": req.get("temperature"),
        } | ({"tools": req["tools"]} if req.get("tools") else {})

    @staticmethod
    def _build_messages(req: dict[str, Any]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if req.get("system"):
            messages.append({"role": "system", "content": req["system"]})
        if req.get("messages"):
            messages.extend(req["messages"])
        else:
            messages.append({"role": "user", "content": req.get("prompt", "")})
        return messages

    def models(self) -> list[str]:
        if self.config.model is None:
            return list(_VALID_MODELS)
        return [self.config.model]


def _classify_openai_error(exc: Exception) -> str:
    """Map an OpenAI SDK exception to a failure kind for the retry policy."""
    # SDK type-based checks first (works with the real SDK and with pluggable
    # OpenAI-compatible clients that subclass or alias these names).
    try:
        import openai

        checks = (
            ("APITimeoutError", "timeout"),
            ("APIConnectionError", "connection"),
            ("RateLimitError", "rate"),
            ("AuthenticationError", "auth"),
            ("PermissionDeniedError", "auth"),
        )
        for name, kind in checks:
            cls = getattr(openai, name, None)
            if cls is not None and isinstance(exc, cls):
                return kind
    except Exception:  # noqa: BLE001 - classifier must never crash the request
        pass

    exc_name = type(exc).__name__
    if exc_name in ("APITimeoutError", "TimeoutError"):
        return "timeout"
    if exc_name == "APIConnectionError":
        return "connection"
    if exc_name in ("RateLimitError",):
        return "rate"

    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status in NON_RETRYABLE_STATUS:
            return "invalid"
        if status in RETRYABLE_STATUS:
            return "server"

    # Keyword fallback for providers/clients that wrap SDK errors in their
    # own exception types (kept in sync with the typed hierarchy).
    text = str(exc).lower()
    if any(k in text for k in ("timed out", "timeout", "deadline exceeded")):
        return "timeout"
    if any(k in text for k in ("connection", "connectionerror", "refused", "econnrefused")):
        return "connection"
    if any(k in text for k in ("rate limit", "rate_limit", "too many requests", "quota")):
        return "rate"
    if any(k in text for k in ("incorrect api key", "authentication", "invalid api key", "unauthorized", "permission denied", "api key")):
        return "auth"

    if exc_name in ("NotFoundError", "BadRequestError", "UnprocessableEntityError", "ConflictError"):
        return "invalid"
    return "unknown"


def _finalize_openai_error(kind: str, exc: Exception) -> ProviderError:
    message = redact_text(str(exc) or type(exc).__name__)
    if kind == "auth":
        return ProviderAuthError(message)
    if kind == "rate":
        return ProviderRateError(message)
    if kind == "timeout":
        return ProviderTimeoutError(message)
    if kind in ("connection", "server"):
        return ProviderUnavailableError(message)
    return ProviderError(message)

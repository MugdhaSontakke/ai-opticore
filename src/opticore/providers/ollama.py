"""Ollama provider for locally-hosted models.

Production contract:
- The HTTP path retries transient failures (connection errors, 408/429/5xx)
  with the shared backoff policy in :mod:`opticore.providers.retry`.
- Errors are mapped to the typed hierarchy; malformed JSON bodies become
  :class:`ProviderResponseError`.
- ``base_url`` passes through SSRF/credential validation (private networks
  are allowed by default since Ollama runs locally).
- Exception messages are redacted.
"""

from __future__ import annotations

import os
import time
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.exceptions import (
    ProviderError,
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

logger = get_logger("opticore.providers.ollama")


class OllamaProvider(BaseProvider):
    """Provider for Ollama' local REST API (http://localhost:11434 by default).

    No API key required. Uses the official ``ollama`` python client when
    available, otherwise falls back to the HTTP endpoint via ``requests``.
    """

    name = "ollama"

    def __init__(
        self,
        config: ProviderConfig | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(config or ProviderConfig(provider="ollama"))
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
        if model:
            self.config.model = model
        if self.config.base_url and self.config.base_url != "openai":
            self.base_url = self.config.base_url
        self.base_url = validate_base_url(
            self.base_url,
            allow_private_networks=self.config.allow_private_networks,
            allowed_hosts=self.config.allowed_hosts,
        )

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.config.max_attempts,
            base_delay_seconds=self.config.backoff_base_seconds,
            max_delay_seconds=self.config.backoff_max_seconds,
        )

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "endpoint": self.base_url,
            "configured": True,
        }

    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        req = self._normalize_request(request)
        model = req["model"] or (self.config.model or "llama3.2")
        prompt = req.get("prompt") or self._messages_to_text(req.get("messages"))

        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError:
            return self._generate_via_http(model, prompt, req)

        started = time.monotonic()
        try:
            resp = retry_call(
                lambda: ollama.chat(  # type: ignore[attr-defined]
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "num_predict": req.get("max_tokens"),
                        "temperature": req.get("temperature"),
                    },
                ),
                policy=self._retry_policy(),
                classify=_classify_ollama_error,
                finalize=_finalize_ollama_error,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak raw client errors
            raise ProviderError(
                f"Ollama chat failed: {redact_text(str(exc) or type(exc).__name__)}"
            ) from exc

        if not isinstance(resp, dict) or "message" not in resp:
            raise ProviderResponseError(
                "Ollama returned an unexpected response shape "
                f"(expected dict with 'message', got {type(resp).__name__})."
            )
        raw = resp.get("message", {}) or {}
        return ProviderResponse(
            content=raw.get("content", "") if isinstance(raw, dict) else "",
            model=model,
            provider=self.name,
            input_tokens=resp.get("prompt_eval_count", 0) or 0,
            output_tokens=resp.get("eval_count", 0) or 0,
            latency_ms=elapsed_ms,
        )

    def _generate_via_http(
        self, model: str, prompt: str, req: dict[str, Any]
    ) -> ProviderResponse:
        import requests  # type: ignore[import-untyped]

        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "num_predict": req.get("max_tokens"),
                "temperature": req.get("temperature"),
            },
            "stream": False,
        }

        def call() -> dict[str, Any]:
            resp = requests.post(
                url, json=payload, timeout=self.config.timeout_seconds
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ProviderResponseError(
                    "Ollama /api/chat returned a non-object JSON body "
                    f"({type(data).__name__})."
                )
            return data

        started = time.monotonic()
        try:
            data = retry_call(
                call,
                policy=self._retry_policy(),
                classify=_classify_ollama_error,
                finalize=_finalize_ollama_error,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
        except ProviderError:
            raise
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.config.timeout_seconds}s"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise ProviderUnavailableError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is the server running?"
            ) from exc
        except requests.exceptions.InvalidJSONError as exc:
            raise ProviderResponseError(
                "Ollama /api/chat returned unparseable JSON."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(
                f"Ollama HTTP request failed: {redact_text(str(exc))}"
            ) from exc

        message = data.get("message") if isinstance(data, dict) else None
        return ProviderResponse(
            content=message.get("content", "") if isinstance(message, dict) else "",
            model=model,
            provider=self.name,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            latency_ms=elapsed_ms,
        )

    def _messages_to_text(self, messages: Any) -> str:
        if not messages:
            return ""
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

    def models(self) -> list[str]:
        try:
            import ollama

            return [m["name"] for m in ollama.list().get("models", [])]
        except Exception:  # noqa: BLE001
            return []


def _classify_ollama_error(exc: Exception) -> str:
    """Map an Ollama/requests exception to a retry-kind string."""
    import requests  # type: ignore[import-untyped]

    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, (requests.exceptions.ConnectionError,)):
        return "connection"
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status in NON_RETRYABLE_STATUS:
            return "invalid"
        if status in RETRYABLE_STATUS:
            return "server"
    if isinstance(exc, requests.exceptions.HTTPError):
        # HTTPError without a recognized status -> treat as server, it may be
        # a proxy/egress failure worth one more attempt.
        return "server"
    return "unknown"


def _finalize_ollama_error(kind: str, exc: Exception) -> ProviderError:
    message = redact_text(str(exc) or type(exc).__name__)
    if kind == "timeout":
        return ProviderTimeoutError(message)
    if kind == "connection":
        return ProviderUnavailableError(message)
    if kind == "server":
        return ProviderUnavailableError(message)
    return ProviderError(message)

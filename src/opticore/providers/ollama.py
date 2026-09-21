"""Ollama provider for locally-hosted models.

Production contract:
- The HTTP path retries transient failures (connection errors, 408/429/5xx)
  with the shared backoff policy in :mod:`opticore.providers.retry`.
- Errors are mapped to the typed hierarchy; malformed JSON bodies become
  :class:`ProviderResponseError`.
- ``base_url`` passes through SSRF/credential validation (private networks
  are allowed by default since Ollama runs locally).
- Exception messages are redacted.
- :meth:`OllamaProvider.health` performs a real, staged connectivity check
  (reachable, installed models, requested model present, optional generation
  probe) that never raises for an offline server.
"""

from __future__ import annotations

import os
import time
import warnings
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


def _import_requests():
    """Import ``requests`` while suppressing the environmental urllib3 ssl warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"urllib3 v2 only supports OpenSSL.*",
            category=Warning,
        )
        import requests  # type: ignore[import-untyped]

        return requests


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

    def _default_model(self) -> str:
        """Resolve the model to use: explicit > config > ``OLLAMA_MODEL`` env."""
        return (
            self.config.model
            or os.environ.get("OLLAMA_MODEL")
            or "llama3.2"
        )

    def health(
        self, model: str | None = None, probe: bool = False
    ) -> dict[str, Any]:
        """Run a live health check against the Ollama endpoint.

        The check is staged so each failure is reported independently:

        1. Is the server reachable? (``GET /api/version``)
        2. What models are installed? (``GET /api/tags``)
        3. Is the requested model installed?
        4. Can a small generation complete? (``POST /api/generate`` when
           ``probe=True``)

        Never raises for an offline server or a missing model; the result
        carries a structured ``error_kind`` (``not_running``, ``timeout``,
        ``http:<status>``, ``malformed_response``, ``model_not_found``) so
        callers can render distinct diagnostics. Generation output is
        truncated and never stored.
        """
        target = model or self.config.model or os.environ.get("OLLAMA_MODEL")
        report: dict[str, Any] = {
            "provider": self.name,
            "endpoint": self.base_url,
            "configured": True,
            "reachable": False,
            "ollama_version": None,
            "installed_models": [],
            "requested_model": target,
            "model_installed": None,
            "probe": None,
            "error_kind": None,
            "error": None,
            "healthy": False,
        }
        data, kind, error = self._probe("GET", "/api/version")
        if kind is not None:
            report["error_kind"], report["error"] = kind, error
            return report
        report["reachable"] = True
        assert data is not None  # guaranteed: kind is None above
        report["ollama_version"] = data.get("version")
        tags, kind, error = self._probe("GET", "/api/tags")
        if kind is not None:
            report["error_kind"], report["error"] = kind, error
            return report
        report["installed_models"] = _tags_to_names(tags)
        if target:
            report["model_installed"] = any(
                self._model_matches(target, name)
                for name in report["installed_models"]
            )
            if not report["model_installed"]:
                report["error_kind"] = "model_not_found"
                report["error"] = (
                    f"model '{target}' is not installed. Run: ollama pull {target}"
                )
                return report
        if probe and target and report["model_installed"]:
            start = time.monotonic()
            gen, kind, error = self._probe(
                "POST",
                "/api/generate",
                {
                    "model": target,
                    "prompt": "Reply with exactly one word: ok",
                    "stream": False,
                    "options": {"num_predict": 8},
                },
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            report["probe"] = {
                "ok": kind is None,
                "elapsed_ms": round(elapsed_ms, 2),
                "output_tokens": gen.get("eval_count", 0) if gen else 0,
                "response_preview": _preview(gen.get("response", "") if gen else ""),
                "error_kind": kind,
                "error": error,
            }
            if kind is not None:
                report["error_kind"], report["error"] = kind, error
                return report
        report["error_kind"] = None
        report["error"] = None
        report["healthy"] = report["reachable"] and (
            not target or report["model_installed"]
        ) and (report["probe"] is None or bool(report["probe"]["ok"]))
        return report

    def _probe(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        """Single-shot HTTP probe; ``(data, error_kind, error)``.

        Never raises for transport or parsing failures. ``error_kind`` is one
        of ``not_running``, ``timeout``, ``http:<status>``,
        ``malformed_response``, ``error``.
        """
        requests = _import_requests()

        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            if payload is None:
                resp = requests.get(url, timeout=self.config.timeout_seconds)
            else:
                resp = requests.post(
                    url, json=payload, timeout=self.config.timeout_seconds
                )
        except requests.exceptions.Timeout:
            return (
                None,
                "timeout",
                f"request timed out after {self.config.timeout_seconds}s",
            )
        except requests.exceptions.ConnectionError:
            return None, "not_running", f"cannot connect to {self.base_url}"
        except requests.exceptions.RequestException as exc:
            return (
                None,
                "error",
                redact_text(str(exc) or type(exc).__name__),
            )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            detail = ""
            try:
                body = resp.json()
                if isinstance(body, dict) and body.get("error"):
                    detail = str(body["error"])
            except ValueError:
                pass
            return (
                None,
                f"http:{resp.status_code}",
                redact_text(detail or f"HTTP {resp.status_code}"),
            )
        try:
            data = resp.json()
        except ValueError:
            return (None, "malformed_response", "response body was not valid JSON")
        if not isinstance(data, dict):
            return (
                None,
                "malformed_response",
                f"response body was not a JSON object ({type(data).__name__})",
            )
        return data, None, None

    @staticmethod
    def _model_matches(requested: str, installed: str) -> bool:
        if installed == requested:
            return True
        return installed.split(":", 1)[0] == requested.split(":", 1)[0]

    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        req = self._normalize_request(request)
        model = req["model"] or self._default_model()
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
        requests = _import_requests()

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
        """Installed models via ``GET /api/tags`` (fast, no SDK required).

        Returns ``[]`` when the server is offline; ``ai-opticore doctor`` and
        :meth:`health` surface the distinction for diagnostics.
        """
        try:
            data, kind, _ = self._probe("GET", "/api/tags")
        except Exception:  # noqa: BLE001 - never break CLI listing on transport hiccups
            return []
        if kind is not None or data is None:
            return []
        return _tags_to_names(data)


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


def _tags_to_names(data: dict[str, Any] | None) -> list[str]:
    """Extract model names from a ``/api/tags`` response body."""
    if not data:
        return []
    names: list[str] = []
    for item in data.get("models", []) or []:
        name = item.get("model", item.get("name", "")) if isinstance(item, dict) else item
        if name:
            names.append(str(name))
    return names


def _preview(text: str, limit: int = 80) -> str:
    if not text:
        return ""
    return f"{text[:limit]}..." if len(text) > limit else text

"""Ollama provider for locally-hosted models."""

from __future__ import annotations

import os
import time
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.exceptions import ProviderTimeoutError
from opticore.logging import get_logger
from opticore.providers.base import BaseProvider, ProviderError, ProviderResponse

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
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": req.get("max_tokens"),
                    "temperature": req.get("temperature"),
                },
            )
            elapsed_ms = (time.monotonic() - started) * 1000
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Ollama chat failed: {exc}") from exc

        raw = resp.get("message", {})
        return ProviderResponse(
            content=raw.get("content", ""),
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
        started = time.monotonic()
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = (time.monotonic() - started) * 1000
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.config.timeout_seconds}s"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise ProviderError(
                f"Ollama HTTP error {getattr(exc.response, 'status_code', '?')}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Ollama HTTP request failed: {exc}") from exc
        return ProviderResponse(
            content=data.get("message", {}).get("content", ""),
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

"""OpenAI-compatible provider via the official SDK.

Also usable for any OpenAI-compatible endpoint (Azure, vLLM, llama.cpp
server, etc.) by overriding ``base_url``.
"""

from __future__ import annotations

import time
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.logging import get_logger
from opticore.providers.base import (
    BaseProvider,
    ProviderAuthError,
    ProviderRateError,
    ProviderResponse,
)

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
        self._read_env_key(self.config.api_key_env)

    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        from openai import OpenAI

        key = self._check_api_key(self.config.api_key_env or "OPENAI_API_KEY")
        client = OpenAI(
            api_key=key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        req = self._normalize_request(request)
        messages = self._build_messages(req)
        started = time.monotonic()
        try:
            completion = client.chat.completions.create(
                model=req["model"] or (self.config.model or "gpt-4o-mini"),
                messages=messages,
                max_tokens=req.get("max_tokens"),
                temperature=req.get("temperature"),
            )
            elapsed_ms = (time.monotonic() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - surface as ProviderError
            text = str(exc)
            if "api key" in text.lower() or "authentication" in text.lower():
                raise ProviderAuthError(text) from exc
            if "rate" in text.lower() or "quota" in text.lower():
                raise ProviderRateError(text) from exc
            raise

        choice = completion.choices[0]
        usage = completion.usage
        return ProviderResponse(
            content=choice.message.content or "",
            model=completion.model or req["model"],
            provider=self.name,
            input_tokens=usage.prompt_tokens if usage else request.get("_input_tokens", 0),
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
            metadata={"finish_reason": choice.finish_reason or ""},
        )

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

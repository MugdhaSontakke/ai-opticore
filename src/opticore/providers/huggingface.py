"""Hugging Face (local) provider via transformers + PyTorch."""

from __future__ import annotations

import time
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.exceptions import ProviderError, ProviderResponseError
from opticore.logging import get_logger
from opticore.providers.base import BaseProvider, ProviderResponse
from opticore.security import redact_text

logger = get_logger("opticore.providers.huggingface")


class HuggingFaceProvider(BaseProvider):
    """Runs a Hugging Face model locally with ``transformers``.

    Loading a model is expensive and memory-heavy, so this provider loads
    lazily on first use and keeps the model in memory. Supply ``tokenizer``
    and ``model_cls`` for non-causal or custom models in tests.
    """

    name = "huggingface"

    def __init__(
        self,
        config: ProviderConfig | None = None,
        model: str | None = None,
        device: str = "auto",
        load_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or ProviderConfig(provider="huggingface"))
        if model:
            self.config.model = model
        self.device = device
        self.load_options = load_options or {}
        self._pipeline = None
        self._loaded_model = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError(
                "transformers is not installed; run `pip install "
                "ai-opticore[huggingface]`"
            ) from exc

        model_id = self.config.model
        if not model_id:
            raise ProviderError("HuggingFaceProvider requires a model id")
        logger.info("loading HF model %s (this can take a while)", model_id)
        self._pipeline = pipeline(
            "text-generation",
            model=model_id,
            device=self.device,
            **self.load_options,
        )

    def generate(self, request: dict[str, Any]) -> ProviderResponse:
        self._load()
        assert self._pipeline is not None
        req = self._normalize_request(request)
        prompt = req.get("prompt") or ""
        temperature = req.get("temperature")
        max_new_tokens = req.get("max_tokens") or 128
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature is not None,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        started = time.monotonic()
        try:
            outputs = self._pipeline(prompt, **kwargs)
            elapsed_ms = (time.monotonic() - started) * 1000
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"HuggingFace generation failed: {redact_text(str(exc))}"
            ) from exc

        if not outputs:
            raise ProviderResponseError(
                "HuggingFace pipeline returned no outputs for the prompt."
            )
        first = outputs[0]
        if not isinstance(first, dict) or "generated_text" not in first:
            raise ProviderResponseError(
                "HuggingFace pipeline returned an unexpected output shape "
                f"({type(first).__name__}); expected a dict with 'generated_text'."
            )
        generated = first.get("generated_text") or ""
        content = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
        model = self.config.model or "unknown"
        return ProviderResponse(
            content=content,
            model=model,
            provider=self.name,
            input_tokens=self._count_tokens(prompt),
            output_tokens=self._count_tokens(content) if content else 0,
            latency_ms=elapsed_ms,
            metadata={
                "device": self.device,
                "loaded": True,
                "token_counts": "tokenizer"
                if self._tokenizer_available()
                else "unmeasured",
                "sampling": "sampled" if temperature is not None else "greedy",
            },
        )

    def _tokenizer_available(self) -> bool:
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        return tokenizer is not None and hasattr(tokenizer, "encode")

    def _count_tokens(self, text: str) -> int:
        """Token counts from the model's real tokenizer when available."""
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text))
            except Exception:  # noqa: BLE001 - never fabricate counts
                return 0
        return 0

    def models(self) -> list[str]:
        if self.config.model:
            return [self.config.model]
        return []

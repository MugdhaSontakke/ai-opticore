"""Hugging Face (local) provider via transformers + PyTorch."""

from __future__ import annotations

import time
from typing import Any

from opticore.core.config import ProviderConfig
from opticore.logging import get_logger
from opticore.providers.base import BaseProvider, ProviderError, ProviderResponse

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
        started = time.monotonic()
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=req.get("max_tokens") or 128,
                do_sample=False,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"HuggingFace generation failed: {exc}") from exc

        generated = outputs[0]["generated_text"] if outputs else prompt
        content = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
        model = self.config.model or "unknown"
        return ProviderResponse(
            content=content,
            model=model,
            provider=self.name,
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()) if content else 0,
            latency_ms=elapsed_ms,
            metadata={"device": self.device, "loaded": True},
        )

    def models(self) -> list[str]:
        if self.config.model:
            return [self.config.model]
        return []

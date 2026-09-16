"""High-level Python API.

Two entry points:
  - :class:`Optimizer` for pure prompt/context optimization (no provider).
  - :class:`AIClient` for an end-to-end optimized generation pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opticore.benchmarks.metrics import MetricsCollector, accumulate_tokens
from opticore.cache.base import MemoryCache, cache_key
from opticore.cache.semantic import SemanticCache
from opticore.core.config import OptimizationConfig
from opticore.core.pipeline import OptimizationPipeline, build_default_pipeline
from opticore.providers import get_provider
from opticore.providers.base import BaseProvider, ProviderResponse

EmbeddingFn = Callable[[str], list[float]]


@dataclass
class OptimizeOutcome:
    """Return type for :meth:`Optimizer.optimize`."""

    optimized_prompt: str
    tokens_saved: int
    reduction_percent: float
    original_tokens: int
    optimized_tokens: int
    optimizers_run: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def warnings_text(self) -> str:
        return "; ".join(self.warnings)


class Optimizer:
    """Optimizes prompts and context without calling any model provider."""

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        pipeline: OptimizationPipeline | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.pipeline = pipeline or build_default_pipeline(self.config)

    def optimize(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> OptimizeOutcome:
        result = self.pipeline.run(prompt=prompt or "", system=system, messages=messages)
        warnings: list[str] = []
        for meta in result.metadata.values():
            if isinstance(meta, dict):
                severity = meta.get("severity")
                if severity and severity != "low":
                    warnings.append(f"{result.optimizer_name}: {severity}")
        return OptimizeOutcome(
            optimized_prompt=result.optimized_request.prompt,
            tokens_saved=result.tokens_saved,
            reduction_percent=round(result.reduction_percent, 2),
            original_tokens=result.original_tokens,
            optimized_tokens=result.optimized_tokens,
            optimizers_run=result.optimizers_run,
            warnings=warnings,
        )


@dataclass
class GenerationResult:
    """Full outcome of an :meth:`AIClient.generate` call."""

    content: str
    model: str
    provider: str
    cache_hit: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    original_tokens: int = 0
    optimized_tokens: int = 0
    tokens_saved: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "cache_hit": self.cache_hit,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "original_tokens": self.original_tokens,
            "optimized_tokens": self.optimized_tokens,
            "tokens_saved": self.tokens_saved,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class AIClient:
    """End-to-end client: optimization pipeline -> semantic cache -> provider.

    When ``optimization=False`` the client short-circuits the pipeline and
    behaves as a thin provider wrapper (a baseline reference).
    """

    def __init__(
        self,
        provider: BaseProvider | str,
        *,
        config: OptimizationConfig | None = None,
        embedding_fn: EmbeddingFn | None = None,
        cache: SemanticCache | MemoryCache | None = None,
        model: str | None = None,
        optimization: bool = True,
    ) -> None:
        if isinstance(provider, str):
            self.provider = get_provider(provider)
            if model:
                self.provider.config.model = model
        else:
            self.provider = provider
            if model:
                self.provider.config.model = model
        self.config = config or OptimizationConfig()
        self.metrics = MetricsCollector()
        self._optimization_enabled = optimization

        if cache is None:
            if self.config.enable_semantic_cache and embedding_fn is not None:
                cache = SemanticCache(
                    embedding_fn=embedding_fn,
                    threshold=self.config.semantic_cache_threshold,
                    ttl_seconds=float(self.config.cache_ttl_seconds),
                )
            else:
                cache = MemoryCache()
        self.cache = cache
        self.pipeline = build_default_pipeline(self.config)

    def generate(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Optimize and generate a response (with optional cache lookup)."""
        self.metrics.incr("request_count", 1)
        provider_config = getattr(self.provider, "config", None)
        provider_model = getattr(provider_config, "model", None)
        active_model = model or provider_model or "unknown"

        if self._optimization_enabled:
            result = self.pipeline.run(
                prompt=prompt or "", system=system, messages=messages
            )
            accumulate_tokens(
                self.metrics, result.original_tokens, result.optimized_tokens
            )
            send_prompt = result.optimized_request.prompt
            send_system = result.optimized_request.system
            send_messages = result.optimized_request.messages
            original_tokens = result.original_tokens
            optimized_tokens = result.optimized_tokens
            tokens_saved = result.tokens_saved
            optimizers_run = result.optimizers_run
        else:
            send_prompt = prompt or ""
            send_system = system
            send_messages = messages
            original_tokens = optimized_tokens = tokens_saved = 0
            optimizers_run = []

        cache_hit = False
        response: ProviderResponse | None = None
        content = ""
        latency_ms = 0.0

        if self.config.enable_semantic_cache:
            hit, entry = self._cache_lookup(send_prompt, send_system, active_model)
            if hit and entry is not None:
                cache_hit = True
                self.metrics.incr("cache_hits", 1)
                content = entry.content
            else:
                self.metrics.incr("cache_misses", 1)

        if not cache_hit:
            started = time.monotonic()
            response = self.provider.generate(
                {
                    "prompt": send_prompt,
                    "system": send_system,
                    "messages": send_messages,
                    "model": active_model,
                    "max_tokens": max_tokens,
                }
            )
            latency_ms = (time.monotonic() - started) * 1000.0
            content = response.content
            self._store_response(send_prompt, send_system, active_model, content)
        else:
            self.metrics.incr("cache_served_requests", 1)

        self.metrics.incr("model_usage", 1 if not cache_hit else 0)
        return GenerationResult(
            content=content,
            model=active_model or "unknown",
            provider=self.provider.name,
            cache_hit=cache_hit,
            input_tokens=response.input_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=tokens_saved,
            latency_ms=latency_ms,
            metadata={"optimizers_run": optimizers_run},
        )

    def _cache_lookup(
        self, text: str, system: str | None, model: str | None
    ) -> tuple[bool, Any]:
        if isinstance(self.cache, SemanticCache):
            return self.cache.lookup(text, model=model)
        key = cache_key(prompt=text, system=system, model=model)
        entry = self.cache.get(key)
        return (True, entry) if entry is not None else (False, None)

    def _store_response(
        self, text: str, system: str | None, model: str | None, content: str
    ) -> None:
        try:
            if isinstance(self.cache, SemanticCache):
                self.cache.put(text=text, content=content, model=model or "unknown")
            else:
                self.cache.set(
                    cache_key(prompt=text, system=system, model=model),
                    content,
                    model or "unknown",
                )
        except Exception:  # noqa: BLE001 - cache must never break generation
            pass

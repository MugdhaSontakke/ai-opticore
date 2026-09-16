"""High-level Python API.

Two entry points:
  - :class:`Optimizer` for pure prompt/context optimization (no provider).
  - :class:`AIClient` for an end-to-end optimized generation pipeline.

Timing contract: ``optimizer_time_ms`` (optimization overhead) is measured
separately from ``model_time_ms`` (provider call) so an optimization that
saves tokens but costs more wall-clock time than it saves is visible.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opticore.benchmarks.metrics import MetricsCollector, accumulate_tokens, token_metrics
from opticore.cache.base import MemoryCache, cache_key
from opticore.cache.semantic import SemanticCache
from opticore.core.config import OptimizationConfig
from opticore.core.model import AIRequest, AIResponse
from opticore.core.pipeline import OptimizationPipeline, build_default_pipeline
from opticore.providers import get_provider
from opticore.providers.base import BaseProvider

logger = logging.getLogger("opticore.api")

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
    changes: list[str] = field(default_factory=list)
    quality_verified: bool = False
    warnings: list[str] = field(default_factory=list)
    optimization_time_ms: float = 0.0
    accepted: bool = True
    rejection_reason: str | None = None

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
        model: str | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> OptimizeOutcome:
        result = self.pipeline.run(
            prompt=prompt or "",
            system=system,
            messages=messages,
            request=AIRequest(
                prompt=prompt or "",
                system=system,
                messages=messages,
                model=model,
                temperature=temperature,
                tools=tools,
            ),
        )
        warnings: list[str] = []
        for meta in result.metadata.values():
            if isinstance(meta, dict):
                severity = meta.get("severity")
                if severity and severity != "low":
                    warnings.append(f"{result.optimizer_name}: {severity}")
        quality = result.metadata.get("quality")
        quality_verified = bool(
            isinstance(quality, dict) and quality.get("verified")
        )
        return OptimizeOutcome(
            optimized_prompt=result.optimized_request.prompt,
            tokens_saved=result.tokens_saved,
            reduction_percent=round(result.reduction_percent, 2),
            original_tokens=result.original_tokens,
            optimized_tokens=result.optimized_tokens,
            optimizers_run=result.optimizers_run,
            changes=result.changes,
            quality_verified=quality_verified,
            warnings=warnings,
            optimization_time_ms=round(result.optimization_time_ms, 3),
            accepted=result.accepted,
            rejection_reason=result.rejection_reason,
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
    optimizer_time_ms: float = 0.0
    model_time_ms: float = 0.0
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "cache_hit": self.cache_hit,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "original_tokens": self.original_tokens,
            "optimized_tokens": self.optimized_tokens,
            "tokens_saved": self.tokens_saved,
            "reduction_percentage": round(
                (self.tokens_saved / self.original_tokens * 100.0)
                if self.original_tokens
                else 0.0,
                4,
            ),
            "optimizer_time_ms": self.optimizer_time_ms,
            "model_time_ms": self.model_time_ms,
            "total_time_ms": self.total_time_ms,
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
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        namespace: str | None = None,
    ) -> GenerationResult:
        """Optimize and generate a response (with optional cache lookup)."""
        self.metrics.incr("request_count", 1)
        provider_config = getattr(self.provider, "config", None)
        provider_model = getattr(provider_config, "model", None)
        active_model = model or provider_model or "unknown"

        optimizer_time_ms = 0.0
        request_meta: dict[str, Any] = {}
        no_cache = bool(request_meta.get("no_cache"))
        if self._optimization_enabled:
            opt_started = time.monotonic()
            result = self.pipeline.run(
                prompt=prompt or "", system=system, messages=messages
            )
            optimizer_time_ms = (time.monotonic() - opt_started) * 1000.0
            request_meta = result.metadata
            no_cache = bool(request_meta.get("no_cache"))
            accumulate_tokens(
                self.metrics, result.original_tokens, result.optimized_tokens
            )
            send = result.optimized_request
            original_tokens = result.original_tokens
            optimized_tokens = result.optimized_tokens
            tokens_saved = result.tokens_saved
            optimizers_run = result.optimizers_run
        else:
            send = AIRequest(prompt=prompt or "", system=system, messages=messages)
            original_tokens = optimized_tokens = tokens_saved = 0
            optimizers_run = []

        cache_hit = False
        response: AIResponse | None = None
        content = ""
        model_time_ms = 0.0

        if self.config.enable_semantic_cache and not no_cache:
            hit, entry = self._cache_lookup(
                send.prompt,
                system=send.system,
                model=active_model,
                temperature=temperature,
                max_tokens=max_tokens,
                namespace=namespace,
            )
            if hit and entry is not None:
                cache_hit = True
                self.metrics.incr("cache_hits", 1)
                content = entry.content
            else:
                self.metrics.incr("cache_misses", 1)

        if not cache_hit:
            payload = {
                "prompt": send.prompt,
                "system": send.system,
                "messages": send.messages,
                "model": active_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "tools": tools,
            }
            started = time.monotonic()
            response = self.provider.generate(payload)
            model_time_ms = (time.monotonic() - started) * 1000.0
            content = response.content
            if self.config.enable_semantic_cache and not no_cache:
                self._store_response(
                    send.prompt,
                    system=send.system,
                    model=active_model,
                    content=content,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    namespace=namespace,
                )
        else:
            self.metrics.incr("cache_served_requests", 1)

        self.metrics.incr("model_usage", 1 if not cache_hit else 0)
        self.metrics.record("optimizer_time_ms", optimizer_time_ms)
        self.metrics.record("model_time_ms", model_time_ms)
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
            optimizer_time_ms=round(optimizer_time_ms, 3),
            model_time_ms=round(model_time_ms, 3),
            total_time_ms=round(optimizer_time_ms + model_time_ms, 3),
            metadata={
                "optimizers_run": optimizers_run,
                **token_metrics(
                    original_tokens,
                    optimized_tokens,
                    output_tokens=response.output_tokens if response else 0,
                ),
            },
        )

    def _cache_lookup(
        self,
        text: str,
        system: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        namespace: str | None,
    ) -> tuple[bool, Any]:
        if isinstance(self.cache, SemanticCache):
            return self.cache.lookup(
                text,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                namespace=namespace,
            )
        key = cache_key(
            prompt=text,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            namespace=namespace,
        )
        entry = self.cache.get(key)
        return (True, entry) if entry is not None else (False, None)

    def _store_response(
        self,
        text: str,
        system: str | None,
        model: str | None,
        content: str,
        temperature: float | None,
        max_tokens: int | None,
        namespace: str | None,
    ) -> None:
        try:
            if isinstance(self.cache, SemanticCache):
                self.cache.put(
                    text=text,
                    content=content,
                    model=model or "unknown",
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    namespace=namespace,
                )
            else:
                self.cache.set(
                    cache_key(
                        prompt=text,
                        system=system,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        namespace=namespace,
                    ),
                    content,
                    model or "unknown",
                )
        except Exception as exc:  # noqa: BLE001 - cache failure must not kill generation
            logger.warning("cache write failed (generation unaffected): %s", exc)
            self.metrics.incr("cache_write_errors", 1)

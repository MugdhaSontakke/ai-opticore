"""High-level Python API.

Two entry points:
  - :class:`Optimizer` for pure prompt/context optimization (no provider).
  - :class:`AIClient` for an end-to-end optimized generation pipeline.

Timing contract: ``optimizer_time_ms`` (optimization overhead) is measured
separately from ``model_time_ms`` (provider call) so an optimization that
saves tokens but costs more wall-clock time than it saves is visible.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opticore.benchmarks.metrics import MetricsCollector, accumulate_tokens, token_metrics
from opticore.cache import DiskCache, MemoryCache, SemanticCache
from opticore.cache.base import cache_key
from opticore.core.config import OptimizationConfig
from opticore.core.model import AIRequest, AIResponse
from opticore.core.pipeline import OptimizationPipeline, build_default_pipeline
from opticore.costing import CostEstimator
from opticore.exceptions import RoutingError
from opticore.logging import get_logger
from opticore.optimizers.token import count_tokens
from opticore.providers import get_provider
from opticore.providers.base import BaseProvider
from opticore.routing.router import ModelRouter

logger = get_logger("opticore.api")

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
    optimized_system: str | None = None
    optimized_messages: list[dict[str, Any]] | None = None

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
            optimized_system=result.optimized_request.system,
            optimized_messages=result.optimized_request.messages,
        )


@dataclass
class GenerationResult:
    """Full outcome of an :meth:`AIClient.generate` call.

    Timing contract: ``optimizer_time_ms`` is optimization overhead,
    ``cache_lookup_ms`` is cache lookup time, ``model_time_ms`` is the provider
    call, and ``total_time_ms`` is end-to-end wall clock (includes metric
    bookkeeping). ``estimated_cost`` is only populated when the user configured
    ``pricing`` and is always labeled ``estimated``.
    """

    content: str
    model: str
    provider: str
    request_id: str = ""
    cache_hit: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    original_tokens: int = 0
    optimized_tokens: int = 0
    tokens_saved: int = 0
    optimizer_time_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    model_time_ms: float = 0.0
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    rejection_reason: str | None = None
    fallback_reason: str | None = None
    estimated_cost: dict[str, Any] = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)
    optimized_system: str | None = None
    optimized_messages: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "request_id": self.request_id,
            "cache_hit": self.cache_hit,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "fallback_reason": self.fallback_reason,
            "changes": self.changes,
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
            "cache_lookup_ms": self.cache_lookup_ms,
            "model_time_ms": self.model_time_ms,
            "total_time_ms": self.total_time_ms,
            "estimated_cost": self.estimated_cost,
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
        cache: SemanticCache | DiskCache | MemoryCache | None = None,
        model: str | None = None,
        optimization: bool = True,
        router: ModelRouter | None = None,
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
        self.router = router
        self.cost_estimator = CostEstimator(
            input_price_per_1k=self.config.pricing_input_per_1k,
            output_price_per_1k=self.config.pricing_output_per_1k,
        )

        if cache is None:
            if self.config.enable_semantic_cache and embedding_fn is not None:
                cache = SemanticCache(
                    embedding_fn=embedding_fn,
                    threshold=self.config.semantic_cache_threshold,
                    ttl_seconds=float(self.config.cache_ttl_seconds),
                )
            elif self.config.cache_backend == "disk":
                cache = DiskCache(
                    path=self.config.cache_disk_path
                    or os.path.join(os.getcwd(), ".opticore", "cache.sqlite"),
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
        no_cache: bool = False,
    ) -> GenerationResult:
        """Optimize and generate a response (with optional cache lookup)."""
        request_id = uuid.uuid4().hex[:12]
        total_started = time.monotonic()
        self.metrics.incr("request_count", 1)
        provider_config = getattr(self.provider, "config", None)
        provider_model = getattr(provider_config, "model", None)
        active_model = model or provider_model or "unknown"
        logger.debug(
            "request %s started provider=%s model=%s",
            request_id,
            self.provider.name,
            active_model,
        )

        fallback_reason: str | None = None
        routing_decision: dict[str, Any] | None = None
        if self.router is not None:
            token_count: int | None = None
            if self._optimization_enabled:
                token_count = count_tokens(
                    AIRequest(prompt=prompt or "", system=system, messages=messages),
                    self.pipeline.tokenizer,
                )
            try:
                decision = self.router.route(
                    prompt=prompt or "", token_count=token_count
                )
                active_model = decision.model or active_model
                routing_decision = {
                    "model": decision.model,
                    "reason": decision.reason,
                    "rules_considered": decision.rules_considered,
                }
                self.metrics.incr(f"routed_to_{decision.model}", 1)
            except RoutingError as exc:
                fallback_reason = f"routing failed: {exc}; using configured model"
                logger.warning(fallback_reason)
                self.metrics.incr("routing_failures", 1)

        optimizer_time_ms = 0.0
        accepted = True
        rejection_reason: str | None = None
        changes: list[str] = []
        optimized_system: str | None = None
        optimized_messages: list[dict[str, Any]] | None = None
        quality_report: dict[str, Any] | None = None
        if self._optimization_enabled:
            opt_started = time.monotonic()
            result = self.pipeline.run(
                prompt=prompt or "", system=system, messages=messages
            )
            optimizer_time_ms = (time.monotonic() - opt_started) * 1000.0
            accumulate_tokens(
                self.metrics, result.original_tokens, result.optimized_tokens
            )
            send = result.optimized_request
            original_tokens = result.original_tokens
            optimized_tokens = result.optimized_tokens
            tokens_saved = result.tokens_saved
            optimizers_run = result.optimizers_run
            accepted = result.accepted
            rejection_reason = result.rejection_reason
            changes = result.changes
            optimized_system = result.optimized_request.system
            optimized_messages = result.optimized_request.messages
            quality_report = (result.metadata or {}).get("quality")
            if isinstance(quality_report, dict) and quality_report.get("rejected"):
                self.metrics.incr("quality_gate_rejections", 1)
        else:
            send = AIRequest(prompt=prompt or "", system=system, messages=messages)
            original_tokens = optimized_tokens = tokens_saved = 0
            optimizers_run = []

        cache_hit = False
        response: AIResponse | None = None
        content = ""
        model_time_ms = 0.0
        cache_lookup_ms = 0.0

        if self.config.enable_semantic_cache and not no_cache:
            lookup_started = time.monotonic()
            try:
                hit, entry = self._cache_lookup(
                    send.prompt,
                    system=send.system,
                    model=active_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    namespace=namespace,
                )
            except Exception as exc:  # noqa: BLE001 - cache failure must not block the request
                hit, entry = False, None
                self.metrics.incr("cache_read_errors", 1)
                logger.warning(
                    "cache lookup failed (continuing to model); request %s: %s",
                    request_id,
                    exc,
                )
            cache_lookup_ms = (time.monotonic() - lookup_started) * 1000.0
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
        self.metrics.record("cache_lookup_ms", cache_lookup_ms)
        self.metrics.record("model_time_ms", model_time_ms)
        if fallback_reason:
            self.metrics.incr("fallbacks", 1)

        total_time_ms = (time.monotonic() - total_started) * 1000.0

        output_tokens = response.output_tokens if response else 0
        input_tokens = response.input_tokens if response else 0
        logger.info(
            "request %s done cache_hit=%s fallback=%s model_time_ms=%.1f "
            "optimizer_time_ms=%.1f total_time_ms=%.1f tokens_in=%d tokens_out=%d",
            request_id,
            cache_hit,
            bool(fallback_reason),
            model_time_ms,
            optimizer_time_ms,
            total_time_ms,
            input_tokens,
            output_tokens,
        )
        baseline_cost = self.cost_estimator.estimate(original_tokens, output_tokens)
        optimized_cost = self.cost_estimator.estimate(optimized_tokens, output_tokens)
        savings = self.cost_estimator.savings(baseline_cost, optimized_cost)
        if baseline_cost.total_cost is not None:
            self.metrics.record("estimated_cost_per_request", baseline_cost.total_cost)

        metadata: dict[str, Any] = {
            "request_id": request_id,
            "optimizers_run": optimizers_run,
            "optimization_mode": "enabled" if self._optimization_enabled else "disabled",
            "safety_mode": self.config.safety_mode.value,
            "quality": quality_report,
            "routing": routing_decision,
            "fallback_reason": fallback_reason,
            **token_metrics(
                original_tokens,
                optimized_tokens,
                output_tokens=output_tokens,
            ),
        }

        return GenerationResult(
            content=content,
            model=active_model or "unknown",
            provider=self.provider.name,
            request_id=request_id,
            cache_hit=cache_hit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=tokens_saved,
            optimizer_time_ms=round(optimizer_time_ms, 3),
            cache_lookup_ms=round(cache_lookup_ms, 3),
            model_time_ms=round(model_time_ms, 3),
            total_time_ms=round(total_time_ms, 3),
            metadata=metadata,
            accepted=accepted,
            rejection_reason=rejection_reason,
            fallback_reason=fallback_reason or rejection_reason,
            estimated_cost={
                "baseline": baseline_cost.to_dict(),
                "optimized": optimized_cost.to_dict(),
                "savings": savings.to_dict(),
            },
            changes=changes,
            optimized_system=optimized_system,
            optimized_messages=optimized_messages,
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
                    ttl_seconds=self.config.cache_ttl_seconds,
                )
        except Exception as exc:  # noqa: BLE001 - cache failure must not kill generation
            logger.warning("cache write failed (generation unaffected): %s", exc)
            self.metrics.incr("cache_write_errors", 1)

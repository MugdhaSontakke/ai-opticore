"""Real benchmark runner: measures baseline vs optimized performance.

The runner only reports what it actually measured. Unavailable metrics are
reported as ``N/A``. No fabricated results are ever produced.

Optimizer overhead is measured separately from model latency so that a run
which saves tokens but is slower overall is clearly visible.
"""

from __future__ import annotations

import logging
import os
import platform
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from opticore import __version__
from opticore.benchmarks.metrics import MetricsCollector, accumulate_tokens
from opticore.cache.base import MemoryCache, cache_key
from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest
from opticore.core.pipeline import build_default_pipeline
from opticore.hardware import detect_backend
from opticore.optimizers.token import Tokenizer
from opticore.providers.base import ProviderResponse

logger = logging.getLogger("opticore.benchmarks")


@dataclass
class BenchmarkResult:
    """Complete benchmark outcome for one sample set."""

    model: str
    provider: str
    hardware: str
    config: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    optimized: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: int = 0
    environment: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a human-readable report, as shown by the CLI."""
        env = self.environment
        lines = [
            "AI-OptiCore Benchmark",
            "--------------------",
            "",
            f"Model: {self.model}",
            f"Provider: {self.provider}",
            f"Hardware: {self.hardware}",
            f"Optimizer overhead (avg): {self._fmt_ms(self.metrics.get('optimizer_overhead_ms_avg'))}",
            f"Samples: {self.samples}",
            f"Timestamp: {env.get('timestamp', 'N/A')}",
            f"Version: {env.get('opticore_version', 'N/A')}",
            "",
            "Baseline (no optimization)",
            f"Input tokens: {self.baseline.get('input_tokens', 'N/A')}",
            f"Output tokens: {self.baseline.get('output_tokens', 'N/A')}",
            f"Latency: {self._fmt_ms(self.baseline.get('latency_ms'))}",
            "",
            "Optimized (AI-OptiCore enabled)",
            f"Input tokens: {self.optimized.get('input_tokens', 'N/A')}",
            f"Output tokens: {self.optimized.get('output_tokens', 'N/A')}",
            f"Latency: {self._fmt_ms(self.optimized.get('latency_ms'))}",
            "",
            f"Token reduction: {self.metrics.get('token_reduction_percent', 'N/A')}",
            f"Latency change: {self.metrics.get('latency_change_percent', 'N/A')}",
            f"Cache hit rate: {self.metrics.get('cache_hit_rate', 'N/A')}",
        ]
        if self.notes:
            lines += ["", "Notes:"]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (no raw prompts)."""
        return {
            "environment": self.environment,
            "model": self.model,
            "provider": self.provider,
            "hardware": self.hardware,
            "samples": self.samples,
            "config": self.config,
            "baseline": self.baseline,
            "optimized": self.optimized,
            "metrics": self.metrics,
            "notes": self.notes,
        }

    @staticmethod
    def _fmt_ms(value: Any) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "N/A"
        return f"{value:.1f} ms"


class BenchmarkRunner:
    """Runs repeated requests through baseline and optimized paths.

    The runner needs a provider callable; for environments without real APIs
    a deterministic fake provider can be supplied. Callers are responsible
    for declaring whether results came from a faked provider.
    """

    def __init__(
        self,
        provider: Any,
        model: str,
        samples: list[OptimizerRequest] | None = None,
        config: OptimizationConfig | None = None,
        repeats: int = 3,
        provider_notes: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.samples = samples or self._default_samples()
        self.config = config or OptimizationConfig()
        self.repeats = repeats
        self.provider_notes = provider_notes or []

    def run(self) -> BenchmarkResult:
        baseline_metrics: dict[str, Any] = {"input_tokens": [], "output_tokens": [], "latency_ms": []}
        optimized_metrics: dict[str, Any] = {"input_tokens": [], "output_tokens": [], "latency_ms": []}

        collector = MetricsCollector()
        optimizer_times: list[float] = []
        tokenizer = Tokenizer()
        self._cache: MemoryCache | None = (
            MemoryCache() if self.config.enable_semantic_cache else None
        )
        cache_hits = 0
        cache_misses = 0
        if getattr(self.provider, "name", "").startswith("fake"):
            self.provider_notes.append("deterministic fake provider; latency is not representative")

        for request in self.samples:
            for _ in range(self.repeats):
                base_input_tokens = self._request_tokens(request, tokenizer)
                latency, out_tokens, _ = self._call(request)
                baseline_metrics["latency_ms"].append(latency)
                baseline_metrics["output_tokens"].append(out_tokens)
                baseline_metrics["input_tokens"].append(base_input_tokens)

                pipeline = build_default_pipeline(self.config)
                opt_started = time.monotonic()
                pipeline_result = pipeline.run(
                    prompt=request.prompt,
                    system=request.system,
                    messages=request.messages,
                )
                optimizer_overhead_ms = (time.monotonic() - opt_started) * 1000.0
                optimizer_times.append(optimizer_overhead_ms)
                optimized_request = pipeline_result.optimized_request
                accumulate_tokens(
                    collector,
                    pipeline_result.original_tokens,
                    pipeline_result.optimized_tokens,
                )

                opt_latency = 0.0
                opt_out = 0.0
                if self._cache is not None:
                    key = cache_key(
                        prompt=optimized_request.prompt,
                        system=optimized_request.system,
                        model=self.model,
                    )
                    lookup_started = time.monotonic()
                    entry = self._cache.get(key)
                    if entry is not None:
                        cache_hits += 1
                        opt_latency = (time.monotonic() - lookup_started) * 1000.0
                        opt_out = float(entry.metadata.get("output_tokens", 0))
                    else:
                        cache_misses += 1
                        opt_latency, opt_out, content = self._call(optimized_request)
                        try:
                            self._cache.set(
                                key,
                                content,
                                self.model,
                                metadata={"output_tokens": opt_out},
                            )
                        except Exception:  # noqa: BLE001 - cache must not break benchmark
                            logger.warning("cache write failed during benchmark")
                else:
                    opt_latency, opt_out, _ = self._call(optimized_request)
                optimized_metrics["latency_ms"].append(opt_latency)
                optimized_metrics["output_tokens"].append(opt_out)
                optimized_metrics["input_tokens"].append(
                    pipeline_result.optimized_tokens
                )

        if cache_hits:
            self.provider_notes.append(
                "optimized run served some requests from the cache "
                "(cache hits have ~zero model latency)"
            )

        benchmark = BenchmarkResult(
            model=self.model,
            provider=getattr(self.provider, "name", "unknown"),
            hardware=detect_backend().backend,
            config=self.config.to_dict(),
            baseline={
                "input_tokens": _avg(baseline_metrics["input_tokens"]),
                "output_tokens": _avg(baseline_metrics["output_tokens"]),
                "latency_ms": _avg(baseline_metrics["latency_ms"]),
            },
            optimized={
                "input_tokens": _avg(optimized_metrics["input_tokens"]),
                "output_tokens": _avg(optimized_metrics["output_tokens"]),
                "latency_ms": _avg(optimized_metrics["latency_ms"]),
            },
            metrics={},
            samples=len(self.samples) * self.repeats,
            environment=_environment_report(tokenizer),
            notes=list(self.provider_notes),
        )

        base_in = benchmark.baseline["input_tokens"]
        opt_in = benchmark.optimized["input_tokens"]
        base_lat = benchmark.baseline["latency_ms"]
        opt_lat = benchmark.optimized["latency_ms"]

        total_cache = cache_hits + cache_misses
        benchmark.metrics = {
            "token_reduction_percent": _percent(base_in, opt_in),
            "latency_change_percent": _percent_change(base_lat, opt_lat),
            "cache_hit_rate": (
                f"{cache_hits / total_cache * 100:.1f}%" if total_cache else "N/A"
            ),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "optimizer_overhead_ms_avg": _avg(optimizer_times),
            "avg_tokens_saved_per_request": max(0, (base_in or 0) - (opt_in or 0)),
            "summary": collector.summary(),
        }
        if base_in is None or base_in == 0:
            benchmark.notes.append(
                "Baseline input tokens could not be measured; reduction is N/A."
            )
        if opt_in is None or opt_in == 0:
            benchmark.notes.append(
                "Optimized input tokens could not be measured; reduction is N/A."
            )
        return benchmark

    def _call(self, request: Any) -> tuple[float, float, str]:
        payload = {
            "prompt": request.prompt,
            "system": request.system,
            "messages": request.messages,
            "model": self.model,
        }
        started = time.monotonic()
        response: ProviderResponse = self.provider.generate(payload)
        elapsed_ms = (time.monotonic() - started) * 1000
        return elapsed_ms, float(response.output_tokens), response.content

    @staticmethod
    def _request_tokens(request: OptimizerRequest, tokenizer: Tokenizer) -> int:
        total = tokenizer.count(request.prompt)
        if request.system:
            total += tokenizer.count(request.system)
        if request.messages:
            for msg in request.messages:
                total += tokenizer.count(str(msg.get("content", "")))
        return total

    def _default_samples(self) -> list[OptimizerRequest]:
        return [
            OptimizerRequest(
                prompt="What is machine learning?",
                system="You are a helpful assistant.",
            ),
            OptimizerRequest(
                prompt="Explain the difference between classification and regression.",
                system="You are a helpful assistant.",
            ),
            OptimizerRequest(
                prompt="Write a haiku about a computer.",
                system="You are a helpful assistant.",
            ),
        ]


def _environment_report(tokenizer: Tokenizer) -> dict[str, Any]:
    """Reproducible environment fingerprint for a benchmark result."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 - timezone.utc kept for py3.9 dev compatibility
        "opticore_version": __version__,
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "tokenizer": tokenizer.report(),
        "cwd": os.getcwd(),
    }


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 2)


def _percent(base: float | None, current: float | None) -> float | str:
    if base is None or current is None or base == 0:
        return "N/A"
    return f"{((base - current) / base) * 100:.2f}%"


def _percent_change(base: float | None, current: float | None) -> float | str:
    if base is None or current is None or base == 0:
        return "N/A"
    return f"{((current - base) / base) * 100:.2f}%"

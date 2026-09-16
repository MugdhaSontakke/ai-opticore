"""Real benchmark runner: measures baseline vs optimized performance.

The runner only reports what it actually measured. Unavailable metrics are
reported as ``N/A``. No fabricated results are ever produced.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from opticore.benchmarks.metrics import MetricsCollector, accumulate_tokens
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
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a human-readable report, as shown by the CLI."""
        lines = [
            "AI-OptiCore Benchmark",
            "--------------------",
            "",
            f"Model: {self.model}",
            f"Provider: {self.provider}",
            f"Hardware: {self.hardware}",
            f"Samples: {self.samples}",
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
    ) -> None:
        self.provider = provider
        self.model = model
        self.samples = samples or self._default_samples()
        self.config = config or OptimizationConfig()
        self.repeats = repeats

    def run(self) -> BenchmarkResult:
        baseline_metrics: dict[str, Any] = {"input_tokens": [], "output_tokens": [], "latency_ms": []}
        optimized_metrics: dict[str, Any] = {"input_tokens": [], "output_tokens": [], "latency_ms": []}

        collector = MetricsCollector()
        tokenizer = Tokenizer()

        for request in self.samples:
            for _ in range(self.repeats):
                base_input_tokens = self._request_tokens(request, tokenizer)
                latency, out_tokens = self._call(request)
                baseline_metrics["latency_ms"].append(latency)
                baseline_metrics["output_tokens"].append(out_tokens)
                baseline_metrics["input_tokens"].append(base_input_tokens)

                pipeline = build_default_pipeline(self.config)
                pipeline_result = pipeline.run(
                    prompt=request.prompt,
                    system=request.system,
                    messages=request.messages,
                )
                optimized_request = pipeline_result.optimized_request
                accumulate_tokens(
                    collector,
                    pipeline_result.original_tokens,
                    pipeline_result.optimized_tokens,
                )

                opt_latency, opt_out = self._call(optimized_request)
                optimized_metrics["latency_ms"].append(opt_latency)
                optimized_metrics["output_tokens"].append(opt_out)
                optimized_metrics["input_tokens"].append(
                    pipeline_result.optimized_tokens
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
        )

        base_in = benchmark.baseline["input_tokens"]
        opt_in = benchmark.optimized["input_tokens"]
        base_lat = benchmark.baseline["latency_ms"]
        opt_lat = benchmark.optimized["latency_ms"]

        benchmark.metrics = {
            "token_reduction_percent": _percent(base_in, opt_in),
            "latency_change_percent": _percent_change(base_lat, opt_lat),
            "cache_hit_rate": "N/A",
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

    def _call(self, request: Any) -> tuple[float, float]:
        payload = {
            "prompt": request.prompt,
            "system": request.system,
            "messages": request.messages,
            "model": self.model,
        }
        started = time.monotonic()
        response: ProviderResponse = self.provider.generate(payload)
        elapsed_ms = (time.monotonic() - started) * 1000
        return elapsed_ms, float(response.output_tokens)

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

"""Real benchmark runner: measures baseline vs optimized performance.

The runner only reports what it actually measured. Unavailable metrics are
reported as ``N/A``. No fabricated results are ever produced.

Optimizer overhead is measured separately from model latency so that a run
which saves tokens but is slower overall is clearly visible.
"""

from __future__ import annotations

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
from opticore.costing import CostEstimator
from opticore.evaluation.evaluator import CharacterSimilarity, TokenJaccardSimilarity
from opticore.hardware import detect_backend
from opticore.logging import get_logger
from opticore.optimizers.token import Tokenizer
from opticore.providers.base import ProviderResponse

logger = get_logger("opticore.benchmarks")

#: Marker for an optimized response served from cache (no new text generated).
OPTIMIZED_RESPONSE_CACHE_HIT_MARKER = "__cache_hit__"


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
    dataset: str = "builtin"
    environment: dict[str, Any] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    sample_responses: list[dict[str, Any]] = field(
        default_factory=list, repr=False
    )

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
            f"Dataset: {self.dataset}",
            f"Samples: {self.samples}",
            f"Timestamp: {env.get('timestamp', 'N/A')}",
            f"Version: {env.get('opticore_version', 'N/A')}",
        ]
        if self.provider_info and any(
            self.provider_info.get(key) is not None
            for key in ("endpoint", "ollama_version", "installed_models_count", "requested_model")
        ):
            lines.append("")
            lines.append("Provider info:")
            for key in ("endpoint", "ollama_version", "installed_models_count", "requested_model"):
                value = self.provider_info.get(key)
                if value is not None:
                    lines.append(f"  {key}: {value}")
            if "reachable" in self.provider_info:
                lines.append(
                    "  reachable: " + ("yes" if self.provider_info.get("reachable") else "no")
                )
        lines += [
            "",
            "Baseline (no optimization)",
            f"Input tokens: {self.baseline.get('input_tokens', 'N/A')}",
            f"Output tokens: {self.baseline.get('output_tokens', 'N/A')}",
            f"Latency (avg): {self._fmt_ms(self.baseline.get('latency_ms'))}",
            f"Latency (median): {self._fmt_ms(self.baseline.get('latency_median_ms'))}",
            f"Latency (p95): {self._fmt_ms(self.baseline.get('latency_p95_ms'))}",
            "",
            "Optimized (AI-OptiCore enabled)",
            f"Input tokens: {self.optimized.get('input_tokens', 'N/A')}",
            f"Output tokens: {self.optimized.get('output_tokens', 'N/A')}",
            f"Latency (avg): {self._fmt_ms(self.optimized.get('latency_ms'))}",
            f"Latency (median): {self._fmt_ms(self.optimized.get('latency_median_ms'))}",
            f"Latency (p95): {self._fmt_ms(self.optimized.get('latency_p95_ms'))}",
            "",
            f"Token reduction: {self.metrics.get('token_reduction_percent', 'N/A')}",
            f"Latency change (avg): {self.metrics.get('latency_change_percent', 'N/A')}",
            f"Net latency change (median): {self._fmt_ms(self.metrics.get('net_latency_change_ms'))}",
            f"Optimizer overhead (avg): {self._fmt_ms(self.metrics.get('optimizer_overhead_ms_avg'))}",
            f"Provider latency (avg): {self._fmt_ms(self.metrics.get('provider_latency_ms_avg'))}",
            f"Cache hit rate: {self.metrics.get('cache_hit_rate', 'N/A')}",
            f"Quality score (avg): {self.metrics.get('quality_score_avg', 'N/A')}",
            f"Response similarity (heuristic): {self.metrics.get('quality_response_similarity_avg', 'N/A')}",
            "Estimated cost baseline: {baseline}  Estimated cost optimized: {opt}  Estimated savings: {savings}".format(
                baseline=self.metrics.get('estimated_cost_baseline', 'N/A'),
                opt=self.metrics.get('estimated_cost_optimized', 'N/A'),
                savings=self.metrics.get('estimated_cost_savings', 'N/A'),
            ),
        ]
        tsp = (
            (self.metrics.get("tokens_per_sec_baseline"), self.metrics.get("tokens_per_sec_optimized"))
            if self.metrics.get("tokens_per_sec_baseline") is not None
            or self.metrics.get("tokens_per_sec_optimized") is not None
            else None
        )
        if tsp:
            lines += [
                "",
                f"Throughput (tokens/sec): baseline={tsp[0]:.0f}  optimized={tsp[1]:.0f}",
            ]
        if (
            "quality_evaluator_type" in self.metrics
            and self.metrics["quality_evaluator_type"]
        ):
            lines += [
                "",
                f"Quality evaluator: {self.metrics['quality_evaluator_type']}",
            ]
        scenarios = self.metrics.get("scenarios")
        if scenarios:
            lines += ["", "Per-scenario (token reduction, avg latency, samples):"]
            for name, summary in scenarios.items():
                lines += [
                    f"  {name}: "
                    f"reduction={summary.get('token_reduction_percent', 'N/A')}, "
                    f"baseline_avg={self._fmt_ms(summary.get('baseline_latency_ms_avg'))}, "
                    f"optimized_avg={self._fmt_ms(summary.get('optimized_latency_ms_avg'))}, "
                    f"samples={summary.get('samples')}"
                ]
        if self.notes:
            lines += ["", "Notes:"]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (no raw prompts or responses)."""
        return {
            "environment": self.environment,
            "provider_info": self.provider_info,
            "model": self.model,
            "provider": self.provider,
            "hardware": self.hardware,
            "samples": self.samples,
            "dataset": self.dataset,
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
        dataset: str = "builtin",
    ) -> None:
        self.provider = provider
        self.model = model
        self.samples = samples or self._default_samples()
        self.config = config or OptimizationConfig()
        self.repeats = repeats
        self.provider_notes = provider_notes or []
        self.dataset = dataset

    def run(self) -> BenchmarkResult:
        baseline_metrics: dict[str, Any] = {"input_tokens": [], "output_tokens": [], "latency_ms": []}
        optimized_metrics: dict[str, Any] = {"input_tokens": [], "output_tokens": [], "latency_ms": []}

        collector = MetricsCollector()
        optimizer_times: list[float] = []
        provider_latencies: list[float] = []
        quality_scores: list[float] = []
        quality_checked = 0
        quality_verified = 0
        tokenizer = Tokenizer()
        self._cache: MemoryCache | None = (
            MemoryCache() if self.config.enable_semantic_cache else None
        )
        cache_hits = 0
        cache_misses = 0
        scenario_runs: dict[str, list[Any]] = {}
        baseline_responses: list[str] = []
        optimized_responses: list[str] = []
        baseline_tokens_per_sec: list[float] = []
        optimized_tokens_per_sec: list[float] = []
        sample_responses: list[dict[str, Any]] = []
        if getattr(self.provider, "name", "").startswith("fake"):
            self.provider_notes.append("deterministic fake provider; latency is not representative")

        for request in self.samples:
            metadata = getattr(request, "metadata", None) or {}
            scenario = metadata.get("scenario") or "default"
            sample_id = metadata.get("id") or scenario
            for _ in range(self.repeats):
                base_input_tokens = self._request_tokens(request, tokenizer)
                latency, out_tokens, base_content = self._call(request)
                baseline_responses.append(base_content or "")
                _record_tokens_per_sec(
                    baseline_tokens_per_sec, out_tokens, latency
                )
                baseline_metrics["latency_ms"].append(latency)
                baseline_metrics["output_tokens"].append(out_tokens)
                baseline_metrics["input_tokens"].append(base_input_tokens)

                pipeline = build_default_pipeline(self.config)
                opt_started = time.monotonic()
                pipeline_result = pipeline.run(
                    prompt=request.prompt,
                    system=request.system,
                    messages=request.messages,
                    tools=getattr(request, "tools", None),
                )
                optimizer_overhead_ms = (time.monotonic() - opt_started) * 1000.0
                optimizer_times.append(optimizer_overhead_ms)
                optimized_request = pipeline_result.optimized_request
                accumulate_tokens(
                    collector,
                    pipeline_result.original_tokens,
                    pipeline_result.optimized_tokens,
                )
                quality = pipeline_result.metadata.get("quality")
                if isinstance(quality, dict):
                    score = quality.get("similarity")
                    if isinstance(score, (int, float)):
                        quality_scores.append(float(score))
                    quality_checked += 1
                    if quality.get("verified") is True:
                        quality_verified += 1

                opt_latency = 0.0
                opt_out = 0.0
                run_provider_latency = 0.0
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
                        opt_latency += optimizer_overhead_ms
                        opt_out = float(entry.metadata.get("output_tokens", 0))
                        optimized_responses.append(
                            OPTIMIZED_RESPONSE_CACHE_HIT_MARKER
                        )
                    else:
                        cache_misses += 1
                        run_provider_latency, opt_out, content = self._call(
                            optimized_request
                        )
                        optimized_responses.append(content or "")
                        _record_tokens_per_sec(
                            optimized_tokens_per_sec, opt_out, run_provider_latency
                        )
                        opt_latency = (
                            time.monotonic() - lookup_started
                        ) * 1000.0 + optimizer_overhead_ms
                        provider_latencies.append(run_provider_latency)
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
                    run_provider_latency, opt_out, opt_content = self._call(optimized_request)
                    optimized_responses.append(opt_content or "")
                    _record_tokens_per_sec(
                        optimized_tokens_per_sec, opt_out, run_provider_latency
                    )
                    opt_latency = run_provider_latency + optimizer_overhead_ms
                    provider_latencies.append(run_provider_latency)
                optimized_metrics["latency_ms"].append(opt_latency)
                optimized_metrics["output_tokens"].append(opt_out)
                optimized_metrics["input_tokens"].append(
                    pipeline_result.optimized_tokens
                )
                scenario_runs.setdefault(scenario, []).append(
                    {
                        "base_tokens": base_input_tokens,
                        "opt_tokens": pipeline_result.optimized_tokens,
                        "base_latency_ms": latency,
                        "opt_latency_ms": opt_latency,
                    }
                )
                if request.prompt:
                    sample_responses.append(
                        {
                            "sample_id": str(sample_id),
                            "scenario": str(scenario),
                            "baseline": baseline_responses[-1]
                            if baseline_responses
                            else "",
                            "optimized": (
                                optimized_responses[-1]
                                if optimized_responses
                                and optimized_responses[-1]
                                != OPTIMIZED_RESPONSE_CACHE_HIT_MARKER
                                else ""
                            ),
                            "cache_hit": entry is not None
                            if self._cache is not None
                            else False,
                        }
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
                "latency_median_ms": _median(baseline_metrics["latency_ms"]),
                "latency_p95_ms": _percentile(baseline_metrics["latency_ms"], 95),
            },
            optimized={
                "input_tokens": _avg(optimized_metrics["input_tokens"]),
                "output_tokens": _avg(optimized_metrics["output_tokens"]),
                "latency_ms": _avg(optimized_metrics["latency_ms"]),
                "latency_median_ms": _median(optimized_metrics["latency_ms"]),
                "latency_p95_ms": _percentile(optimized_metrics["latency_ms"], 95),
            },
            metrics={},
            samples=len(self.samples) * self.repeats,
            dataset=self.dataset,
            environment=_environment_report(tokenizer),
            provider_info=_provider_info(self.provider, model=self.model),
            notes=list(self.provider_notes),
            sample_responses=sample_responses,
        )

        base_in = benchmark.baseline["input_tokens"]
        opt_in = benchmark.optimized["input_tokens"]
        base_lat = benchmark.baseline["latency_ms"]
        opt_lat = benchmark.optimized["latency_ms"]
        base_lat_median = benchmark.baseline["latency_median_ms"]
        opt_lat_median = benchmark.optimized["latency_median_ms"]

        total_cache = cache_hits + cache_misses
        cost = CostEstimator(
            input_price_per_1k=self.config.pricing_input_per_1k,
            output_price_per_1k=self.config.pricing_output_per_1k,
        )
        base_out = benchmark.baseline["output_tokens"]
        opt_out_tokens = benchmark.optimized["output_tokens"]
        base_cost = cost.estimate(base_in, base_out)
        opt_cost = cost.estimate(opt_in, opt_out_tokens)
        savings = cost.savings(base_cost, opt_cost)

        benchmark.metrics = {
            "token_reduction_percent": _percent(base_in, opt_in),
            "latency_change_percent": _percent_change(base_lat, opt_lat),
            "net_latency_change_ms": (
                round(opt_lat_median - base_lat_median, 2)
                if base_lat_median is not None and opt_lat_median is not None
                else None
            ),
            "base_total_latency_ms": base_lat,
            "optimized_total_latency_ms": opt_lat,
            "optimizer_latency_ms_avg": _avg(optimizer_times),
            "optimizer_latency_ms_median": _median(optimizer_times),
            "provider_latency_ms_avg": _avg(provider_latencies),
            "optimization_overhead_tokens": 0,
            "cache_hit_rate": (
                f"{cache_hits / total_cache * 100:.1f}%" if total_cache else "N/A"
            ),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "optimizer_overhead_ms_avg": _avg(optimizer_times),
            "tokens_per_sec_baseline": _avg(baseline_tokens_per_sec),
            "tokens_per_sec_optimized": _avg(optimized_tokens_per_sec),
            "quality_response_similarity_avg": _pair_similarity(
                baseline_responses, optimized_responses, CharacterSimilarity()
            ),
            "quality_response_token_jaccard_avg": _pair_similarity(
                baseline_responses, optimized_responses, TokenJaccardSimilarity()
            ),
            "quality_evaluator_type": (
                "heuristic: character_similarity + token_jaccard "
                "(deterministic, no model-based judge configured)"
            ),
            "avg_tokens_saved_per_request": max(0, (base_in or 0) - (opt_in or 0)),
            "quality_score_avg": (
                round(statistics.mean(quality_scores), 4) if quality_scores else "N/A"
            ),
            "quality_verified": (
                f"{quality_verified}/{quality_checked}"
                if quality_checked
                else "N/A"
            ),
            "estimated_cost_baseline": _fmt_cost(base_cost.total_cost),
            "estimated_cost_optimized": _fmt_cost(opt_cost.total_cost),
            "estimated_cost_savings": _fmt_cost(savings.total_cost),
            "estimated_cost_labeled": "estimated (user-configured pricing)" if cost.configured else "not configured",
            "scenarios": _scenario_summaries(scenario_runs),
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
        if base_lat_median is not None and opt_lat_median is not None:
            net = opt_lat_median - base_lat_median
            if net > 1:
                benchmark.notes.append(
                    f"Net median latency increased by {net:.1f} ms with "
                    "optimization enabled - reported honestly, not hidden."
                )
        if cost.configured:
            benchmark.notes.append(
                "Cost figures are ESTIMATED from user-configured pricing, not "
                "provider-quoted bills."
            )
        else:
            benchmark.notes.append(
                "No pricing configured; cost figures are N/A. Provide pricing "
                "in config (pricing.input_per_1k / output_per_1k) to estimate cost."
            )
        benchmark.notes.append(
            "optimization_overhead_tokens is 0 by design: optimizers are "
            "deterministic/local heuristics with no model-in-the-loop token spend."
        )
        return benchmark

    def _call(self, request: Any) -> tuple[float, float, str]:
        payload = {
            "prompt": request.prompt,
            "system": request.system,
            "messages": request.messages,
            "tools": getattr(request, "tools", None),
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
        """Built-in scenario samples (PHASE 12 scenarios A-H).

        Each sample is tagged with a ``metadata.scenario`` so per-scenario
        results can be reported without fabricating cross-case numbers.
        """
        def tagged(scenario: str, prompt: str, system: str = "You are a helpful assistant.") -> OptimizerRequest:
            return OptimizerRequest(
                prompt=prompt,
                system=system,
                metadata={"scenario": scenario},
            )

        return [
            # A - simple prompt
            tagged("a_simple", "What is machine learning?"),
            # B - redundant prompt (verbatim repetition)
            tagged(
                "b_redundant",
                "Explain gradient descent. Explain gradient descent. "
                "Explain gradient descent in detail please.",
            ),
            # C - long context
            tagged(
                "c_long_context",
                "Summarize the following meeting notes for the team: " * 40
                + "Action items: ship the package, update docs, run benchmarks.",
            ),
            # D - repeated request (exact duplicate)
            tagged("d_repeated", "What is the capital of France?"),
            # E - semantic duplicate request
            tagged("e_semantic_duplicate", "What is the capital of France?"),
            # F - routing complexity variant (rule-relevant shape)
            tagged(
                "f_routing_complex",
                "Compare the trade-offs between vector databases and "
                "relational databases for storing embeddings, considering "
                "latency, cost, and maintenance.",
            ),
            # G - optimization disabled (runner always optimizes; this case is
            #     used by the CLI pair of --no-optimization runs)
            tagged("g_optimization_disabled", "Write a haiku about a computer."),
            # H - optimization enabled (mirror of G)
            tagged("h_optimization_enabled", "Write a haiku about a computer."),
        ]


def load_benchmark_dataset(path: str) -> list[OptimizerRequest]:
    """Load benchmark samples (with optional scenarios) from a JSON file.

    Each entry accepts ``prompt``, optional ``system``/``messages``/``tools``,
    and a ``scenario`` tag. The dataset schema also maps ``category`` to
    ``scenario`` and keeps ``id``/``expected_keywords`` in the sample metadata
    for traceability (the keyword lists can be reused by evaluators). Raises
    when the file is missing or malformed.
    """
    import json

    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Benchmark dataset must be a JSON list of samples")
    samples: list[OptimizerRequest] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or not isinstance(entry.get("prompt"), str):
            raise ValueError(f"Dataset entry {i} must be a dict with a 'prompt' string")
        metadata: dict[str, Any] = {}
        scenario = entry.get("scenario") or entry.get("category")
        if isinstance(scenario, str):
            metadata["scenario"] = scenario
        sample_id = entry.get("id")
        if isinstance(sample_id, str):
            metadata["id"] = sample_id
        keywords = entry.get("expected_keywords")
        if isinstance(keywords, list):
            metadata["expected_keywords"] = list(keywords)
        samples.append(
            OptimizerRequest(
                prompt=entry["prompt"],
                system=entry.get("system"),
                messages=entry.get("messages"),
                tools=entry.get("tools"),
                metadata=metadata,
            )
        )
    return samples


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


def _record_tokens_per_sec(sink: list[float], tokens: float, latency_ms: float) -> None:
    """Record generation throughput only for real, measurable latency."""
    if not tokens or not latency_ms or latency_ms < 1.0:
        return
    sink.append(tokens / (latency_ms / 1000.0))


def _pair_similarity(
    baseline: list[str], optimized: list[str], evaluator: Any
) -> float | str:
    """Mean heuristic similarity between paired baseline/optimized responses.

    Cache-hit optimized entries have no new text, so they are excluded from
    the average (they were compared already when first generated).
    """
    scores: list[float] = []
    for i, opt_text in enumerate(optimized):
        if opt_text == OPTIMIZED_RESPONSE_CACHE_HIT_MARKER or i >= len(baseline):
            continue
        base_text = baseline[i]
        if not base_text.strip() or not opt_text.strip():
            continue
        scores.append(evaluator.similarity(base_text, opt_text))
    if not scores:
        return "N/A"
    return round(statistics.mean(scores), 4)


def _provider_info(provider: Any, model: str | None = None) -> dict[str, Any]:
    """Rendering-safe provider details for the report.

    Only Ollama performs a live probe here; key-based providers report only
    what is safe without network calls or secrets.
    """
    name = getattr(provider, "name", "unknown")
    if name != "ollama":
        return {"provider": name}
    try:
        health = provider.health(model=model)
    except Exception:  # noqa: BLE001 - health must never break a benchmark
        return {"provider": name, "note": "health probe unavailable"}
    return {
        "provider": name,
        "endpoint": health.get("endpoint"),
        "reachable": bool(health.get("reachable")),
        "ollama_version": health.get("ollama_version"),
        "requested_model": health.get("requested_model"),
        "installed_models_count": len(health.get("installed_models") or []),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 2)


def _percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    upper = lower + 1
    if upper >= len(ordered):
        return round(float(ordered[-1]), 2)
    fraction = rank - lower
    return round(ordered[lower] + fraction * (ordered[upper] - ordered[lower]), 2)


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.6f} (estimated)"


def _scenario_summaries(scenario_runs: dict[str, list[Any]]) -> dict[str, Any]:
    """Aggregate per-scenario token reduction and latency (avg across runs)."""
    summaries: dict[str, Any] = {}
    for name, runs in scenario_runs.items():
        num = len(runs)
        if not num:
            continue
        base_tokens = [r["base_tokens"] for r in runs]
        opt_tokens = [r["opt_tokens"] for r in runs]
        base_lat = [r["base_latency_ms"] for r in runs]
        opt_lat = [r["opt_latency_ms"] for r in runs]
        base_sum = sum(base_tokens)
        opt_sum = sum(opt_tokens)
        reductions: list[float] = []
        for i in range(len(base_tokens)):
            b, o = base_tokens[i], opt_tokens[i]
            if b:
                reductions.append((b - o) / b * 100.0)
        summaries[name] = {
            "samples": num,
            "token_reduction_percent": (
                _percent(base_sum / num, opt_sum / num)
                if num
                else "N/A"
            ),
            "avg_reduction_percent": (
                round(sum(reductions) / len(reductions), 2) if reductions else "N/A"
            ),
            "baseline_latency_ms_avg": _avg(base_lat),
            "optimized_latency_ms_avg": _avg(opt_lat),
        }
    return summaries


def _percent(base: float | None, current: float | None) -> float | str:
    if base is None or current is None or base == 0:
        return "N/A"
    return f"{((base - current) / base) * 100:.2f}%"


def _percent_change(base: float | None, current: float | None) -> float | str:
    if base is None or current is None or base == 0:
        return "N/A"
    return f"{((current - base) / base) * 100:.2f}%"

"""Unified metrics collection.

The collector is extensible: new metrics are added by calling ``record`` with
a key, or by registering a custom aggregator for a named series.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Aggregator = Callable[[list[float]], float]


def percentile(values: list[float], p: float) -> float:
    """Deterministic linear-interpolation percentile (p in [0, 1]).

    Used instead of ``statistics.quantiles`` so results are stable and easy
    to test (nearest-rank interpolation when ``(n-1)*p`` is integral).
    """
    if not values:
        raise ValueError("percentile of an empty series is undefined")
    data = sorted(values)
    k = (len(data) - 1) * p
    low = int(k)
    high = low + 1
    if high >= len(data):
        return data[-1]
    frac = k - low
    return data[low] + (data[high] - data[low]) * frac


@dataclass
class MetricsCollector:
    """Tracks both counters and value series.

    - Counters accumulate integers (requests, hits, errors).
    - Series accumulate floats and expose min/max/sum/mean plus p50/p95.

    Series are bounded: only the most recent ``max_series_len`` samples are
    kept so memory usage is flat even under sustained load.
    """

    counters: dict[str, int] = field(default_factory=dict)
    series: dict[str, list[float]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    _custom_aggregators: dict[str, Aggregator] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    max_series_len: int = 1000

    def __post_init__(self) -> None:
        if self.max_series_len < 1:
            raise ValueError("max_series_len must be >= 1")

    def incr(self, key: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + amount

    def record(self, key: str, value: float) -> None:
        """Append a value to a named series (kept bounded by max_series_len)."""
        with self._lock:
            bucket = self.series.setdefault(key, [])
            bucket.append(value)
            if len(bucket) > self.max_series_len:
                del bucket[0 : len(bucket) - self.max_series_len]

    def register_custom_aggregator(self, key: str, fn: Aggregator) -> None:
        """Register a custom aggregator for a named series."""
        self._custom_aggregators[key] = fn

    def summary(self) -> dict[str, Any]:
        """Return a flat summary suitable for logging and reports."""
        with self._lock:
            out: dict[str, Any] = {"counters": dict(self.counters)}
            series_summary: dict[str, Any] = {}
            for key, values in self.series.items():
                if not values:
                    series_summary[key] = {}
                    continue
                summary: dict[str, Any] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "sum": sum(values),
                    "mean": sum(values) / len(values),
                }
                if len(values) >= 4:
                    summary["p50"] = round(percentile(values, 0.5), 6)
                    summary["p95"] = round(percentile(values, 0.95), 6)
                agg = self._custom_aggregators.get(key)
                if agg:
                    summary["custom"] = agg(values)
                series_summary[key] = summary
            out["series"] = series_summary
            out["uptime_seconds"] = time.monotonic() - self.started_at
            out["tokens_saved"] = (
                self.counters.get("original_tokens", 0)
                - self.counters.get("optimized_tokens", 0)
            )
            return out

    def merge(self, other: MetricsCollector) -> None:
        """Fold another collector's counters/series into this one."""
        with self._lock:
            for k, v in other.counters.items():
                self.counters[k] = self.counters.get(k, 0) + v
            for k, values in other.series.items():
                target = self.series.get(k)
                if target is None:
                    target = self.series[k] = []
                target.extend(values)
                if len(target) > self.max_series_len:
                    del target[0 : len(target) - self.max_series_len]


def accumulate_tokens(collector: MetricsCollector, original: int, optimized: int) -> None:
    """Helper to record token metrics for a single optimization run."""
    collector.incr("request_count", 1)
    collector.incr("original_tokens", original)
    collector.incr("optimized_tokens", optimized)
    collector.incr("tokens_saved", max(0, original - optimized))
    if original:
        collector.record("reduction_percent", (original - optimized) / original * 100.0)


def token_metrics(
    original_input: int,
    optimized_input: int,
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """Canonical token metric keys used across pipeline, API and benchmarks.

    ``output_tokens`` is optional: pipelines cannot know it before a provider
    responds, so downstream code passes the real value when available.
    """
    saved = max(0, original_input - optimized_input)
    total = optimized_input + output_tokens if output_tokens is not None else None
    metrics: dict[str, Any] = {
        "original_input_tokens": original_input,
        "optimized_input_tokens": optimized_input,
        "tokens_saved": saved,
        "reduction_percentage": round(
            (saved / original_input * 100.0) if original_input else 0.0, 4
        ),
    }
    if output_tokens is not None:
        metrics["output_tokens"] = output_tokens
        metrics["total_tokens"] = total
    return metrics

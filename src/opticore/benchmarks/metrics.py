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


@dataclass
class MetricsCollector:
    """Tracks both counters and value series.

    - Counters accumulate integers (requests, hits, errors).
    - Series accumulate floats and expose min/max/sum/mean.
    """

    counters: dict[str, int] = field(default_factory=dict)
    series: dict[str, list[float]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    _custom_aggregators: dict[str, Aggregator] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def incr(self, key: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + amount

    def record(self, key: str, value: float) -> None:
        """Append a value to a named series."""
        with self._lock:
            self.series.setdefault(key, []).append(value)

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
                self.series.setdefault(k, []).extend(values)


def accumulate_tokens(collector: MetricsCollector, original: int, optimized: int) -> None:
    """Helper to record token metrics for a single optimization run."""
    collector.incr("request_count", 1)
    collector.incr("original_tokens", original)
    collector.incr("optimized_tokens", optimized)
    collector.incr("tokens_saved", max(0, original - optimized))
    if original:
        collector.record("reduction_percent", (original - optimized) / original * 100.0)

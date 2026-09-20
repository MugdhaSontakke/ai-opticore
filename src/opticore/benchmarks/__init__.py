"""Benchmarking subpackage: real measurement, no fabrication."""

from __future__ import annotations

from opticore.benchmarks.metrics import MetricsCollector
from opticore.benchmarks.runner import BenchmarkResult, BenchmarkRunner, load_benchmark_dataset

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "MetricsCollector",
    "load_benchmark_dataset",
]

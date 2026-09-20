"""Integration tests: end-to-end flow using fake providers (no paid APIs)."""

from __future__ import annotations

import json

from opticore import AIClient, OptimizationConfig, Optimizer
from opticore.benchmarks.runner import BenchmarkRunner
from opticore.providers.base import FakeProviderMixin, ProviderResponse


class CountingFakeProvider(FakeProviderMixin):
    name = "counting_fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, request: dict) -> ProviderResponse:
        self.calls += 1
        return super().generate(request)


def test_optimizer_end_to_end() -> None:
    optimizer = Optimizer()
    result = optimizer.optimize("hello   world", system="be helpful")
    assert result.optimized_prompt == "hello world"
    assert result.original_tokens > 0


def test_optimizer_warnings_for_aggressive() -> None:
    config = OptimizationConfig(enable_context_optimization=False)
    optimizer = Optimizer(config=config)
    prompt = "Explain X. " * 50
    outcome = optimizer.optimize(prompt)
    assert isinstance(outcome.warnings, list)


def test_aiclient_returns_generation_result() -> None:
    provider = CountingFakeProvider()
    client = AIClient(provider=provider, config=OptimizationConfig())
    result = client.generate(prompt="hello there")
    assert result.content
    assert result.provider == "counting_fake"
    assert provider.calls == 1
    assert result.total_time_ms >= 0


def test_aiclient_cache_prevents_second_call() -> None:
    provider = CountingFakeProvider()
    client = AIClient(provider=provider)
    first = client.generate(prompt="same prompt")
    second = client.generate(prompt="same prompt")
    assert second.cache_hit is True
    assert provider.calls == 1
    assert first.content == second.content


def test_aiclient_optimization_disabled_treats_as_baseline() -> None:
    provider = CountingFakeProvider()
    client = AIClient(provider=provider, optimization=False)
    result = client.generate(prompt="hello")
    assert result.optimized_tokens == 0
    assert result.tokens_saved == 0


def test_aiclient_semantic_cache_when_embeddings_enabled() -> None:
    provider = CountingFakeProvider()

    def embedding(text: str) -> list[float]:
        # A cheap, deterministic embedding where similar words overlap.
        base = [0.0, 0.0, 0.0, 0.0]
        for i, ch in enumerate(text.lower()):
            base[i % 4] += ord(ch)
        return base

    client = AIClient(
        provider=provider,
        config=OptimizationConfig(enable_semantic_cache=True),
        embedding_fn=embedding,
    )
    client.generate(prompt="capital of France")
    client.generate(prompt="capital of France")
    assert client.metrics.counters["cache_hits"] >= 1


def test_benchmark_runner_with_fake() -> None:
    from opticore.core.interfaces import OptimizerRequest

    provider = CountingFakeProvider()
    runner = BenchmarkRunner(
        provider=provider,
        model="fake-model",
        samples=[OptimizerRequest(prompt="hello world")],
        repeats=1,
    )
    result = runner.run()
    render = result.render()
    assert "AI-OptiCore Benchmark" in render
    # The reduction must be a real measured percentage (never the vacuous pass
    # "N/A or True" style assertion that used to exist here).
    reduction = result.metrics["token_reduction_percent"]
    assert isinstance(reduction, str) and reduction.endswith("%")
    assert result.baseline["latency_ms"] is not None
    assert "Model: fake-model" in render


def test_benchmark_uses_only_real_measurements() -> None:
    provider = CountingFakeProvider()
    runner = BenchmarkRunner(provider=provider, model="m", repeats=1)
    result = runner.run()
    baseline = result.baseline
    assert baseline["input_tokens"] is not None or baseline["latency_ms"] is not None


def test_benchmark_result_to_dict_contains_metadata() -> None:
    provider = CountingFakeProvider()
    runner = BenchmarkRunner(provider=provider, model="m", repeats=1)
    result = runner.run()
    data = result.to_dict()
    env = data["environment"]
    assert "timestamp" in env
    assert "opticore_version" in env
    assert "python_version" in env
    assert "os" in env
    assert data["provider"] == "counting_fake"
    assert "optimizer_overhead_ms_avg" in data["metrics"]
    serialized = json.dumps(data, default=str)
    assert isinstance(serialized, str)


def test_generation_result_to_dict_has_total_tokens_and_overhead() -> None:

    provider = CountingFakeProvider()
    client = AIClient(provider=provider, optimization=True)
    result = client.generate(prompt="hello")
    d = result.to_dict()
    assert "total_tokens" in d
    assert "optimizer_time_ms" in d
    assert "model_time_ms" in d
    assert d["total_tokens"] == d["input_tokens"] + d["output_tokens"]
    assert d["reduction_percentage"] >= 0.0

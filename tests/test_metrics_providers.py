"""Tests for metrics, evaluation, and providers (with fakes)."""

from __future__ import annotations

import pytest

from opticore.benchmarks.metrics import MetricsCollector, accumulate_tokens
from opticore.evaluation.evaluator import (
    CharacterSimilarity,
    DefaultResponseEvaluator,
    QualityGate,
    TokenJaccardSimilarity,
)
from opticore.providers import available_providers, get_provider
from opticore.providers.base import FakeProviderMixin, ProviderResponse


def test_metrics_collector_counters() -> None:
    metrics = MetricsCollector()
    metrics.incr("requests", 3)
    metrics.incr("requests", 2)
    assert metrics.counters["requests"] == 5


def test_metrics_collector_series() -> None:
    metrics = MetricsCollector()
    metrics.record("latency", 1.0)
    metrics.record("latency", 3.0)
    summary = metrics.summary()
    assert summary["series"]["latency"]["mean"] == 2.0
    assert summary["series"]["latency"]["max"] == 3.0


def test_metrics_merge() -> None:
    a = MetricsCollector()
    b = MetricsCollector()
    a.incr("x", 1)
    b.incr("x", 2)
    a.incr("y", 5)
    b.incr("y", 2)
    a.merge(b)
    assert a.counters["x"] == 3
    assert a.counters["y"] == 7


def test_accumulate_tokens() -> None:
    metrics = MetricsCollector()
    accumulate_tokens(metrics, original=100, optimized=60)
    assert metrics.counters["request_count"] == 1
    assert metrics.counters["tokens_saved"] == 40
    assert metrics.summary()["tokens_saved"] == 40


def test_character_similarity_identical() -> None:
    assert CharacterSimilarity().similarity("hello", "hello") == 1.0


def test_character_similarity_empty() -> None:
    assert CharacterSimilarity().similarity("", "") == 1.0


def test_token_jaccard() -> None:
    jac = TokenJaccardSimilarity()
    assert jac.similarity("cat dog", "cat dog") == 1.0
    assert jac.similarity("cat dog", "cat bird") > 0.0


def test_default_evaluator() -> None:
    evaluator = DefaultResponseEvaluator()
    result = evaluator.evaluate(
        original_response="the answer is 42",
        optimized_response="the answer is 42",
        original_request="what is the answer",
        optimized_request="the answer",
    )
    assert "response_similarity" in result
    assert "request_similarity" in result


def test_quality_gate_passes_on_identical() -> None:
    gate = QualityGate()
    verdict = gate.evaluate(1.0, "safe")
    assert verdict["passed"] is True


def test_quality_gate_warns_on_low_similarity() -> None:
    gate = QualityGate()
    verdict = gate.evaluate(0.1, "safe")
    assert verdict["passed"] is False
    assert verdict["warnings"]


def test_fake_provider_deterministic() -> None:
    class Fake(FakeProviderMixin):
        name = "fake"

    provider = Fake()
    resp = provider.generate({"prompt": "hello"})
    assert isinstance(resp, ProviderResponse)
    assert resp.provider.startswith("fake:")
    assert resp.input_tokens > 0


def test_providers_registration() -> None:
    names = available_providers()
    assert "openai" in names
    assert "ollama" in names
    assert "huggingface" in names


def test_get_provider_unknown_raises() -> None:
    from opticore.providers.base import ProviderError

    with pytest.raises(ProviderError):
        get_provider("not-a-real-provider")

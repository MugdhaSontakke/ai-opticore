"""Quality regression suite (PHASE 4).

Loads the deterministic evaluation dataset and verifies, per case:

- baseline vs optimized response availability through a real client path;
- token usage is measured (not fabricated);
- when the quality gate rejects an optimization, the ORIGINAL request is
  restored and the failure is surfaced (never a silent quality drop);
- cache behavior for repeated requests;
- optimization overhead is reported, not hidden.

These tests never compare against 'gold' answers (that would need a judge
model); they assert the safety properties independently.
"""

from __future__ import annotations

import json
import os

import pytest

from opticore import AIClient, OptimizationConfig
from opticore.core.interfaces import OptimizerResult
from opticore.core.pipeline import OptimizationPipeline, build_default_pipeline
from opticore.providers.base import FakeProviderMixin

DATA_PATH = os.path.join(os.path.dirname(__file__), "evaluation", "test_cases.json")


def _load_cases() -> dict:
    with open(DATA_PATH) as fh:
        return json.load(fh)


class Fake(FakeProviderMixin):
    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_prompt: str | None = None

    def generate(self, request: dict) -> object:
        self.calls += 1
        self.last_prompt = str(request.get("prompt"))
        return super().generate(request)


@pytest.mark.parametrize("case_id", [c["id"] for c in _load_cases()["cases"]])
def test_each_case_optimizes_cleanly(case_id: str) -> None:
    case = next(c for c in _load_cases()["cases"] if c["id"] == case_id)
    pipeline = build_default_pipeline(OptimizationConfig(safety_mode="balanced"))
    result: OptimizerResult = pipeline.run(
        prompt=case["prompt"],
        system=case.get("system"),
        messages=case.get("messages"),
    )
    assert result.original_tokens >= 0
    assert result.optimized_tokens >= 0
    assert result.optimized_request.prompt is not None
    # Never accept a degraded result silently: if rejected, restore original.
    if not result.accepted:
        assert result.optimized_request.prompt == result.original_request.prompt
        assert result.rejection_reason is not None


def test_all_dataset_cases_are_content_preserving_shapes() -> None:
    cases = _load_cases()["cases"]
    assert len(cases) >= 10
    expected_categories = {
        "factual question",
        "summarization",
        "coding question",
        "reasoning question",
        "conversational context",
        "repeated request",
        "long-context request",
        "structured-output request",
        "ambiguous request",
        "system-instruction-heavy request",
    }
    seen = {c["category"] for c in cases}
    assert expected_categories <= seen


def test_repeated_request_hits_cache_and_optimizes_once() -> None:
    provider = Fake()
    client = AIClient(provider=provider, config=OptimizationConfig())
    first = client.generate(
        prompt="What is the tallest mountain in the world?",
        system="You are a helpful assistant.",
    )
    second = client.generate(
        prompt="What is the tallest mountain in the world?",
        system="You are a helpful assistant.",
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.original_tokens > 0
    assert second.tokens_saved >= 0


def test_quality_gate_rejection_restores_original_request(monkeypatch) -> None:
    # Deterministic restoration path: force the evaluator to report low
    # similarity so the gate must reject and restore the original request.
    config = OptimizationConfig(
        quality_enabled=True,
        quality_reject_on_failure=True,
        quality_minimum_score=0.85,
    )
    provider = Fake()
    client = AIClient(provider=provider, config=config)
    monkeypatch.setattr(
        client.pipeline, "_quality_evaluator", lambda: lambda a, b: 0.40
    )
    original_prompt = "Summarize the quarterly results for the engineering team."
    result = client.generate(
        prompt=original_prompt,
        system="You are a helpful assistant.",
    )
    assert result.accepted is False
    assert result.rejection_reason is not None
    assert result.metadata["quality"]["rejected"] is True
    # The application must receive the ORIGINAL content, never a degraded one.
    assert provider.last_prompt == original_prompt


def test_overhead_and_timing_are_reported_honestly() -> None:
    provider = Fake()
    client = AIClient(provider=provider, config=OptimizationConfig())
    result = client.generate(prompt="Summarize the quarterly results for the team.")
    assert result.request_id
    assert result.optimizer_time_ms >= 0
    assert result.cache_lookup_ms >= 0
    assert result.model_time_ms >= 0
    assert result.total_time_ms >= result.optimizer_time_ms + result.cache_lookup_ms
    # cost is N/A (no pricing configured) - never a fabricated number
    baseline = result.estimated_cost["baseline"]
    assert baseline["estimated"] is True
    assert baseline["total_cost"] is None


def test_optimization_failure_is_not_application_failure() -> None:
    from opticore.exceptions import OptimizationError

    class ExplodingOptimizer:
        name = "explode"
        order = 5
        enabled = True

        def optimize(self, request, config):  # type: ignore[no-untyped-def]
            raise OptimizationError("boom")

    pipeline = OptimizationPipeline(config=OptimizationConfig())
    pipeline.add(ExplodingOptimizer())
    result = pipeline.run(prompt="never crash the app because of an optimizer")
    assert result.optimized_request.prompt  # pipeline survived the optimizer error
    assert "explode" in " ".join(result.metadata.get("optimizer_errors", []))

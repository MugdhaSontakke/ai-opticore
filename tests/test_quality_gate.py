"""Tests for the quality gate and safe-fallback contract."""

from __future__ import annotations

from opticore import OptimizationConfig
from opticore.core.pipeline import build_default_pipeline


class DummyFailEvaluator:
    """Force the gate to fail regardless of actual similarity."""

    def similarity(self, a: str, b: str) -> float:
        return 0.0


def test_quality_gate_notes_verification() -> None:
    pipeline = build_default_pipeline(OptimizationConfig())
    result = pipeline.run(prompt="hello world")
    quality = result.metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("verified") is True


def test_quality_disabled_notes_not_verified() -> None:
    config = OptimizationConfig(quality_enabled=False)
    pipeline = build_default_pipeline(config)
    result = pipeline.run(prompt="hello world")
    assert result.metadata["quality"]["verified"] is False


def test_quality_reject_falls_back_to_original() -> None:
    config = OptimizationConfig(
        quality_enabled=True,
        quality_minimum_score=0.95,
        quality_reject_on_failure=True,
        quality_evaluator="character",
    )
    pipeline = build_default_pipeline(config)
    request = _build_dummy_request()
    # Monkey patch the evaluator so similarity is effectively 0 -> rejection.
    pipeline._quality_evaluator = lambda: DummyFailEvaluator().similarity
    result = pipeline.run(request=request)
    assert result.metadata["quality"]["rejected"] is True
    # Safe fallback: optimized request IS the original.
    assert result.optimized_request.prompt == request.prompt
    assert any("reverted" in change for change in result.changes)


def test_quality_no_reject_keeps_optimized() -> None:
    config = OptimizationConfig(
        quality_enabled=True,
        quality_minimum_score=0.95,
        quality_reject_on_failure=False,
    )
    pipeline = build_default_pipeline(config)
    pipeline._quality_evaluator = lambda: DummyFailEvaluator().similarity
    result = pipeline.run(prompt="hello   world")
    assert result.metadata["quality"]["rejected"] is False
    assert result.optimized_request.prompt == "hello world"


def _build_dummy_request():
    from opticore.core.model import AIRequest

    return AIRequest(prompt="hello   world", system="be helpful")

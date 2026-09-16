"""Tests for the canonical data model and optimizer contract."""

from __future__ import annotations

from typing import Any

from opticore import AIRequest, AIResponse, OptimizationConfig, Optimizer
from opticore.core.interfaces import OptimizerRequest
from opticore.core.model import ProviderResponse
from opticore.core.pipeline import build_default_pipeline


def test_ai_request_aliases_optimizer_request() -> None:
    assert AIRequest is OptimizerRequest


def test_ai_response_aliases_provider_response() -> None:
    assert AIResponse is ProviderResponse


def test_ai_request_new_fields() -> None:
    tools = [{"type": "function", "function": {"name": "search"}}]
    req = AIRequest(
        prompt="hi",
        model="m",
        temperature=0.2,
        tools=tools,
        namespace="team-a",
    )
    assert req.temperature == 0.2
    assert req.tools == tools
    assert req.namespace == "team-a"
    assert req.to_dict()["temperature"] == 0.2


def test_ai_response_total_tokens() -> None:
    response = AIResponse(content="x", model="m", provider="p", input_tokens=10, output_tokens=5)
    assert response.total_tokens == 15


def test_pipeline_preserves_schema_fields_verbatim() -> None:
    """Tool schemas and structured-output material must pass through untouched."""
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    pipeline = build_default_pipeline(OptimizationConfig(safety_mode="aggressive"))
    result = pipeline.run(
        request=AIRequest(
            prompt="What's the weather? " * 10,
            system="Be concise.",
            tools=tools,
            temperature=0.0,
            namespace="ns1",
        )
    )
    assert result.optimized_request.tools == tools
    assert result.optimized_request.namespace == "ns1"
    assert result.optimized_request.temperature == 0.0


def test_optimizer_result_has_audit_trail() -> None:
    optimizer = Optimizer()
    outcome = optimizer.optimize("hello   hello\nworld  world", system="sys")
    assert isinstance(outcome.changes, list)
    assert outcome.changes  # something was changed / described


def test_pipeline_metrics_use_canonical_keys() -> None:
    pipeline = build_default_pipeline()
    result = pipeline.run(prompt="hello world", system="be helpful")
    for key in (
        "original_input_tokens",
        "optimized_input_tokens",
        "tokens_saved",
        "reduction_percentage",
    ):
        assert key in result.metrics
    assert result.metrics["tokens_saved"] >= 0


def test_original_request_never_mutated() -> None:
    original = AIRequest(prompt="A  B  C", system="s")
    pipeline = build_default_pipeline()
    pipeline.run(request=original)
    assert original.prompt == "A  B  C"

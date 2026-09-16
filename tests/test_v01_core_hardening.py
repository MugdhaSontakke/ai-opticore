"""Tests for the v0.1 core-hardening pass.

Covers: pipeline timing/acceptance metadata, quality-gate correctness for
multi-turn requests, evaluator resolution, context budget=0, cache bounds and
stale policy, opt-in prompt logging, CLI exit codes, tool-calling passthrough,
measured cache hit rate in benchmarks, and the experimental inference package.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from opticore import AIClient, AIRequest, AIResponse, OptimizationConfig
from opticore.benchmarks.runner import BenchmarkRunner
from opticore.cache.base import CachePolicy, MemoryCache
from opticore.cache.semantic import SemanticCache
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.core.pipeline import build_default_pipeline
from opticore.exceptions import QualityEvaluationError
from opticore.logging import log_prompt_content
from opticore.optimizers.base import BaseOptimizer


class RecordingProvider:
    """Captures every payload sent to the provider."""

    name = "recording"

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def generate(self, request: dict[str, Any]) -> AIResponse:
        self.payloads.append(dict(request))
        return AIResponse(content="ok", model="m", provider=self.name)


class DrasticRewriteOptimizer(BaseOptimizer):
    """Changes the prompt to something completely different."""

    name = "drastic"
    order = 99

    def optimize(self, request, config) -> OptimizerResult:  # type: ignore[override]
        r = OptimizerRequest(
            prompt="totally different text than the user wrote",
            system=request.system,
            messages=request.messages,
            tools=request.tools,
            temperature=request.temperature,
            metadata=dict(request.metadata),
        )
        return OptimizerResult(
            optimized_request=r,
            original_request=request,
            original_tokens=1,
            optimized_tokens=1,
            optimizer_name=self.name,
            changes=["drastic rewrite applied"],
            metadata={},
        )


# --- Pipeline timing & acceptance -------------------------------------------


def test_pipeline_measures_optimizer_time() -> None:
    result = build_default_pipeline().run(prompt="hello   world", system="s")
    assert result.optimization_time_ms >= 0.0
    assert result.metadata["optimizer_time_ms"] >= 0.0
    assert result.metadata["optimizer_time_by_step_ms"]  # per-optimizer timing
    assert result.accepted is True
    assert result.rejection_reason is None


def test_quality_gate_includes_messages_in_comparison() -> None:
    """Similarity must consider conversation history, not just the prompt."""
    shared = "abcdefghijklmnopqrstuvwxyz0123456789" * 3
    messages = [{"role": "user", "content": shared}]
    config = OptimizationConfig(
        quality_enabled=True,
        quality_reject_on_failure=True,
        quality_minimum_score=0.85,
    )
    pipeline = build_default_pipeline(config)
    pipeline.add(DrasticRewriteOptimizer())
    result = pipeline.run(
        request=AIRequest(prompt="alpha", messages=messages),
    )
    # The shared (unchanged) message text dominates the character set, so the
    # optimization should NOT be rejected when messages are part of the gate.
    quality = result.metadata["quality"]
    assert quality["similarity"] >= 0.9
    assert result.accepted is True


def test_unknown_quality_evaluator_raises() -> None:
    config = OptimizationConfig(quality_evaluator="semantic")
    pipeline = build_default_pipeline(config)
    with pytest.raises(QualityEvaluationError):
        pipeline.run(prompt="hello world", system="s")


def test_token_jaccard_evaluator_is_resolved() -> None:
    config = OptimizationConfig(quality_evaluator="token_jaccard")
    pipeline = build_default_pipeline(config)
    result = pipeline.run(prompt="hello world", system="s")
    assert result.metadata["quality"]["evaluator"] == "token_jaccard"
    assert result.metadata["quality"]["verified"] is True


def test_pipeline_optimizer_disabled_flag_is_respected() -> None:
    pipeline = build_default_pipeline()
    disabled = DrasticRewriteOptimizer(enabled=False)
    pipeline.add(disabled)
    result = pipeline.run(prompt="alpha", system="s")
    assert "drastic" not in result.optimizers_run
    assert result.optimized_request.prompt == "alpha"


# --- Context optimizer budget -----------------------------------------------


def test_context_budget_zero_drops_all_messages() -> None:
    config = OptimizationConfig(max_token_budget=0)
    pipeline = build_default_pipeline(config)
    result = pipeline.run(
        request=AIRequest(
            prompt="hi",
            system="sys",
            messages=[
                {"role": "user", "content": "turn one"},
                {"role": "assistant", "content": "reply one"},
                {"role": "user", "content": "turn two"},
            ],
        )
    )
    assert result.optimized_request.messages == []
    meta = result.metadata["context"]
    assert meta["dropped_messages"] == 3
    assert any("dropped" in c for c in result.changes)


# --- Cache bounds & stale policy -------------------------------------------


def test_memory_cache_serve_stale_under_policy() -> None:
    cache = MemoryCache(policy=CachePolicy(serve_stale_for_seconds=5.0))
    cache.set("k", "content", "m", ttl_seconds=0.01)
    cache._store["k"].expires_at = time.monotonic() - 1.0  # force expiry
    entry = cache.get("k")
    assert entry is not None and entry.content == "content"


def test_memory_cache_never_serves_stale_by_default() -> None:
    cache = MemoryCache()
    cache.set("k", "content", "m", ttl_seconds=0.01)
    cache._store["k"].expires_at = time.monotonic() - 1.0
    assert cache.get("k") is None


def test_semantic_cache_index_is_bounded() -> None:
    cache = SemanticCache(max_entries=3)
    for i in range(10):
        cache.put(f"prompt {i}", f"content {i}", "m")
    assert len(cache._entries) <= 3
    # The store is independently bounded by its own max_entries default.
    assert cache.size() <= 10_000


# --- Opt-in prompt logging ---------------------------------------------------


def test_log_prompt_content_opt_in(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("opticore.test_prompt_log")
    with caplog.at_level(logging.DEBUG, logger="opticore.test_prompt_log"):
        log_prompt_content(logger, "original", "secret question", enabled=True)
    assert "[prompt:original] secret question" in caplog.text


def test_log_prompt_content_off_by_default(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("opticore.test_prompt_log_off")
    with caplog.at_level(logging.DEBUG, logger="opticore.test_prompt_log_off"):
        log_prompt_content(logger, "original", "secret question", enabled=False)
    assert "secret question" not in caplog.text


# --- Tool calling / structured output passthrough ----------------------------


def test_tools_reach_provider_unchanged() -> None:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    provider = RecordingProvider()
    client = AIClient(provider=provider, optimization=True)
    client.generate(
        prompt="What is the weather?   in Paris",
        system="call tools when needed",
        tools=tools,
        temperature=0.0,
        max_tokens=42,
    )
    payload = provider.payloads[-1]
    assert payload["tools"] == tools
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 42


def test_conversation_messages_reach_provider_unchanged() -> None:
    messages = [
        {"role": "system", "content": "use the schema"},
        {"role": "user", "content": "return JSON with key 'answer'"},
        {"role": "assistant", "content": "understood"},
    ]
    provider = RecordingProvider()
    client = AIClient(provider=provider, optimization=False)
    client.generate(prompt="ok", messages=messages)
    payload = provider.payloads[-1]
    assert payload["messages"] == messages


# --- Benchmark cache hit rate ------------------------------------------------


def test_benchmark_measures_cache_hit_rate() -> None:
    from opticore.providers.base import FakeProviderMixin

    class Fake(FakeProviderMixin):
        name = "fake-bench"

    config = OptimizationConfig(enable_semantic_cache=True)
    runner = BenchmarkRunner(
        provider=Fake(),
        model="m",
        samples=[OptimizerRequest(prompt="hit me twice", system="s")],
        config=config,
        repeats=2,
    )
    result = runner.run()
    assert result.metrics["cache_hits"] == 1
    assert result.metrics["cache_misses"] == 1
    assert result.metrics["cache_hit_rate"] == "50.0%"
    assert any("cache" in note for note in result.notes)


def test_benchmark_cache_disabled_reports_na() -> None:
    from opticore.providers.base import FakeProviderMixin

    class Fake(FakeProviderMixin):
        name = "fake-bench-2"

    runner = BenchmarkRunner(
        provider=Fake(), model="m", config=OptimizationConfig(enable_semantic_cache=False)
    )
    result = runner.run()
    assert result.metrics["cache_hit_rate"] == "N/A"


# --- CLI exit codes ----------------------------------------------------------


def test_cli_misuse_exits_with_code_2() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = {"PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "opticore.cli.main", "benchmark", "--samples", "nope"],
        capture_output=True,
        text=True,
        cwd=root,
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 2


def test_cli_broken_default_config_fails_loudly(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / "opticore.yaml").write_text("safety_mode: does-not-exist\n")
    root = Path(__file__).resolve().parent.parent
    env = {"PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "opticore.cli.main", "optimize", "--prompt", "hi"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 1
    assert "invalid" in result.stderr


# --- Experimental inference package -----------------------------------------


def test_batch_scheduler_interface_behavior() -> None:
    from opticore.inference.batching import MemoryBatchScheduler

    sched = MemoryBatchScheduler(max_batch_size=2)
    rid = sched.submit({"prompt": "a"})
    sched.submit({"prompt": "b"})
    results = sched.flush()
    assert [r.request_id for r in results] == [rid, "req-2"]
    assert sched.stats() == {"scheduler": "memory", "status": "experimental"}


def test_quantization_config_declares_planned_status() -> None:
    from opticore.inference.quantization import QuantizationConfig

    q = QuantizationConfig(bits=4)
    assert q.status() == "planned"
    assert "4-bit" in q.description()
    with pytest.raises(ValueError):
        QuantizationConfig(bits=7)


def test_memory_tracker_reports_something() -> None:
    from opticore.inference.runtime import MemoryTracker

    tracker = MemoryTracker()
    tracker.sample()
    report = tracker.reporting()
    assert report["samples_count"] == 1
    assert "peak_rss_bytes" in report
    assert "measurement" in report

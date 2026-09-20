"""Tests for pipeline and configuration."""

from __future__ import annotations

import pytest

from opticore.core.config import OptimizationConfig, ProviderConfig, SafetyMode
from opticore.core.interfaces import Optimizer, OptimizerRequest, OptimizerResult
from opticore.core.pipeline import OptimizationPipeline, build_default_pipeline
from opticore.exceptions import OptimizationError
from opticore.optimizers.token import Tokenizer


class CrashOptimizer(Optimizer):
    name = "crash"
    order = 0

    def optimize(self, request: OptimizerRequest, config: OptimizationConfig) -> OptimizerResult:  # type: ignore[override]
        raise RuntimeError("unexpected internal crash")


class OkOptimizer(CrashOptimizer):
    name = "ok"
    order = 1

    def optimize(self, request: OptimizerRequest, config: OptimizationConfig) -> OptimizerResult:  # type: ignore[override]
        return OptimizerResult(
            optimized_request=request,
            original_request=request,
            original_tokens=request.prompt.count(" "),
            optimized_tokens=request.prompt.count(" "),
            optimizer_name=self.name,
            optimizers_run=[self.name],
            tokens_saved=0,
            reduction_percent=0.0,
            changes=[],
            metrics={},
            metadata={},
            optimization_time_ms=0.0,
            accepted=True,
            rejection_reason=None,
        )


def test_safety_mode_value_stability() -> None:
    assert SafetyMode.SAFE.value == "safe"
    assert SafetyMode("balanced") is SafetyMode.BALANCED


def test_config_default() -> None:
    config = OptimizationConfig()
    assert config.safety_mode is SafetyMode.SAFE
    assert config.max_token_budget == 4096
    assert config.enable_token_optimization is True


def test_config_to_dict_roundtrip() -> None:
    config = OptimizationConfig(safety_mode=SafetyMode.BALANCED, max_token_budget=2048)
    d = config.to_dict()
    assert d["safety_mode"] == "balanced"
    reconstructed = OptimizationConfig.from_dict(d)
    assert reconstructed.safety_mode is SafetyMode.BALANCED
    assert reconstructed.max_token_budget == 2048


def test_default_pipeline_runs() -> None:
    pipeline = build_default_pipeline()
    result = pipeline.run(prompt="hello world", system="be helpful")
    assert result.optimized_request.prompt is not None
    assert result.original_tokens > 0


def test_pipeline_result_immutability() -> None:
    pipeline = build_default_pipeline()
    request = OptimizerRequest(prompt="test", system="s")
    result = pipeline.run(request=request)
    assert request.prompt == "test"
    assert result.original_request.prompt == "test"


def test_pipeline_reports_tokens_saved() -> None:
    pipeline = build_default_pipeline()
    result = pipeline.run(prompt="hello world", system="be helpful")
    assert result.tokens_saved >= 0
    assert "token" in result.optimizer_name or "prompt" in result.optimizer_name


def test_pipeline_respects_enable_flags() -> None:
    config = OptimizationConfig(enable_prompt_optimization=False, enable_context_optimization=False)
    pipeline = build_default_pipeline(config)
    result = pipeline.run(prompt="hello", system="sys")
    assert "prompt" not in result.optimizers_run
    assert "context" not in result.optimizers_run


def test_provider_config() -> None:
    config = ProviderConfig(provider="openai", model="gpt-4o", api_key_env="MY_KEY")
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.api_key_env == "MY_KEY"


def test_pipeline_skips_crashing_optimizer_by_default() -> None:
    pipeline = OptimizationPipeline(
        optimizers=[CrashOptimizer(), OkOptimizer()],
        tokenizer=Tokenizer(),
    )
    result = pipeline.run(prompt="a b c", system="s")
    assert result.optimized_request.prompt == "a b c"
    assert result.metadata.get("optimizer_errors")
    assert any("crash" in err for err in result.metadata["optimizer_errors"])


def test_pipeline_strict_mode_raises_on_crash() -> None:
    pipeline = OptimizationPipeline(
        optimizers=[CrashOptimizer()],
        tokenizer=Tokenizer(),
        strict_optimizers=True,
    )
    with pytest.raises(OptimizationError, match="crash"):
        pipeline.run(prompt="a b c", system="s")

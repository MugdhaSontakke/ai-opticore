"""Tests for the exception hierarchy and configuration validation."""

from __future__ import annotations

import pytest

from opticore import (
    CacheError,
    ConfigurationError,
    OptiCoreError,
    OptimizationConfig,
    ProviderTimeoutError,
    UnsupportedModelError,
)
from opticore.core.config import ProviderConfig, load_config
from opticore.optimizers.token import Tokenizer


def test_all_errors_share_base() -> None:
    assert issubclass(ConfigurationError, OptiCoreError)
    assert issubclass(CacheError, OptiCoreError)
    assert issubclass(ProviderTimeoutError, OptiCoreError)


def test_config_rejects_negative_budget() -> None:
    with pytest.raises(ConfigurationError):
        OptimizationConfig(max_token_budget=-1)


def test_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ConfigurationError):
        OptimizationConfig(semantic_cache_threshold=1.5)
    with pytest.raises(ConfigurationError):
        OptimizationConfig(quality_minimum_score=0.0)


def test_config_rejects_unknown_safety_mode() -> None:
    with pytest.raises(ValueError):
        OptimizationConfig(safety_mode="nuclear")


def test_provider_config_validation() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(provider="", timeout_seconds=0)
    with pytest.raises(ConfigurationError):
        ProviderConfig(provider="x", max_retries=-1)


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError):
        load_config("/tmp/does-not-exist-opticore.yaml")


def test_load_config_rejects_secret_in_file(tmp_path) -> None:
    secret_file = tmp_path / "bad.yaml"
    secret_file.write_text("api_key: sk-secret123\n")
    with pytest.raises(ConfigurationError):
        load_config(str(secret_file))


def test_load_config_json(tmp_path) -> None:
    config_file = tmp_path / "opticore.json"
    config_file.write_text('{"safety_mode": "balanced", "max_token_budget": 2048}')
    config = load_config(str(config_file), use_env=False)
    assert config.safety_mode.value == "balanced"
    assert config.max_token_budget == 2048


def test_load_config_yaml_nested_quality(tmp_path) -> None:
    yaml = pytest.importorskip("yaml")
    config_file = tmp_path / "opticore.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "safety_mode": "aggressive",
                "quality": {
                    "enabled": True,
                    "minimum_score": 0.9,
                    "reject_on_failure": True,
                },
            }
        )
    )
    config = load_config(str(config_file), use_env=False)
    assert config.safety_mode.value == "aggressive"
    assert config.quality_minimum_score == 0.9
    assert config.quality_reject_on_failure is True


def test_tokenizer_strict_unknown_model_raises() -> None:
    with pytest.raises(UnsupportedModelError):
        Tokenizer(model="definitely-not-a-model-xyz", strict=True).count("hi")


def test_tokenizer_lenient_falls_back_explicitly() -> None:
    tokenizer = Tokenizer(model="definitely-not-a-model-xyz")
    assert tokenizer.count("hi") >= 1
    assert tokenizer.mode == "fallback"
    assert tokenizer.fallback_for == "definitely-not-a-model-xyz"
    assert "fallback" in tokenizer.description()


def test_tokenizer_report_shape() -> None:
    report = Tokenizer().report()
    assert report["mode"] in ("encoding", "model")
    assert "description" in report


def test_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPTICORE_SAFETY_MODE", "balanced")
    monkeypatch.setenv("OPTICORE_MAX_TOKEN_BUDGET", "512")
    config = load_config(use_env=True)
    assert config.safety_mode.value == "balanced"
    assert config.max_token_budget == 512


def test_env_override_invalid_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPTICORE_MAX_TOKEN_BUDGET", "not-a-number")
    with pytest.raises(ConfigurationError):
        load_config(use_env=True)

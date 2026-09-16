"""Core configuration types for AI-OptiCore."""

from __future__ import annotations

from enum import Enum
from typing import Any


class SafetyMode(str, Enum):  # noqa: UP042 - kept (str, Enum) for py3.9 dev compatibility
    """Operational safety mode for optimizations.

    SAFE: Minimal, reversible transformations only.
    BALANCED: Moderate transformations that preserve information.
    AGGRESSIVE: Maximum reduction with potentially higher quality impact.

    The default is SAFE. Optimizers must respect this setting.
    """

    SAFE = "safe"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

    def aggressiveness(self) -> int:
        """Return the numeric aggressiveness level (1-3)."""
        return {
            SafetyMode.SAFE: 1,
            SafetyMode.BALANCED: 2,
            SafetyMode.AGGRESSIVE: 3,
        }[self]


class OptimizationConfig:
    """Configuration for the optimization pipeline.

    All optimizers receive this config and act accordingly. Fields are
    deliberately plain so the config can be serialized and extended without
    depending on a specific validation framework at import time.
    """

    def __init__(
        self,
        safety_mode: SafetyMode | str = SafetyMode.SAFE,
        max_token_budget: int = 4096,
        enable_token_optimization: bool = True,
        enable_prompt_optimization: bool = True,
        enable_context_optimization: bool = True,
        enable_semantic_cache: bool = True,
        enable_model_routing: bool = True,
        semantic_cache_threshold: float = 0.90,
        cache_ttl_seconds: int = 3600,
        prioritize_recent: bool = True,
        log_prompts: bool = False,
        **extra: Any,
    ) -> None:
        self.safety_mode = SafetyMode(safety_mode)
        self.max_token_budget = max_token_budget
        self.enable_token_optimization = enable_token_optimization
        self.enable_prompt_optimization = enable_prompt_optimization
        self.enable_context_optimization = enable_context_optimization
        self.enable_semantic_cache = enable_semantic_cache
        self.enable_model_routing = enable_model_routing
        self.semantic_cache_threshold = semantic_cache_threshold
        self.cache_ttl_seconds = cache_ttl_seconds
        self.prioritize_recent = prioritize_recent
        self.log_prompts = log_prompts
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation for logging/metrics."""
        return {
            "safety_mode": self.safety_mode.value,
            "max_token_budget": self.max_token_budget,
            "enable_token_optimization": self.enable_token_optimization,
            "enable_prompt_optimization": self.enable_prompt_optimization,
            "enable_context_optimization": self.enable_context_optimization,
            "enable_semantic_cache": self.enable_semantic_cache,
            "enable_model_routing": self.enable_model_routing,
            "semantic_cache_threshold": self.semantic_cache_threshold,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "prioritize_recent": self.prioritize_recent,
            "log_prompts": self.log_prompts,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationConfig:
        """Rebuild a config from a dict, ignoring unknown keys is not done here.

        Unknown keys are preserved in ``extra``.
        """
        known = {
            "safety_mode",
            "max_token_budget",
            "enable_token_optimization",
            "enable_prompt_optimization",
            "enable_context_optimization",
            "enable_semantic_cache",
            "enable_model_routing",
            "semantic_cache_threshold",
            "cache_ttl_seconds",
            "prioritize_recent",
            "log_prompts",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        config = cls(**{k: v for k, v in data.items() if k in known})
        config.extra = extra
        return config


class ProviderConfig:
    """Configuration for a model provider connection.

    API keys are read from environment variables by the provider itself and
    are never required to be stored in this object.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.extra = extra or {}


def default_config() -> OptimizationConfig:
    """Return the default optimization configuration."""
    return OptimizationConfig()

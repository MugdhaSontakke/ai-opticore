"""Core configuration types and loaders for AI-OptiCore."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from opticore.exceptions import ConfigurationError

_SENSITIVE_CONFIG_KEYS = {"api_key", "apikey", "token", "secret", "password"}


def _find_sensitive_keys(mapping: dict[str, Any], prefix: str = "") -> list[str]:
    """Find secret-like keys at any nesting depth (e.g. ``provider.api_key``).

    Guards against nested secrets bypassing the top-level-only check.
    """
    found: list[str] = []
    for key, value in mapping.items():
        location = f"{prefix}{key}"
        if key.lower() in _SENSITIVE_CONFIG_KEYS:
            found.append(location)
        if isinstance(value, dict):
            found.extend(_find_sensitive_keys(value, f"{location}."))
    return found


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


def _validate(value: Any, name: str, allowed: set[Any]) -> None:
    if value not in allowed:
        raise ConfigurationError(
            f"Invalid value for {name!r}: {value!r}. Allowed: {sorted(allowed)}"
        )


class OptimizationConfig:
    """Configuration for the optimization pipeline.

    Validation is eager: invalid values raise :class:`ConfigurationError`
    at construction time so misconfiguration fails fast.
    """

    def __init__(
        self,
        safety_mode: SafetyMode | str = SafetyMode.SAFE,
        max_token_budget: int = 4096,
        enable_token_optimization: bool = True,
        enable_prompt_optimization: bool = True,
        enable_context_optimization: bool = True,
        enable_semantic_cache: bool = True,
        enable_model_routing: bool = False,
        semantic_cache_threshold: float = 0.90,
        cache_ttl_seconds: float = 3600.0,
        prioritize_recent: bool = True,
        quality_enabled: bool = True,
        quality_minimum_score: float = 0.85,
        quality_reject_on_failure: bool = False,
        quality_evaluator: str = "character",
        log_prompts: bool = False,
        **extra: Any,
    ) -> None:
        try:
            self.safety_mode = SafetyMode(safety_mode)
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid value for 'safety_mode': {safety_mode!r}. "
                f"Allowed: {[m.value for m in SafetyMode]}"
            ) from exc
        if max_token_budget < 0:
            raise ConfigurationError("max_token_budget must be >= 0")
        self.max_token_budget = max_token_budget
        self.enable_token_optimization = bool(enable_token_optimization)
        self.enable_prompt_optimization = bool(enable_prompt_optimization)
        self.enable_context_optimization = bool(enable_context_optimization)
        self.enable_semantic_cache = bool(enable_semantic_cache)
        self.enable_model_routing = bool(enable_model_routing)
        if not 0.0 < semantic_cache_threshold <= 1.0:
            raise ConfigurationError(
                "semantic_cache_threshold must be in (0, 1]"
            )
        self.semantic_cache_threshold = float(semantic_cache_threshold)
        if cache_ttl_seconds < 0:
            raise ConfigurationError("cache_ttl_seconds must be >= 0")
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self.prioritize_recent = bool(prioritize_recent)
        self.quality_enabled = bool(quality_enabled)
        if not 0.0 < quality_minimum_score <= 1.0:
            raise ConfigurationError("quality_minimum_score must be in (0, 1]")
        self.quality_minimum_score = float(quality_minimum_score)
        self.quality_reject_on_failure = bool(quality_reject_on_failure)
        self.quality_evaluator = quality_evaluator or "character"
        self.log_prompts = bool(log_prompts)
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation for logging/metrics.

        Sensitive keys are never included; callers must redact ``extra``.
        """
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
            "quality_enabled": self.quality_enabled,
            "quality_minimum_score": self.quality_minimum_score,
            "quality_reject_on_failure": self.quality_reject_on_failure,
            "quality_evaluator": self.quality_evaluator,
            "log_prompts": self.log_prompts,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationConfig:
        """Rebuild a config from a dict; unknown keys go to ``extra``.

        A nested ``quality`` mapping (``quality: {enabled, minimum_score,
        reject_on_failure, evaluator}``) is flattened into the known fields.
        """
        data = dict(data)
        quality = data.pop("quality", None)
        if isinstance(quality, dict):
            for key, value in quality.items():
                data[f"quality_{key}"] = value
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
            "quality_enabled",
            "quality_minimum_score",
            "quality_reject_on_failure",
            "quality_evaluator",
            "log_prompts",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        try:
            config = cls(**{k: v for k, v in data.items() if k in known})
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid configuration: {exc}") from exc
        config.extra = extra
        return config


class ProviderConfig:
    """Configuration for a model provider connection.

    API keys are read from environment variables by the provider itself and
    are never required to be stored in this object. If a key is passed, it is
    treated as an environment variable name (``api_key_env``), not a secret.
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
        if not provider:
            raise ConfigurationError("provider name must not be empty")
        if timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be > 0")
        if max_retries < 0:
            raise ConfigurationError("max_retries must be >= 0")
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


def _apply_env_overrides(config: OptimizationConfig) -> OptimizationConfig:
    """Apply ``OPTICORE_*`` environment overrides on top of a config."""
    try:
        if (value := os.environ.get("OPTICORE_SAFETY_MODE")) is not None:
            config.safety_mode = SafetyMode(value)
        if (value := os.environ.get("OPTICORE_MAX_TOKEN_BUDGET")) is not None:
            config.max_token_budget = int(value)
        if (value := os.environ.get("OPTICORE_SEMANTIC_CACHE_THRESHOLD")) is not None:
            config.semantic_cache_threshold = float(value)
        if (value := os.environ.get("OPTICORE_CACHE_TTL_SECONDS")) is not None:
            config.cache_ttl_seconds = float(value)
        if (value := os.environ.get("OPTICORE_QUALITY_MINIMUM_SCORE")) is not None:
            config.quality_minimum_score = float(value)
        for flag in (
            "enable_semantic_cache",
            "enable_token_optimization",
            "enable_prompt_optimization",
            "enable_context_optimization",
            "enable_model_routing",
            "quality_enabled",
            "quality_reject_on_failure",
            "log_prompts",
        ):
            env_name = f"OPTICORE_{flag.upper()}"
            if (value := os.environ.get(env_name)) is not None:
                setattr(config, flag, value.lower() in ("1", "true", "yes", "on"))
        return config
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Environment override invalid: {exc}") from exc


def load_config(path: str | None = None, *, use_env: bool = True) -> OptimizationConfig:
    """Load configuration from a YAML/JSON file plus optional env overrides.

    Secrets are not read here; ``api_key``-like keys in a file are rejected
    with a warning-level :class:`ConfigurationError` because AI-OptiCore never
    stores secrets in config files.
    """
    data: dict[str, Any] = {}

    if path:
        if not os.path.exists(path):
            raise ConfigurationError(f"Configuration file not found: {path}")
        suffix = path.rsplit(".", 1)[-1].lower()
        if suffix in ("yaml", "yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ConfigurationError(
                    "Reading YAML config requires PyYAML (pip install pyyaml)"
                ) from exc
            with open(path) as fh:
                loaded = yaml.safe_load(fh)
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ConfigurationError("YAML config root must be a mapping")
                data = loaded
        elif suffix == "json":
            with open(path) as fh:
                import json

                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ConfigurationError("JSON config root must be a mapping")
            data = loaded
        else:
            raise ConfigurationError(
                f"Unsupported config extension {suffix!r} (use .yaml, .yml or .json)"
            )

        secrets = _find_sensitive_keys(data)
        if secrets:
            raise ConfigurationError(
                "Do not store API keys/secrets in config files. "
                f"Found keys: {', '.join(secrets)}. Use environment variables instead."
            )

    # Support a nested {"provider": ...} block without losing it.
    config = OptimizationConfig.from_dict(data)
    if use_env:
        config = _apply_env_overrides(config)
    return config


def config_template() -> dict[str, Any]:
    """Return an annotated example configuration dict (used by CLI ``init``)."""
    return {
        # yaml-language-server: $schema=...
        "safety_mode": "safe",
        "max_token_budget": 4096,
        "enable_token_optimization": True,
        "enable_prompt_optimization": True,
        "enable_context_optimization": True,
        "enable_semantic_cache": True,
        "enable_model_routing": False,
        "semantic_cache_threshold": 0.90,
        "cache_ttl_seconds": 3600,
        "prioritize_recent": True,
        "quality": {
            "enabled": True,
            "minimum_score": 0.85,
            "reject_on_failure": False,
            "evaluator": "character",
        },
        "log_prompts": False,
    }

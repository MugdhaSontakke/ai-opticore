"""Public exception hierarchy for AI-OptiCore.

Errors are typed and actionable. Components never swallow exceptions silently;
they raise a specific subclass so callers can react appropriately.
"""

from __future__ import annotations


class OptiCoreError(Exception):
    """Base class for all AI-OptiCore-specific errors."""


class ConfigurationError(OptiCoreError):
    """Raised when user configuration is invalid or unreadable."""


class OptimizationError(OptiCoreError):
    """Raised when an optimizer fails during request transformation."""


class EvaluationError(OptiCoreError):
    """Base class for evaluation failures (quality gates, similarity checks)."""


class QualityEvaluationError(EvaluationError):
    """Raised when a quality evaluation cannot be computed."""


class CacheError(OptiCoreError):
    """Raised when a cache backend fails outside of a best-effort path."""


class RoutingError(OptiCoreError):
    """Raised when no eligible model can be selected for a request."""


class ProviderError(OptiCoreError):
    """Base class for provider failures."""


class ProviderAuthError(ProviderError):
    """Raised when a provider reports missing/invalid credentials."""


class ProviderRateError(ProviderError):
    """Raised on provider-side rate limiting or quota exhaustion."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call exceeds its configured timeout."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider cannot be reached or fails transiently.

    Covers connection failures and transient server errors (5xx) that remain
    after the configured retry budget is exhausted.
    """


class ProviderResponseError(ProviderError):
    """Raised when a provider returns a malformed or invalid response.

    This includes missing fields, empty completion choices, and unparseable
    bodies — anything that prevents a usable :class:`AIResponse` from being
    constructed.
    """


class UnsupportedModelError(OptiCoreError):
    """Raised when a model is not supported for the requested operation."""


class TokenizerNotFoundError(OptiCoreError):
    """Raised when a tokenizer for the requested model/encoding is unavailable."""


class HardwareBackendUnavailableError(OptiCoreError):
    """Raised when a required hardware backend is not available."""

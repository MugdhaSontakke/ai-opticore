"""AI-OptiCore - An open-source optimization layer for AI/LLM applications.

Public API (stable) is intentionally small:

    from opticore import Optimizer, AIClient, OptimizationConfig

Stability classification is documented in ``docs/api_stability.md``:

- **Stable** — backward-compatible (``AIClient``, ``Optimizer``,
  ``OptimizationConfig``, ``SafetyMode``, the canonical request/response and
  result types, and the exception hierarchy).
- **Experimental** — may change without notice (``OptimizationPipeline``,
  cache backends, model routing, costing, ``Tokenizer``, ``ProviderConfig``).
- **Internal** — use at your own risk; not part of the public contract.
"""

from __future__ import annotations

__title__ = "ai-opticore"
__version__ = "0.3.0"
__license__ = "MIT"

from opticore.api import AIClient, GenerationResult, OptimizeOutcome, Optimizer
from opticore.cache import BaseCache, DiskCache, MemoryCache, SemanticCache, cache_key
from opticore.core.config import OptimizationConfig, ProviderConfig, SafetyMode
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.core.model import AIRequest, AIResponse
from opticore.core.pipeline import OptimizationPipeline
from opticore.costing import CostBreakdown, CostEstimator
from opticore.exceptions import (
    CacheError,
    ConfigurationError,
    EvaluationError,
    HardwareBackendUnavailableError,
    OptiCoreError,
    OptimizationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QualityEvaluationError,
    RoutingError,
    TokenizerNotFoundError,
    UnsupportedModelError,
)
from opticore.optimizers.token import Tokenizer
from opticore.routing.router import ModelRouter, RouteRule, RoutingDecision

__all__ = [
    "AIClient",
    "AIRequest",
    "AIResponse",
    "BaseCache",
    "CacheError",
    "ConfigurationError",
    "CostBreakdown",
    "CostEstimator",
    "DiskCache",
    "EvaluationError",
    "GenerationResult",
    "HardwareBackendUnavailableError",
    "MemoryCache",
    "ModelRouter",
    "OptiCoreError",
    "OptimizationConfig",
    "OptimizationError",
    "OptimizationPipeline",
    "Optimizer",
    "OptimizerRequest",
    "OptimizerResult",
    "OptimizeOutcome",
    "ProviderAuthError",
    "ProviderConfig",
    "ProviderError",
    "ProviderRateError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QualityEvaluationError",
    "RouteRule",
    "RoutingDecision",
    "RoutingError",
    "SafetyMode",
    "SemanticCache",
    "Tokenizer",
    "TokenizerNotFoundError",
    "UnsupportedModelError",
    "cache_key",
]

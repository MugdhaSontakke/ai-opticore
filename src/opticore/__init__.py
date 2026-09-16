"""AI-OptiCore - An open-source optimization layer for AI/LLM applications.

Public API (stable) is intentionally small:

    from opticore import Optimizer, AIClient, OptimizationConfig

Everything else is internal or experimental and may change without notice.
"""

from __future__ import annotations

__title__ = "ai-opticore"
__version__ = "0.1.0"
__license__ = "MIT"

from opticore.api import AIClient, GenerationResult, OptimizeOutcome, Optimizer
from opticore.core.config import OptimizationConfig, SafetyMode
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.core.model import AIRequest, AIResponse
from opticore.core.pipeline import OptimizationPipeline
from opticore.exceptions import (
    CacheError,
    ConfigurationError,
    HardwareBackendUnavailableError,
    OptiCoreError,
    OptimizationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateError,
    ProviderTimeoutError,
    QualityEvaluationError,
    TokenizerNotFoundError,
    UnsupportedModelError,
)

__all__ = [
    "AIClient",
    "AIRequest",
    "AIResponse",
    "CacheError",
    "ConfigurationError",
    "GenerationResult",
    "HardwareBackendUnavailableError",
    "OptiCoreError",
    "OptimizationConfig",
    "OptimizationError",
    "OptimizationPipeline",
    "Optimizer",
    "OptimizerRequest",
    "OptimizerResult",
    "OptimizeOutcome",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateError",
    "ProviderTimeoutError",
    "QualityEvaluationError",
    "SafetyMode",
    "TokenizerNotFoundError",
    "UnsupportedModelError",
]

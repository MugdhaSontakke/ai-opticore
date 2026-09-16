"""AI-OptiCore - An open-source optimization layer for AI/LLM applications."""

from __future__ import annotations

__title__ = "ai-opticore"
__version__ = "0.1.0"
__license__ = "MIT"

from opticore.api import AIClient, GenerationResult, OptimizeOutcome, Optimizer
from opticore.core.config import OptimizationConfig, SafetyMode
from opticore.core.interfaces import OptimizerRequest
from opticore.core.pipeline import OptimizationPipeline, OptimizerResult

__all__ = [
    "AIClient",
    "GenerationResult",
    "OptimizationConfig",
    "OptimizationPipeline",
    "Optimizer",
    "OptimizerRequest",
    "OptimizerResult",
    "OptimizeOutcome",
    "SafetyMode",
]

"""Core interfaces for AI-OptiCore.

The canonical data model lives in :mod:`opticore.core.model`. Here we keep the
optimizer/pipeline interfaces and the backward-compatible request name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from opticore.core.model import AIRequest, AIResponse, OptimizerRequest

__all__ = [
    "AIRequest",
    "AIResponse",
    "OptimizerRequest",
    "OptimizerResult",
    "ModelProvider",
    "HardwareBackend",
    "Optimizer",
]


@dataclass
class OptimizerResult:
    """The result of running one or more optimizers on a request.

    - ``optimized_request`` contains the transformed request.
    - ``original_request`` is the untouched caller-owned request.
    - ``changes`` is an auditable list of what each optimizer modified.
    - ``metrics`` uses canonical token keys (see :func:`opticore.benchmarks
      .metrics.token_metrics`).
    """

    optimized_request: OptimizerRequest
    original_request: OptimizerRequest
    original_tokens: int = 0
    optimized_tokens: int = 0
    optimizer_name: str = ""
    optimizers_run: list[str] = field(default_factory=list)
    tokens_saved: int = 0
    reduction_percent: float = 0.0
    changes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    """Protocol for language model providers."""

    name: str

    def generate(self, request: dict[str, Any]) -> AIResponse:
        """Generate a response for a request.

        Args:
            request: A dict with keys such as ``prompt``, ``system``,
                ``max_tokens``, ``model``.

        Returns:
            An :class:`AIResponse` with content and usage data.
        """


class HardwareBackend(Protocol):
    """Protocol for hardware backends."""

    name: str

    def available(self) -> bool:
        """Return True if this hardware is available in the environment."""

    def capabilities(self) -> dict[str, Any]:
        """Return a dict describing backend-specific capabilities."""


class Optimizer(Protocol):
    """Protocol for pipeline optimizers.

    Every optimizer must expose ``name``, ``order`` and ``optimize``; see
    optimizers.base.BaseOptimizer for the reference implementation.
    """

    name: str
    order: int = 100

    def optimize(
        self, request: OptimizerRequest, config: Any
    ) -> OptimizerResult:
        """Transform a request and report token impact."""

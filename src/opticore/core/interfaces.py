"""Core interfaces and data structures for AI-OptiCore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class OptimizerRequest:
    """The normalized request object flowing through the pipeline.

    Optimizers transform this object; they never mutate caller-owned data.
    """

    prompt: str
    system: str | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerResult:
    """The result of running one or more optimizers on a request.

    ``optimized_request`` contains the transformed request. The original
    request is retained unchanged on ``original_request``.
    """

    optimized_request: OptimizerRequest
    original_request: OptimizerRequest
    original_tokens: int = 0
    optimized_tokens: int = 0
    optimizer_name: str = ""
    optimizers_run: list[str] = field(default_factory=list)
    tokens_saved: int = 0
    reduction_percent: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    """Protocol for language model providers.

    Implementers must also expose their capability metadata. The abstract
    base class in providers/base.py provides a convenient starting point.
    """

    name: str

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate a response for a request.

        Args:
            request: A dict with keys such as ``prompt``, ``system``,
                ``max_tokens``, ``model``.

        Returns:
            A dict with ``content``, ``model``, ``provider``, ``input_tokens``,
            ``output_tokens`` and ``metadata``.
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

    Every optimizer must expose ``name`` and ``optimize``; see
    optimizers.base.BaseOptimizer for the reference implementation.
    """

    name: str
    order: int = 100

    def optimize(
        self, request: OptimizerRequest, config: Any
    ) -> OptimizerResult:
        """Transform a request and report token impact."""

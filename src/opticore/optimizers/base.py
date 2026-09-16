"""Base optimizer interface shared by all optimizers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest, OptimizerResult

logger = logging.getLogger("opticore.optimizers")


class BaseOptimizer(ABC):
    """Reference implementation of the :class:`Optimizer` protocol.

    Subclasses override :meth:`optimize` and :meth:`tokenize` usage. Each
    optimizer reports token counts for both the input and output request so
    the pipeline can aggregate savings.
    """

    name: str = "base"
    order: int = 100
    description: str = "Base optimizer"

    def __init__(self, **kwargs: Any) -> None:
        self.batch_size = kwargs.get("batch_size")
        self.enabled = kwargs.get("enabled", True)

    @abstractmethod
    def optimize(
        self, request: OptimizerRequest, config: OptimizationConfig
    ) -> OptimizerResult:
        """Transform the request according to this optimizer's logic."""

"""Configurable, clearly-labeled *estimated* cost accounting.

Pricing is user-configured (never hardcoded provider prices, which go stale).
When no prices are configured, costs are reported as ``None`` and the
``estimated`` flag is ``False`` so callers never mistake a real bill for a
guess (or a guess for a bill).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CostEstimator", "CostBreakdown"]


@dataclass(frozen=True)
class CostBreakdown:
    """Estimated cost for one request (or averaged per request).

    All money fields are ``None`` when pricing is not configured; ``estimated``
    is always ``True`` because pricing is user-supplied, not provider-quoted.
    """

    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    estimated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "estimated": self.estimated,
        }


class CostEstimator:
    """Computes estimated cost from configurable per-1k-token prices.

    A price of ``0.0`` means "not configured" and yields ``None`` costs rather
    than a confident $0.00 figure.
    """

    def __init__(
        self,
        input_price_per_1k: float = 0.0,
        output_price_per_1k: float = 0.0,
    ) -> None:
        if input_price_per_1k < 0 or output_price_per_1k < 0:
            raise ValueError("pricing must be non-negative")
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k

    @property
    def configured(self) -> bool:
        """True when at least one price is set, so costs can be estimated."""
        return self.input_price_per_1k > 0 or self.output_price_per_1k > 0

    def estimate(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> CostBreakdown:
        """Return an estimated cost breakdown for a token pair."""
        if not self.configured:
            return CostBreakdown(None, None, None)
        input_cost = (input_tokens / 1000.0) * self.input_price_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_price_per_1k
        return CostBreakdown(
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(input_cost + output_cost, 6),
        )

    def savings(self, baseline: CostBreakdown, optimized: CostBreakdown) -> CostBreakdown:
        """Difference between two estimated breakdowns (baseline minus optimized)."""
        if baseline.total_cost is None or optimized.total_cost is None:
            return CostBreakdown(None, None, None)
        b_in, b_out = baseline.input_cost or 0.0, baseline.output_cost or 0.0
        o_in, o_out = optimized.input_cost or 0.0, optimized.output_cost or 0.0
        return CostBreakdown(
            input_cost=round(b_in - o_in, 6),
            output_cost=round(b_out - o_out, 6),
            total_cost=round((b_in + b_out) - (o_in + o_out), 6),
        )

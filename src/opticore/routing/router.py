"""Optional model routing based on request characteristics.

The router does NOT promise accuracy improvements. It selects a model based
on configurable signals (complexity, budget, latency, cost). Users can disable
routing entirely; when disabled, the configured default model is always used.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger("opticore.routing")

ComplexityFn = Callable[[str], float]


@dataclass
class RouteRule:
    """A single routing rule: match condition -> target model."""

    model: str
    min_complexity: float = 0.0
    max_complexity: float = 1.0
    max_tokens: int | None = None
    max_latency_ms: float | None = None


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    model: str
    reason: str
    rules_considered: list[str] = field(default_factory=list)


def default_complexity(text: str) -> float:
    """A conservative, deterministic complexity heuristic (0..1).

    Based on length and structural depth. This is only a signal; it does not
    measure true reasoning difficulty.
    """
    if not text:
        return 0.0
    length = len(text)
    sentences = max(1, text.count(".") + text.count("?") + text.count("!"))
    avg_sentence = length / sentences
    score = min(1.0, avg_sentence / 120)
    if "explain" in text.lower() or "why" in text.lower() or "compare" in text.lower():
        score = min(1.0, score + 0.1)
    return score


class ModelRouter:
    """Routes requests to eligible models.

    The router picks the first rule whose conditions match the request
    features. Rules are evaluated in order; tie-breaking favors rules
    declared earlier.
    """

    def __init__(
        self,
        rules: list[RouteRule] | None = None,
        default_model: str = "small",
        complexity_fn: ComplexityFn = default_complexity,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.rules = rules or []
        self.default_model = default_model
        self.complexity_fn = complexity_fn
        self.fallback_models = fallback_models or []

    def route(self, *, prompt: str, token_count: int | None = None) -> RoutingDecision:
        """Determine the best model for ``prompt``."""
        complexity = self.complexity_fn(prompt)
        for index, rule in enumerate(self.rules):
            if (
                rule.min_complexity is not None
                and complexity < rule.min_complexity
            ):
                continue
            if rule.max_complexity is not None and complexity > rule.max_complexity:
                continue
            if (
                rule.max_tokens is not None
                and token_count is not None
                and token_count > rule.max_tokens
            ):
                continue
            # Latency preference is currently a soft constraint; callers can
            # provide a latency target via future routing config. The rule
            # stays structurally available without pretending to measure
            # latency here.
            considered = [r.model for r in self.rules[: index + 1]]
            return RoutingDecision(
                model=rule.model,
                reason=f"complexity={complexity:.2f} matched rule order {index}",
                rules_considered=considered,
            )
        considered = [r.model for r in self.rules]
        return RoutingDecision(
            model=self.default_model,
            reason=f"no rule matched (complexity={complexity:.2f}); using default",
            rules_considered=considered,
        )

    def models_in_rules(self) -> list[str]:
        names = [r.model for r in self.rules]
        names += [self.default_model]
        names += self.fallback_models
        return list(dict.fromkeys(names))


def default_rules() -> list[RouteRule]:
    """Sample routing rules that users should review and customize."""
    return [
        RouteRule(model="small", max_complexity=0.35),
        RouteRule(
            model="medium", min_complexity=0.35, max_complexity=0.7, max_tokens=1200
        ),
        RouteRule(model="large", min_complexity=0.7),
    ]

"""Quality/safety evaluation: original vs optimized requests and responses.

Provides interfaces for semantic similarity, response similarity, and task
success evaluation. Evaluators are pluggable; a deterministic character-level
similarity is provided by default so the system works without ML deps.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from opticore.logging import get_logger

logger = get_logger("opticore.evaluation")

SimilarityFn = Callable[[str, str], float]


def character_similarity(a: str, b: str) -> float:
    """Normalized character overlap similarity in [0, 1].

    A cheap deterministic proxy that requires no ML dependencies. For real
    semantic evaluation, plug in an embedding-based function.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return inter / (len(set_a) + len(set_b) - inter)


class SimilarityEvaluator(ABC):
    """Approach for comparing two texts."""

    name = "base"

    @abstractmethod
    def similarity(self, original: str, optimized: str) -> float:
        """Return similarity in [0, 1] where 1 means identical."""


class CharacterSimilarity(SimilarityEvaluator):
    """Character-level similarity (deterministic, no dependencies)."""

    name = "character"

    def similarity(self, original: str, optimized: str) -> float:
        return character_similarity(original, optimized)


class TokenJaccardSimilarity(SimilarityEvaluator):
    """Jaccard similarity over whitespace tokens (deterministic)."""

    name = "token_jaccard"

    def similarity(self, original: str, optimized: str) -> float:
        if not original and not optimized:
            return 1.0
        if not original or not optimized:
            return 0.0
        a = set(re.findall(r"\w+", original.lower()))
        b = set(re.findall(r"\w+", optimized.lower()))
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)


class ResponseEvaluator(ABC):
    """Evaluates whether an optimized path preserved task quality."""

    name = "base"

    @abstractmethod
    def evaluate(
        self,
        original_response: str,
        optimized_response: str,
        original_request: str,
        optimized_request: str,
    ) -> dict[str, Any]:
        """Return a dict of evaluation metrics."""


class DefaultResponseEvaluator(ResponseEvaluator):
    """Reports similarities between original and optimized requests/responses.

    It does not claim task success; it reports measurable proxies and lets
    callers set a threshold for the "quality preserved" verdict.
    """

    name = "default"

    def __init__(
        self,
        request_similarity: SimilarityEvaluator | None = None,
        response_similarity: SimilarityEvaluator | None = None,
    ) -> None:
        self._req_sim = request_similarity or CharacterSimilarity()
        self._resp_sim = response_similarity or CharacterSimilarity()

    def evaluate(
        self,
        original_response: str,
        optimized_response: str,
        original_request: str,
        optimized_request: str,
    ) -> dict[str, Any]:
        return {
            "request_similarity": round(
                self._req_sim.similarity(original_request, optimized_request), 4
            ),
            "response_similarity": round(
                self._resp_sim.similarity(original_response, optimized_response), 4
            ),
            "quality_preserved": self._resp_sim.similarity(
                original_response, optimized_response
            )
            >= 0.98,
            "evaluator": self._resp_sim.name,
        }


class QualityGate:
    """Warns when aggressive optimization may affect quality.

    Compares optimized vs original; a verdict is returned with a severity
    level so callers can surface warnings.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def evaluate(self, similarity: float, safety_mode: str) -> dict[str, Any]:
        warning: list[str] = []
        if similarity < self.threshold:
            warning.append(
                f"response similarity {similarity:.2f} below gate {self.threshold}"
            )
        if safety_mode == "aggressive":
            warning.append("aggressive mode may affect quality; verify outputs")
        return {
            "passed": not warning,
            "similarity": round(similarity, 4),
            "warnings": warning,
        }


class TaskSuccessFunc(ABC):
    """Optional task-success evaluator (e.g., for structured outputs)."""

    name = "task_success"

    @abstractmethod
    def success(self, response: str, expected: Any) -> float:
        """Return success probability in [0,1] or 0 if unknown."""

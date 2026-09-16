"""The optimization pipeline: runs optimizers in sequence over a request.

The pipeline enforces the safe-optimization contract:

    Original request
         ↓
    Optimizer(s)
         ↓
    Optimized request
         ↓
    Quality gate (if configured)
         ↓
    PASS -> optimized request
    FAIL -> original request

A failing optimizer never crashes the pipeline: it is recorded on the result
and skipped. Token metrics are aggregated under canonical key names.
"""

from __future__ import annotations

import time
from typing import Any

from opticore.benchmarks.metrics import token_metrics
from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import Optimizer, OptimizerResult
from opticore.core.model import AIRequest
from opticore.exceptions import OptimizationError, QualityEvaluationError
from opticore.logging import get_logger, log_prompt_content
from opticore.optimizers.token import Tokenizer, count_tokens

logger = get_logger("opticore.pipeline")


class OptimizationPipeline:
    """Sequentially applies optimizers to an :class:`AIRequest`.

    Optimizers run in ascending ``order``. Each optimizer receives the request
    output from the previous one. The original request is preserved for
    comparison and evaluation.
    """

    def __init__(
        self,
        optimizers: list[Optimizer] | None = None,
        config: OptimizationConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.optimizers = sorted(optimizers or [], key=lambda o: o.order)
        self.config = config or OptimizationConfig()
        self.tokenizer = tokenizer or Tokenizer()

    def add(self, optimizer: Optimizer) -> None:
        """Register an optimizer and keep the list sorted by ``order``."""
        self.optimizers.append(optimizer)
        self.optimizers.sort(key=lambda o: o.order)

    def run(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        request: AIRequest | None = None,
    ) -> OptimizerResult:
        """Execute all enabled optimizers against a request.

        Either pass the individual parts or a pre-built ``request``.
        """
        if request is None:
            request = AIRequest(prompt=prompt or "", system=system, messages=messages)

        current = request
        names_run: list[str] = []
        changes: list[str] = []
        result_meta: dict[str, Any] = {}
        optimizer_errors: list[str] = []
        timing: dict[str, float] = {}
        optimization_time_ms = 0.0
        for optimizer in self.optimizers:
            if not self._is_enabled(optimizer.name):
                continue
            if getattr(optimizer, "enabled", True) is False:
                logger.debug("optimizer %s disabled", optimizer.name)
                continue
            started = time.monotonic()
            try:
                outcome = optimizer.optimize(current, self.config)
            except OptimizationError as exc:
                optimizer_errors.append(f"{optimizer.name}: {exc}")
                logger.warning("optimizer %s skipped: %s", optimizer.name, exc)
                continue
            elapsed_ms = (time.monotonic() - started) * 1000.0
            timing[outcome.optimizer_name or optimizer.name] = round(elapsed_ms, 3)
            optimization_time_ms += elapsed_ms
            current = outcome.optimized_request
            names_run.append(outcome.optimizer_name)
            changes.extend(outcome.changes)
            if outcome.metadata:
                result_meta[outcome.optimizer_name] = outcome.metadata
            logger.debug(
                "optimizer=%s tokens=%d->%d time_ms=%.2f",
                outcome.optimizer_name,
                outcome.original_tokens,
                outcome.optimized_tokens,
                elapsed_ms,
            )

        original_token_count = count_tokens(request, self.tokenizer)
        optimized_token_count = count_tokens(current, self.tokenizer)

        # Quality gate with safe fallback.
        quality = self._run_quality_gate(request, current)
        result_meta["quality"] = quality
        accepted = True
        rejection_reason: str | None = None
        if quality.get("rejected"):
            logger.info(
                "quality gate rejected optimization "
                "(similarity=%.3f < %s); falling back to original request",
                quality.get("similarity", 0.0),
                self.config.quality_minimum_score,
            )
            current = request
            accepted = False
            rejection_reason = (
                f"quality similarity {quality.get('similarity', 0):.3f} below "
                f"minimum {self.config.quality_minimum_score}"
            )
            changes.append(
                "quality gate failed; reverted to original request "
                f"(similarity {quality.get('similarity', 0):.3f})"
            )

        if optimizer_errors:
            result_meta["optimizer_errors"] = optimizer_errors
        result_meta["optimizer_time_ms"] = round(optimization_time_ms, 3)
        if timing:
            result_meta["optimizer_time_by_step_ms"] = timing

        if self.config.log_prompts:
            log_prompt_content(
                logger, "original", request.prompt or "", enabled=True
            )
            log_prompt_content(
                logger, "optimized", current.prompt or "", enabled=True
            )

        return OptimizerResult(
            optimized_request=current,
            original_request=request,
            original_tokens=original_token_count,
            optimized_tokens=count_tokens(current, self.tokenizer),
            optimizer_name="+".join(names_run) or "none",
            optimizers_run=names_run,
            tokens_saved=original_token_count - optimized_token_count,
            reduction_percent=(
                ((original_token_count - optimized_token_count) / original_token_count)
                * 100.0
                if original_token_count
                else 0.0
            ),
            changes=changes,
            metrics=token_metrics(original_token_count, optimized_token_count),
            metadata=result_meta,
            optimization_time_ms=round(optimization_time_ms, 3),
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def _run_quality_gate(
        self, original: AIRequest, optimized: AIRequest
    ) -> dict[str, Any]:
        """Evaluate original vs optimized; returns a structured verdict."""
        if not self.config.quality_enabled:
            return {"enabled": False, "verified": False, "note": "quality not verified"}
        evaluator = self._quality_evaluator()
        if evaluator is None:
            return {
                "enabled": True,
                "verified": False,
                "note": "no evaluator configured; quality not verified",
            }
        original_text = self._request_text(original)
        optimized_text = self._request_text(optimized)
        similarity = evaluator(original_text, optimized_text)
        rejected = (
            self.config.quality_reject_on_failure
            and similarity < self.config.quality_minimum_score
        )
        return {
            "enabled": True,
            "verified": True,
            "similarity": round(similarity, 4),
            "minimum_score": self.config.quality_minimum_score,
            "rejected": rejected,
            "evaluator": (self.config.quality_evaluator or "character").lower(),
        }

    def _quality_evaluator(self) -> Any:
        """Resolve the configured evaluator to a callable.

        Raises :class:`QualityEvaluationError` when an unknown evaluator name
        is configured so misconfiguration fails loudly instead of silently
        skipping quality verification.
        """
        kind = (self.config.quality_evaluator or "character").lower()
        from opticore.evaluation.evaluator import (
            CharacterSimilarity,
            TokenJaccardSimilarity,
        )

        evaluators = {
            "character": CharacterSimilarity,
            "character_similarity": CharacterSimilarity,
            "token_jaccard": TokenJaccardSimilarity,
        }
        evaluator_cls = evaluators.get(kind)
        if evaluator_cls is None:
            raise QualityEvaluationError(
                "Unknown quality evaluator "
                f"{self.config.quality_evaluator!r}. Supported: "
                + ", ".join(sorted(evaluators))
            )
        return evaluator_cls().similarity

    @staticmethod
    def _request_text(request: AIRequest) -> str:
        parts = [request.prompt or ""]
        if request.system:
            parts.append(request.system)
        if request.messages:
            for msg in request.messages:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _is_enabled(self, name: str) -> bool:
        """Consult optimizer-specific enable flags on the config."""
        flag = {
            "token": "enable_token_optimization",
            "prompt": "enable_prompt_optimization",
            "context": "enable_context_optimization",
            "semantic_cache": "enable_semantic_cache",
            "model_router": "enable_model_routing",
            "cache": "enable_semantic_cache",
        }.get(name)
        if flag is None:
            return True
        return bool(getattr(self.config, flag, True))


def build_default_pipeline(
    config: OptimizationConfig | None = None,
) -> OptimizationPipeline:
    """Build a pipeline with all bundled optimizers enabled per config."""
    from opticore.optimizers.context import ContextOptimizer
    from opticore.optimizers.prompt import PromptOptimizer
    from opticore.optimizers.token import TokenOptimizer

    config = config or OptimizationConfig()
    tokenizer = Tokenizer()
    pipeline = OptimizationPipeline(
        config=config,
        tokenizer=tokenizer,
    )
    pipeline.add(TokenOptimizer(tokenizer=tokenizer))
    pipeline.add(PromptOptimizer(tokenizer=tokenizer))
    pipeline.add(ContextOptimizer(tokenizer=tokenizer))
    return pipeline

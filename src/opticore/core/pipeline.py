"""The optimization pipeline: runs optimizers in sequence over a request."""

from __future__ import annotations

import logging
from typing import Any

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import Optimizer, OptimizerRequest, OptimizerResult
from opticore.optimizers.token import Tokenizer, count_tokens

logger = logging.getLogger("opticore.pipeline")


class OptimizationPipeline:
    """Sequentially applies optimizers to an :class:`OptimizerRequest`.

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
        request: OptimizerRequest | None = None,
    ) -> OptimizerResult:
        """Execute all enabled optimizers against a request.

        Either pass the individual parts or a pre-built ``request``.
        """
        if request is None:
            request = OptimizerRequest(
                prompt=prompt or "",
                system=system,
                messages=messages,
            )

        current = request
        names_run: list[str] = []
        result_meta: dict[str, Any] = {}
        for optimizer in self.optimizers:
            if not self._is_enabled(optimizer.name):
                continue
            outcome = optimizer.optimize(current, self.config)
            current = outcome.optimized_request
            names_run.append(outcome.optimizer_name)
            if outcome.metadata:
                result_meta[outcome.optimizer_name] = outcome.metadata
            logger.debug(
                "optimizer=%s tokens=%d->%d",
                outcome.optimizer_name,
                outcome.original_tokens,
                outcome.optimized_tokens,
            )

        original_token_count = count_tokens(request, self.tokenizer)
        optimized_token_count = count_tokens(current, self.tokenizer)
        tokens_saved = original_token_count - optimized_token_count
        reduction = (
            (tokens_saved / original_token_count) * 100.0
            if original_token_count
            else 0.0
        )

        return OptimizerResult(
            optimized_request=current,
            original_request=request,
            original_tokens=original_token_count,
            optimized_tokens=optimized_token_count,
            optimizer_name="+".join(names_run) or "none",
            optimizers_run=names_run,
            tokens_saved=tokens_saved,
            reduction_percent=reduction,
            metadata=result_meta,
        )

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

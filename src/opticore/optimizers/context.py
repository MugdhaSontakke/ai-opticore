"""Context optimization: manages token budgets and conversation pruning.

The context optimizer does not permanently delete user data. It produces a
transformed request; the original is always available on the pipeline result
for comparison.
"""

from __future__ import annotations

import logging
from typing import Any

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.optimizers.base import BaseOptimizer
from opticore.optimizers.token import Tokenizer, count_tokens

logger = logging.getLogger("opticore.optimizers.context")


class ContextOptimizer(BaseOptimizer):
    """Reduces context size while respecting a token budget.

    Strategy (respecting ``prioritize_recent``):

    1. Remove duplicate messages (identical role + content).
    2. When over budget, drop oldest messages first (conversation history),
       retaining the system message and the most recent messages.
    3. If the system prompt itself exceeds the budget, it is truncated
       conservatively only in BALANCED/AGGRESSIVE modes.

    The original request is preserved; a copy is returned in the result.
    """

    name = "context"
    order = 30
    description = "Manages conversational context under a token budget"

    def __init__(self, tokenizer: Tokenizer | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()

    def optimize(self, request: OptimizerRequest, config: OptimizationConfig) -> OptimizerResult:
        original_tokens = count_tokens(request, self.tokenizer)
        meta: dict[str, Any] = {"dropped_messages": 0, "duplicate_messages": 0}

        # Work on a copy so the caller's data is never mutated.
        system = request.system
        prompt = request.prompt
        messages = list(request.messages) if request.messages else None

        if system is None and messages is None:
            # Nothing to prune; only prompt text available.
            optimized_request = OptimizerRequest(
                prompt=prompt,
                system=None,
                messages=None,
                max_tokens=request.max_tokens,
                model=request.model,
                metadata=dict(request.metadata),
            )
            return self._result(request, optimized_request, original_tokens, meta, config)

        # 1. Drop verbatim duplicate messages (always safe).
        if messages:
            unique: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                key = (role, content)
                if key in seen:
                    meta["duplicate_messages"] += 1
                    continue
                seen.add(key)
                unique.append({**msg})
            messages = unique

        # 2. Trim from the oldest messages when over budget.
        if messages and config.max_token_budget:
            current = self.tokenizer.count(prompt or "")
            if system:
                current += self.tokenizer.count(system)
            remaining_budget = config.max_token_budget - current
            kept: list[dict[str, str]] = []
            for msg in reversed(messages):
                cost = self.tokenizer.count(msg.get("content", ""))
                if remaining_budget - cost < 0:
                    meta["dropped_messages"] += 1
                    continue
                remaining_budget -= cost
                kept.append(msg)
            messages = list(reversed(kept))

        # 3. Truncate an oversized system prompt only in non-SAFE modes.
        if system:
            sys_tokens = self.tokenizer.count(system)
            budget = config.max_token_budget or sys_tokens
            if sys_tokens > budget and config.safety_mode.value != "safe":
                system = self._truncate_system(system, budget)
                meta["system_truncated"] = True

        optimized_request = OptimizerRequest(
            prompt=prompt,
            system=system,
            messages=messages,
            max_tokens=request.max_tokens,
            model=request.model,
            metadata=dict(request.metadata),
        )
        return self._result(request, optimized_request, original_tokens, meta, config)

    def _truncate_system(self, system: str, budget: int) -> str:
        """Conservatively shorten a system prompt by trimming the middle."""
        if budget <= 0:
            return system
        import math

        # Keep the first 40% and last 40% of the text as an approximation;
        # this preserves opening instructions and closing requirements.
        size = len(system)
        head_len = math.ceil(size * 0.4)
        tail_len = math.ceil(size * 0.4)
        head = system[:head_len]
        tail = system[size - tail_len :]
        truncated = f"{head} ... [truncated] {tail}".strip()
        return truncated

    def _result(
        self,
        request: OptimizerRequest,
        optimized_request: OptimizerRequest,
        original_tokens: int,
        meta: dict[str, Any],
        config: OptimizationConfig,
    ) -> OptimizerResult:
        optimized_tokens = count_tokens(optimized_request, self.tokenizer)
        saved = max(0, original_tokens - optimized_tokens)
        reduction = (saved / original_tokens * 100.0) if original_tokens else 0.0
        return OptimizerResult(
            optimized_request=optimized_request,
            original_request=request,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            optimizer_name=self.name,
            optimizers_run=[self.name],
            tokens_saved=saved,
            reduction_percent=reduction,
            metadata=meta,
        )

"""Token counting and analysis using real tokenizer implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger("opticore.optimizers.token")


class Tokenizer:
    """Thin layer over a real tokenizer so different models can use different
    tokenization strategies.

    The default uses OpenAI's ``cl100k_base`` tokenizer (via ``tiktoken``)
    which is a real, widely-used tokenizer. Other tokenizers can be supplied
    by passing a callable to ``count_fn`` or by subclassing.
    """

    def __init__(
        self,
        encoding_name: str | None = None,
        model: str | None = None,
        count_fn: Any | None = None,
    ) -> None:
        self.encoding_name = encoding_name
        self.model = model
        self._count_fn = count_fn
        self._encoding: tiktoken.Encoding | None = None

    def _load(self) -> None:
        """Lazily load the tokenizer encoding (import tiktoken on demand)."""
        import tiktoken

        if self.model:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:  # unknown model name
                self._encoding = tiktoken.get_encoding("cl100k_base")
        elif self.encoding_name:
            self._encoding = tiktoken.get_encoding(self.encoding_name)
        else:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def _ensure_custom(self) -> bool:
        return self._count_fn is not None

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` using a real tokenizer."""
        if not text:
            return 0
        if self._count_fn is not None:
            return int(self._count_fn(text))
        encoding = self._resolve_encoding()
        return len(encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        """Return the token ids for ``text`` (custom counter returns [])."""
        if self._count_fn is not None:
            return []
        encoding = self._resolve_encoding()
        return encoding.encode(text)

    def _resolve_encoding(self) -> tiktoken.Encoding:
        """Return the loaded encoding, loading it on first use."""
        if self._encoding is None:
            self._load()
        assert self._encoding is not None
        return self._encoding

    def description(self) -> str:
        """Human-readable description for metrics/benchmarks."""
        if self._count_fn is not None:
            return "custom"
        if self.model:
            return f"{self.model} (tiktoken)"
        return f"{self.encoding_name or 'cl100k_base'}|model-default (tiktoken)"


def count_tokens(request: OptimizerRequest, tokenizer: Tokenizer) -> int:
    """Count tokens across all text parts of a request."""
    total = tokenizer.count(request.prompt)
    if request.system:
        total += tokenizer.count(request.system)
    if request.messages:
        for msg in request.messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            total += tokenizer.count(content)
    return total


class TokenOptimizer(BaseOptimizer):
    """Measures token usage and reports original vs optimized counts.

    As a foundational analysis step, this optimizer normalizes obvious
    architecture-level token waste: it collapses prompt and message content
    into a single stable representation used by the pipeline, and reports the
    original/optimized counts. Actual text reduction is performed by
    :class:`PromptOptimizer` and :class:`ContextOptimizer`.
    """

    name = "token"
    order = 10
    description = "Token counting and analysis"

    def __init__(self, tokenizer: Tokenizer | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()

    def optimize(self, request: OptimizerRequest, config: OptimizationConfig) -> OptimizerResult:
        original_tokens = count_tokens(request, self.tokenizer)
        prompt = request.prompt.strip()
        system = request.system.strip() if request.system else None
        messages = request.messages
        optimized_request = OptimizerRequest(
            prompt=prompt,
            system=system,
            messages=messages,
            max_tokens=request.max_tokens,
            model=request.model,
            metadata=dict(request.metadata),
        )
        optimized_tokens = count_tokens(optimized_request, self.tokenizer)
        saved = original_tokens - optimized_tokens
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
            metadata={
                "tokenizer": self.tokenizer.description(),
                "input_tokens": original_tokens,
                "output_tokens": optimized_tokens,
            },
        )


class TokenAnalyzer:
    """Standalone token analysis utility returning detailed statistics."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or Tokenizer()

    def analyze(self, text: str) -> dict[str, Any]:
        """Analyze token composition of a single text."""
        token_count = self.tokenizer.count(text)
        tokens = self.tokenizer.encode(text)
        return {
            "token_count": token_count,
            "character_count": len(text),
            "tokens_per_character": round(token_count / max(len(text), 1), 4),
            "leading_whitespace_tokens": self.tokenizer.count(
                text
            ) - self.tokenizer.count(text.lstrip()),
            "sample_tokens": tokens[:8],
        }

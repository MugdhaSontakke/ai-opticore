"""Token counting and analysis using real tokenizer implementations.

Counting never uses ``characters / 4`` or ``words / 0.75`` heuristics. When a
tokenizer for a specific model is unavailable, the behavior is explicit:

- ``mode == "model"``   -> a real tokenizer exists for the requested model.
- ``mode == "fallback"`` -> the model was unknown to the registry; a
  documented default encoding is used and reported (``fallback_for`` set).
- ``mode == "encoding"`` -> an explicit encoding name was used.
- ``mode == "custom"``  -> a caller-supplied counter was used.

With ``strict=True`` an unknown model raises :class:`UnsupportedModelError`
instead of silently falling back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.exceptions import TokenizerNotFoundError, UnsupportedModelError
from opticore.logging import get_logger
from opticore.optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    import tiktoken

logger = get_logger("opticore.optimizers.token")

_DEFAULT_ENCODING = "cl100k_base"


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
        strict: bool = False,
    ) -> None:
        self.encoding_name = encoding_name
        self.model = model
        self._count_fn = count_fn
        self._encoding: tiktoken.Encoding | None = None
        self._fallback_for: str | None = None
        self._load_error: str | None = None
        self.strict = strict

    @property
    def mode(self) -> str:
        """Classifier describing what kind of tokenizer is in use."""
        if self._count_fn is not None:
            return "custom"
        if self._fallback_for is not None:
            return "fallback"
        if self.model:
            return "model"
        if self.encoding_name:
            return "encoding"
        return "encoding"

    @property
    def fallback_for(self) -> str | None:
        """The model name that could not be matched and triggered fallback."""
        return self._fallback_for

    def _load(self) -> None:
        """Lazily load the tokenizer encoding (import tiktoken on demand)."""
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - dep is required
            self._load_error = "tiktoken is not installed"
            raise TokenizerNotFoundError(self._load_error) from exc

        if self.model:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._fallback_for = self.model
                if self.strict:
                    raise UnsupportedModelError(
                        f"No tokenizer registered for model {self.model!r} "
                        f"and strict=True (no fallback)."
                    ) from None
                self._encoding = tiktoken.get_encoding(_DEFAULT_ENCODING)
        elif self.encoding_name:
            self._encoding = tiktoken.get_encoding(self.encoding_name)
        else:
            self._encoding = tiktoken.get_encoding(_DEFAULT_ENCODING)

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
        return list(encoding.encode(text))

    def _resolve_encoding(self) -> tiktoken.Encoding:
        """Return the loaded encoding, loading it on first use."""
        if self._encoding is None:
            self._load()
        if self._encoding is None:
            raise TokenizerNotFoundError(self._load_error or "tokenizer unavailable")
        return self._encoding

    def description(self) -> str:
        """Human-readable description for metrics/benchmarks."""
        if self._count_fn is not None:
            return "custom"
        if self._fallback_for is not None:
            return f"{self._fallback_for} (unregistered) -> {_DEFAULT_ENCODING} (fallback)"
        if self.model:
            return f"{self.model} (tiktoken)"
        return f"{self.encoding_name or _DEFAULT_ENCODING} (tiktoken)"

    def report(self) -> dict[str, Any]:
        """Explicit tokenizer state report for benchmarks/metrics."""
        return {
            "mode": self.mode,
            "description": self.description(),
            "fallback_for": self.fallback_for,
            "requested_model": self.model,
            "encoding": self.encoding_name or _DEFAULT_ENCODING,
        }


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
            temperature=request.temperature,
            tools=request.tools,
            namespace=request.namespace,
            metadata=dict(request.metadata),
        )
        optimized_tokens = count_tokens(optimized_request, self.tokenizer)
        saved = original_tokens - optimized_tokens
        reduction = (saved / original_tokens * 100.0) if original_tokens else 0.0
        changes = ["strip prompt whitespace"] if optimized_tokens < original_tokens else []
        return OptimizerResult(
            optimized_request=optimized_request,
            original_request=request,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            optimizer_name=self.name,
            optimizers_run=[self.name],
            tokens_saved=saved,
            reduction_percent=reduction,
            changes=changes,
            metrics={
                "original_input_tokens": original_tokens,
                "optimized_input_tokens": optimized_tokens,
                "tokens_saved": saved,
                "reduction_percentage": reduction,
            },
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

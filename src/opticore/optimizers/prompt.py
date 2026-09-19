"""Safe prompt optimization: removes redundancy without deleting meaning."""

from __future__ import annotations

import re
from typing import Any

from opticore.core.config import OptimizationConfig, SafetyMode
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.logging import get_logger
from opticore.optimizers.base import BaseOptimizer
from opticore.optimizers.token import Tokenizer, count_tokens

logger = get_logger("opticore.optimizers.prompt")


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces (harmless & lossless)."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_repeated_lines(text: str) -> str:
    """Remove consecutive duplicated lines block-wise.

    Only exact consecutive duplicates are removed, which is always safe.
    """
    seen: list[str] = []
    out: list[str] = []
    for line in text.splitlines():
        key = line.strip()
        if key and seen and seen[-1] == key:
            continue
        seen.append(key)
        out.append(line)
    return "\n".join(out)


def _dedupe_sentences(text: str) -> str:
    """Remove sentences that appear verbatim more than once.

    Only executed in BALANCED or AGGRESSIVE modes because collapsing non-fatal
    repetition could alter emphasis.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen: set[str] = set()
    kept: list[str] = []
    for sent in sentences:
        normalized = sent.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(sent)
    return " ".join(kept)


def _dedupe_duplicated_prose(text: str) -> str:
    """Remove large repeated blocks of text (paragraph-level duplicates)."""
    if len(text) < 80:
        return text
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen: set[str] = set()
    kept: list[str] = []
    for para in paragraphs:
        key = para.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(para)
    return "\n\n".join(kept)


class PromptOptimizer(BaseOptimizer):
    """Reduces token usage by removing safe, obvious redundancies.

    Behavior is driven by :class:`SafetyMode`:

    - SAFE: collapse whitespace, strip, and remove an explicit role-prefix
      hint like ``System:`` from the user prompt text when a ``system`` field
      is already provided (avoids duplication).
    - BALANCED: additionally remove verbatim duplicate sentences.
    - AGGRESSIVE: additionally remove verbatim repeated paragraphs and
      known boilerplate phrases.

    Content is never deleted based on word-count heuristics alone; only
    verbatim duplicates and structural whitespace are touched.
    """

    name = "prompt"
    order = 20
    description = "Removes safe redundant text from prompts"

    def __init__(self, tokenizer: Tokenizer | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()

    def optimize(self, request: OptimizerRequest, config: OptimizationConfig) -> OptimizerResult:
        original_tokens = count_tokens(request, self.tokenizer)
        meta: dict[str, Any] = {"transformations": []}

        prompt = request.prompt or ""
        mode = config.safety_mode

        # Perform structural deduplication FIRST while newlines are intact.
        cleaned = _remove_repeated_lines(prompt)
        meta["transformations"].append("remove_repeated_lines")

        if mode == SafetyMode.AGGRESSIVE:
            collapsed = _dedupe_duplicated_prose(cleaned)
            if collapsed != cleaned:
                meta["transformations"].append("dedupe_paragraphs")
            cleaned = collapsed

        if mode in (SafetyMode.BALANCED, SafetyMode.AGGRESSIVE):
            deduped = _dedupe_sentences(cleaned)
            if deduped != cleaned:
                meta["transformations"].append("dedupe_sentences")
            cleaned = deduped

        # Collapse whitespace AFTER structural dedup so paragraph/line
        # boundaries are available for dedup logic.
        collapsed = _collapse_whitespace(cleaned)
        if collapsed != cleaned:
            meta["transformations"].append("collapse_whitespace")
        cleaned = collapsed

        # If the prompt contains a role-prefix and a system field exists
        # do not strip it (keeps SAFE well-behaved); only report it.
        prefix_removed = False
        for prefix in ("System:", "system:", "USER:", "user:"):
            if cleaned.startswith(prefix) and request.system:
                cleaned = cleaned[len(prefix):].lstrip()
                prefix_removed = True
                break
        if prefix_removed:
            meta["transformations"].append("strip_role_prefix_hint")

        optimized_request = OptimizerRequest(
            prompt=cleaned,
            system=request.system,
            messages=request.messages,
            max_tokens=request.max_tokens,
            model=request.model,
            temperature=request.temperature,
            tools=request.tools,
            namespace=request.namespace,
            metadata=dict(request.metadata),
        )
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
            changes=meta["transformations"],
            metrics={
                "original_input_tokens": original_tokens,
                "optimized_input_tokens": optimized_tokens,
                "tokens_saved": saved,
                "reduction_percentage": reduction,
            },
            metadata={
                "mode": mode.value,
                "transformations": meta["transformations"],
                "severity": self._severity_hint(mode, saved, original_tokens),
            },
        )

    def _severity_hint(self, mode: SafetyMode, saved: int, total: int) -> str:
        """Flag levels where quality impact may be non-trivial."""
        if total and (saved / total) > 0.25:
            if mode == SafetyMode.AGGRESSIVE:
                return "high-impact: aggressive dedup may change emphasis"
            if mode == SafetyMode.BALANCED:
                return "verify: balanced dedup affecting >25% of tokens"
        return "low"

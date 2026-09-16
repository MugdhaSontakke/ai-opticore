"""Canonical, provider-independent data model for AI-OptiCore.

``AIRequest`` and ``AIResponse`` are the single request/response pair used
across the pipeline, cache, and providers. ``OptimizerRequest`` and
``ProviderResponse`` remain as aliases for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AIRequest:
    """A provider-independent request for a language model.

    Optimizers and providers operate on this shape; nothing in this object
    is owned by a specific vendor SDK.

    Fields marked *schema-preserving* (``tools``, ``metadata``) are passed
    through verbatim by every optimizer and must never be rewritten unless a
    dedicated, validated schema optimizer is added later.
    """

    prompt: str
    system: str | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int | None = None
    model: str | None = None
    temperature: float | None = None
    tools: list[dict[str, Any]] | None = None
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation (deep enough for JSON)."""
        return {
            "prompt": self.prompt,
            "system": self.system,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "temperature": self.temperature,
            "tools": self.tools,
            "namespace": self.namespace,
            "metadata": dict(self.metadata),
        }


@dataclass
class AIResponse:
    """A provider-independent model response.

    ``input_tokens``/``output_tokens`` are provider-reported where available
    and 0 otherwise; metrics code treats 0-with-unknown as unmeasured, not as
    a fact. ``total_tokens`` is computed on demand.
    """

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @property
    def total_tokens(self) -> int:
        """Combined input + output tokens (includes cached metadata if any)."""
        cached = self.metadata.get("cached_tokens", 0)
        return self.input_tokens + self.output_tokens + int(cached)


# Backward-compatible aliases. New code should use the canonical names.
OptimizerRequest = AIRequest
ProviderResponse = AIResponse

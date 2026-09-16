"""Semantic cache: exact + embedding-based similarity."

Uses optional embedding providers for semantic matching. When no embedding
provider is installed the cache falls back to exact-match only behavior.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opticore.cache.base import BaseCache, CacheEntry, MemoryCache

logger = logging.getLogger("opticore.cache.semantic")

EmbeddingFn = Callable[[str], list[float]]


@dataclass
class SemanticCacheEntry:
    """Internal entry pairing a stored response with its embedding."""

    text: str
    embedding: list[float]
    content: str
    model: str
    created_at: float
    expires_at: float | None = None
    metadata: dict[str, Any] | None = None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() > self.expires_at


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0


class SemanticCache(BaseCache):
    """A cache that answers exact and semantically-similar requests.

    Exact matches short-circuit. Semantic matches use cosine similarity over
    embeddings; the result is looked up in a name-stable store and only
    returned when similarity >= ``threshold``.

    If no ``embedding_fn`` is provided, semantic lookups degrade gracefully to
    exact-match behavior (miss on non-identical text).
    """

    name = "semantic"

    def __init__(
        self,
        embedding_fn: EmbeddingFn | None = None,
        threshold: float = 0.90,
        ttl_seconds: float | None = None,
        store: BaseCache | None = None,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self._store = store or MemoryCache()
        self._entries: list[SemanticCacheEntry] = []
        self._lock = threading.RLock()
        self.semantic_hits = 0
        self.exact_hits = 0
        self.misses = 0

    def get(self, key: str) -> CacheEntry | None:
        """Exact-match lookup by key (used when key is precomputed)."""
        return self._store.get(key)

    def lookup(
        self,
        text: str,
        model: str | None = None,
    ) -> tuple[bool, CacheEntry | None]:
        """Look up ``text``; returns (hit, entry)."""
        model = model or "unknown"
        exact_key = self._derive_key(text, model)
        exact = self._store.get(exact_key)
        if exact is not None:
            self.exact_hits += 1
            return True, exact

        if self.embedding_fn is None:
            self.misses += 1
            return False, None

        try:
            query_embedding = self.embedding_fn(text)
        except Exception:  # pragma: no cover - embedding provider failure
            logger.warning("embedding computation failed; treating as miss")
            self.misses += 1
            return False, None

        best: SemanticCacheEntry | None = None
        best_score = -1.0
        with self._lock:
            # Drop expired entries lazily.
            self._entries = [e for e in self._entries if not e.expired]
            for entry in self._entries:
                if model and entry.model != model:
                    continue
                score = _cosine(query_embedding, entry.embedding)
                if score >= self.threshold and score > best_score:
                    best = entry
                    best_score = score

        if best is not None:
            self.semantic_hits += 1
            return True, CacheEntry(
                key=self._derive_key(best.text, best.model),
                content=best.content,
                model=best.model,
                created_at=best.created_at,
                expires_at=best.expires_at,
                metadata=best.metadata or {},
            )
        self.misses += 1
        return False, None

    def set(
        self,
        key: str,
        content: str,
        model: str,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Exact-key storage, compatible with :class:`BaseCache`.

        The original text may be supplied via ``metadata["semantic_text"]`` so
        the entry can participate in semantic lookups; otherwise it is indexed
        for exact-match only.
        """
        text = (metadata or {}).get("semantic_text", "")
        model = model or "unknown"
        embedding: list[float] = []
        if text and self.embedding_fn is not None:
            try:
                embedding = self.embedding_fn(text)
            except Exception:
                logger.warning("embedding failed; entry stored exact-match only")
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        now = time.monotonic()
        expires = now + ttl if ttl is not None else None
        self._store.set(key, content, model, ttl_seconds=ttl, metadata=metadata or {})
        if text:
            with self._lock:
                self._entries.append(
                    SemanticCacheEntry(
                        text=text,
                        embedding=embedding,
                        content=content,
                        model=model,
                        created_at=now,
                        expires_at=expires,
                        metadata=metadata,
                    )
                )

    def put(
        self,
        text: str,
        content: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store content keyed by its text (exact key + embedding index)."""
        model = model or "unknown"
        meta = dict(metadata or {})
        meta["semantic_text"] = text
        self.set(
            key=self._derive_key(text, model),
            content=content,
            model=model,
            ttl_seconds=self.ttl_seconds,
            metadata=meta,
        )

    def delete(self, key: str) -> bool:
        deleted = self._store.delete(key)
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if self._derive_key(e.text, e.model) != key]
            deleted = deleted or len(self._entries) != before
        return deleted

    def clear(self) -> None:
        self._store.clear()
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        base = self._store.stats()
        base["semantic_hits"] = self.semantic_hits
        base["exact_hits"] = self.exact_hits
        base["total_hits"] = self.exact_hits + self.semantic_hits
        base["misses"] = self.misses
        div = self.exact_hits + self.semantic_hits + self.misses
        base["hit_rate"] = (self.exact_hits + self.semantic_hits) / div if div else 0.0
        base["type"] = self.name
        return base

    @staticmethod
    def _derive_key(text: str, model: str | None) -> str:
        blob = f"{text}|{model or ''}"
        return hashlib.sha256(blob.encode()).hexdigest()

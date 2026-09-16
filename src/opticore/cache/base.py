"""Cache abstraction: base interface and in-memory storage."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    key: str
    content: str
    model: str
    created_at: float
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() > self.expires_at


@dataclass
class CachePolicy:
    """Policies for serving cached entries.

    The default is conservative: stale entries are never served. Semantics
    that are clearly time-sensitive should either raise their TTL expectation
    or disable caching for specific requests (e.g. ``metadata["no_cache"]``).
    """

    serve_stale_for_seconds: float = 0.0
    """How long past TTL an entry may still be served (0 disables)."""

    def allows_stale(self, entry: CacheEntry) -> bool:
        """Return True when a stale entry may still be served under policy."""
        if not entry.expired or entry.expires_at is None:
            return False
        if self.serve_stale_for_seconds <= 0:
            return False
        return time.monotonic() <= entry.expires_at + self.serve_stale_for_seconds


class BaseCache(ABC):
    """Storage-agnostic cache interface."""

    name = "base"

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Return the entry for ``key`` or None (including expired entries)."""

    @abstractmethod
    def set(
        self,
        key: str,
        content: str,
        model: str,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a response under ``key``."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a key; returns True if it existed."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return cache statistics (hits, misses, size)."""

    def size(self) -> int:
        """Approximate number of stored entries."""
        return int(self.stats().get("size", 0))


class MemoryCache(BaseCache):
    """Thread-safe in-memory cache.

    An optional :class:`CachePolicy` controls stale-serving; by default stale
    entries are never served. When ``max_entries`` is reached the oldest entry
    (by ``created_at``) is evicted.
    """

    name = "memory"

    def __init__(
        self, max_entries: int = 10_000, policy: CachePolicy | None = None
    ) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._policy = policy or CachePolicy()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expired:
                if self._policy.allows_stale(entry):
                    self.hits += 1
                    return entry
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry

    def set(
        self,
        key: str,
        content: str,
        model: str,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if len(self._store) >= self._max_entries:
                # Evict the oldest entry.
                oldest = min(self._store.values(), key=lambda e: e.created_at)
                del self._store[oldest.key]
            now = time.monotonic()
            expires = now + ttl_seconds if ttl_seconds is not None else None
            self._store[key] = CacheEntry(
                key=key,
                content=content,
                model=model,
                created_at=now,
                expires_at=expires,
                metadata=metadata or {},
            )

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = sum(1 for e in self._store.values() if not e.expired)
            return {
                "type": self.name,
                "size": active,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses)
                if (self.hits + self.misses)
                else 0.0,
            }


def cache_key(
    *,
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    namespace: str | None = None,
) -> str:
    """Deterministic exact-match key isolating all cache-relevant fields.

    Temperature and max_tokens are included because two identical prompts
    served with different generation parameters may return substantially
    different outputs; caching across them would be incorrect.
    """
    import hashlib

    parts = [
        prompt,
        system or "",
        model or "",
        f"temp={temperature}" if temperature is not None else "temp=none",
        f"maxt={max_tokens}" if max_tokens is not None else "maxt=none",
        namespace or "",
    ]
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()


class CacheResponse:
    """Wrapper returned by the cache subsystem for a lookup."""

    def __init__(self, hit: bool, entry: CacheEntry | None = None) -> None:
        self.hit = hit
        self.entry = entry

    def content(self) -> str | None:
        return self.entry.content if self.entry else None

"""Cache backends: in-memory, disk (SQLite), and the semantic cache."""

from __future__ import annotations

from opticore.cache.base import (
    BaseCache,
    CacheEntry,
    CachePolicy,
    CacheResponse,
    MemoryCache,
    cache_key,
)
from opticore.cache.disk import DiskCache
from opticore.cache.semantic import SemanticCache

__all__ = [
    "BaseCache",
    "CacheEntry",
    "CachePolicy",
    "CacheResponse",
    "MemoryCache",
    "DiskCache",
    "SemanticCache",
    "cache_key",
]

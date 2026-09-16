"""Tests for the memory cache and semantic cache."""

from __future__ import annotations

import time

from opticore.cache.base import MemoryCache, cache_key


def test_memory_cache_exact_hit() -> None:
    cache = MemoryCache()
    cache.set("k1", "response", "m")
    entry = cache.get("k1")
    assert entry is not None
    assert entry.content == "response"
    assert cache.stats()["hits"] == 1


def test_memory_cache_miss() -> None:
    cache = MemoryCache()
    assert cache.get("missing") is None
    assert cache.stats()["misses"] == 1


def test_memory_cache_ttl_expiry() -> None:
    cache = MemoryCache()
    cache.set("k", "c", "m", ttl_seconds=0.01)
    time.sleep(0.02)
    assert cache.get("k") is None
    assert cache.size() == 0


def test_memory_cache_evicts_oldest() -> None:
    cache = MemoryCache(max_entries=2)
    cache.set("a", "x", "m")
    cache.set("b", "y", "m")
    cache.set("c", "z", "m")  # should evict 'a'
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_memory_cache_delete() -> None:
    cache = MemoryCache()
    cache.set("x", "v", "m")
    assert cache.delete("x") is True
    assert cache.get("x") is None
    assert cache.delete("x") is False


def test_cache_key_deterministic() -> None:
    k1 = cache_key(prompt="hello", model="m")
    k2 = cache_key(prompt="hello", model="m")
    assert k1 == k2


def test_cache_key_varies_by_input() -> None:
    assert cache_key(prompt="a") != cache_key(prompt="b")


def test_cache_stats_include_all_fields() -> None:
    stats = MemoryCache().stats()
    assert "type" in stats
    assert stats["type"] == "memory"
    assert "hit_rate" in stats


def test_cache_entry_expired_property() -> None:
    from opticore.cache.base import CacheEntry

    entry = CacheEntry(
        key="k", content="c", model="m",
        created_at=0.0, expires_at=0.0,
    )
    assert entry.expired is True
    entry.expires_at = time.monotonic() + 100
    assert entry.expired is False


def test_semantic_cache_exact_hit() -> None:
    from opticore.cache.semantic import SemanticCache

    cache = SemanticCache(embedding_fn=None, threshold=0.9)
    cache.put(text="hi", content="hello", model="m")
    hit, entry = cache.lookup("hi", model="m")
    assert hit is True
    assert entry.content == "hello"


def test_semantic_cache_miss_without_embedding_fn() -> None:
    from opticore.cache.semantic import SemanticCache

    cache = SemanticCache(embedding_fn=None, threshold=0.9)
    hit, entry = cache.lookup("unknown", model="m")
    assert hit is False
    assert entry is None


def test_semantic_cache_exact_vs_semantic_tracks_stats() -> None:
    from opticore.cache.semantic import SemanticCache

    def fake_embedding(text: str) -> list[float]:
        return [float(ord(c)) for c in text[:4]]

    cache = SemanticCache(embedding_fn=fake_embedding, threshold=0.0)
    cache.put(text="exact", content="c1", model="m")
    cache.put(text="other", content="c2", model="m")
    # Exact match should hit.
    hit, _ = cache.lookup("exact", model="m")
    assert hit is True
    stats = cache.stats()
    assert stats["exact_hits"] >= 1

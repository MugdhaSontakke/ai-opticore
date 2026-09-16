"""Tests for cache isolation, overhead timing, and cache policy."""

from __future__ import annotations

from opticore import AIClient, OptimizationConfig
from opticore.cache.base import CachePolicy, MemoryCache, cache_key
from opticore.cache.semantic import SemanticCache
from opticore.providers.base import FakeProviderMixin


class Fake(FakeProviderMixin):
    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, request: dict) -> object:
        self.calls += 1
        return super().generate(request)


def test_cache_key_isolates_system_and_temperature() -> None:
    base = cache_key(prompt="p", model="m")
    with_system = cache_key(prompt="p", model="m", system="s")
    with_temp = cache_key(prompt="p", model="m", temperature=0.7)
    with_namespace = cache_key(prompt="p", model="m", namespace="ns")
    assert base != with_system
    assert base != with_temp
    assert base != with_namespace
    assert cache_key(prompt="p", model="m", temperature=0.7) == cache_key(
        prompt="p", model="m", temperature=0.7
    )


def test_memory_cache_isolates_temperature() -> None:
    cache = MemoryCache()
    key_a = cache_key(prompt="prompt", model="m", temperature=0.0)
    key_b = cache_key(prompt="prompt", model="m", temperature=1.0)
    cache.set(key_a, "cold", "m")
    assert cache.get(key_a).content == "cold"
    assert cache.get(key_b) is None


def test_semantic_cache_isolates_system() -> None:
    cache = SemanticCache(embedding_fn=None, threshold=0.9)
    cache.put(text="hello", content="r1", model="m", system="sys-a")
    hit, entry = cache.lookup("hello", model="m", system="sys-a")
    assert hit is True
    hit_other, _ = cache.lookup("hello", model="m", system="sys-b")
    assert hit_other is False


def test_semantic_cache_isolates_temperature() -> None:
    cache = SemanticCache(embedding_fn=None, threshold=0.9)
    cache.put(text="hello", content="r1", model="m", temperature=0.0)
    hit, _ = cache.lookup("hello", model="m", temperature=0.5)
    assert hit is False


def test_aiclient_measures_overhead_and_model_time() -> None:
    provider = Fake()
    client = AIClient(provider=provider, config=OptimizationConfig())
    result = client.generate(prompt="hello there")
    assert result.total_time_ms >= 0
    assert result.optimizer_time_ms >= 0
    assert result.model_time_ms >= 0
    assert abs(
        result.total_time_ms - (result.optimizer_time_ms + result.model_time_ms)
    ) <= 0.01


def test_aiclient_temperature_isolation_no_cross_cache() -> None:
    provider = Fake()
    client = AIClient(provider=provider, config=OptimizationConfig())
    client.generate(prompt="same", temperature=0.0)
    result = client.generate(prompt="same", temperature=1.0)
    assert result.cache_hit is False
    assert provider.calls == 2


def test_cache_policy_does_not_serve_stale_by_default() -> None:
    import time

    from opticore.cache.base import CacheEntry

    entry = CacheEntry(
        key="k",
        content="c",
        model="m",
        created_at=0.0,
        expires_at=time.monotonic() - 10,
    )
    assert CachePolicy().allows_stale(entry) is False
    assert CachePolicy(serve_stale_for_seconds=60).allows_stale(entry) is True


def test_semantic_cache_ttl_expiry() -> None:
    import time

    cache = SemanticCache(embedding_fn=None, ttl_seconds=0.01)
    cache.put(text="hi", content="hello", model="m")
    time.sleep(0.03)
    hit, _ = cache.lookup("hi", model="m")
    assert hit is False

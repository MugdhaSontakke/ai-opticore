"""Disk cache backend tests + cache-failure-safe-fallback tests (PHASE 7/11)."""

from __future__ import annotations

import pytest

from opticore.api import AIClient
from opticore.cache import DiskCache
from opticore.cache.base import cache_key
from opticore.core.config import OptimizationConfig
from opticore.exceptions import CacheError
from opticore.providers.base import FakeProviderMixin


class Fake(FakeProviderMixin):
    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, request: dict) -> object:
        self.calls += 1
        return super().generate(request)


@pytest.fixture()
def disk_path(tmp_path: pytest.TempPathFactory) -> str:
    return str(tmp_path / "cache.sqlite")


def test_disk_cache_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    cache = DiskCache(str(tmp_path / "c.sqlite"))
    cache.set("k1", "hello", "model-a", ttl_seconds=60)
    entry = cache.get("k1")
    assert entry is not None
    assert entry.content == "hello"
    assert entry.model == "model-a"


def test_disk_cache_ttl_expiration(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    cache = DiskCache(str(tmp_path / "c.sqlite"))
    cache.set("stale", "old", "m", ttl_seconds=0.001)
    # Same-process caching of monotonic time; force expiry with a short sleep.
    time.sleep(0.005)
    assert cache.get("stale") is None
    assert cache.stats()["size"] == 0


def test_disk_cache_isolation_by_key(tmp_path: pytest.TempPathFactory) -> None:
    cache = DiskCache(str(tmp_path / "c.sqlite"))
    cache.set("k", "v1", "model-a", metadata={"namespace": "ns1"})
    cache.set("k2", "v2", "model-b", metadata={"namespace": "ns2"})
    assert cache.get("k").content == "v1"  # type: ignore[union-attr]
    assert cache.delete("k") is True
    assert cache.delete("k") is False
    assert cache.stats()["size"] >= 1


def test_disk_cache_clear_and_stats(tmp_path: pytest.TempPathFactory) -> None:
    cache = DiskCache(str(tmp_path / "c.sqlite"))
    cache.set("a", "A", "m")
    cache.set("b", "B", "m")
    assert cache.stats()["size"] == 2
    cache.clear()
    assert cache.stats()["size"] == 0


def test_disk_cache_persists_across_instances(tmp_path: pytest.TempPathFactory) -> None:
    path = str(tmp_path / "c.sqlite")
    DiskCache(path).set("persist", "kept", "m")
    cache2 = DiskCache(path)
    entry = cache2.get("persist")
    assert entry is not None
    assert entry.content == "kept"


def test_disk_cache_corrupt_db_raises_typed_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = str(tmp_path / "corrupt.sqlite")
    with open(path, "w") as fh:
        fh.write("this is not a sqlite database at all, definitely corrupted")
    with pytest.raises(CacheError):
        DiskCache(path)


def test_aiclient_disk_backend_hits_across_instances(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = str(tmp_path / "cache.sqlite")
    config = OptimizationConfig(cache_backend="disk", cache_disk_path=path)
    provider = Fake()
    client1 = AIClient(provider=provider, config=config)
    r1 = client1.generate(prompt="same prompt cached on disk")
    assert r1.cache_hit is False
    client2 = AIClient(provider=provider, config=config)
    r2 = client2.generate(prompt="same prompt cached on disk")
    assert r2.cache_hit is True
    assert provider.calls >= 1


def test_aiclient_survives_cache_lookup_failure(tmp_path: pytest.TempPathFactory) -> None:
    class BrokenCache(DiskCache):
        def get(self, key: str):  # type: ignore[override]
            raise CacheError("backend exploded")

        def set(self, *args, **kwargs):
            raise CacheError("backend exploded")

    cache = BrokenCache(str(tmp_path / "b.sqlite"))
    provider = Fake()
    client = AIClient(provider=provider, config=OptimizationConfig(), cache=cache)
    result = client.generate(prompt="should fall through to the model")
    assert result.cache_hit is False
    assert result.content  # served by the provider, not the broken cache
    assert client.metrics.counters.get("cache_read_errors", 0) >= 1


def test_aiclient_survives_cache_write_failure(tmp_path: pytest.TempPathFactory) -> None:
    class BrokenCache(DiskCache):
        def set(self, *args, **kwargs):
            raise CacheError("backend exploded")

    cache = BrokenCache(str(tmp_path / "w.sqlite"))
    provider = Fake()
    client = AIClient(provider=provider, config=OptimizationConfig(), cache=cache)
    result = client.generate(prompt="write should not crash")
    assert result.content
    assert client.metrics.counters.get("cache_write_errors", 0) >= 1


def test_disk_cache_key_is_namespace_isolated(tmp_path: pytest.TempPathFactory) -> None:
    cache = DiskCache(str(tmp_path / "c.sqlite"))
    k1 = cache_key(prompt="q", model="m", namespace="ns1")
    k2 = cache_key(prompt="q", model="m", namespace="ns2")
    assert k1 != k2
    cache.set(k1, "ns1 result", "m")
    assert cache.get(k2) is None

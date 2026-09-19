"""SQLite-backed disk cache implementing the same :class:`BaseCache` interface.

Optional backend for larger datasets or process restarts. Uses only the
standard library; SQLite is thread-safe with a per-connection guard. Corrupt
or unreadable databases raise :class:`CacheError` so callers can fall back to
the model (the ``AIClient`` does exactly that).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

from opticore.cache.base import BaseCache, CacheEntry
from opticore.exceptions import CacheError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key        TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);
"""


class DiskCache(BaseCache):
    """A file-backed cache with TTL support (entries expire on read).

    ``max_entries`` (when > 0) evicts the oldest entries on insert, keeping the
    store bounded. The cache is safe for multiple threads within one process.
    """

    name = "disk"

    def __init__(
        self,
        path: str,
        max_entries: int = 10_000,
    ) -> None:
        if not path:
            raise ValueError("DiskCache requires a path")
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        self.path = str(path)
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.path, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except sqlite3.Error as exc:
            raise CacheError(f"DiskCache cannot open {self.path}: {exc}") from exc

    def _init_db(self) -> None:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(_SCHEMA)
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                raise CacheError(f"DiskCache init failed: {exc}") from exc

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT key, content, model, created_at, expires_at, metadata "
                        "FROM cache_entries WHERE key = ?",
                        (key,),
                    ).fetchone()
                    if row is None:
                        self.misses += 1
                        return None
                    entry = self._row_to_entry(row)
                    if entry.expired:
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        self.misses += 1
                        return None
                    self.hits += 1
                    return entry
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                raise CacheError(f"DiskCache read failed: {exc}") from exc

    def set(
        self,
        key: str,
        content: str,
        model: str,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    now = time.monotonic()
                    expires = now + ttl_seconds if ttl_seconds is not None else None
                    conn.execute(
                        "INSERT INTO cache_entries "
                        "(key, content, model, created_at, expires_at, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET "
                        "content=excluded.content, model=excluded.model, "
                        "created_at=excluded.created_at, expires_at=excluded.expires_at, "
                        "metadata=excluded.metadata",
                        (
                            key,
                            content,
                            model,
                            now,
                            expires,
                            json.dumps(metadata or {}, sort_keys=True),
                        ),
                    )
                    if self._max_entries > 0:
                        conn.execute(
                            "DELETE FROM cache_entries WHERE key NOT IN ("
                            "SELECT key FROM cache_entries ORDER BY created_at DESC "
                            f"LIMIT {int(self._max_entries)}"
                            ")"
                        )
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                raise CacheError(f"DiskCache write failed: {exc}") from exc

    def delete(self, key: str) -> bool:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                raise CacheError(f"DiskCache delete failed: {exc}") from exc

    def clear(self) -> None:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                raise CacheError(f"DiskCache clear failed: {exc}") from exc

    def stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM cache_entries WHERE "
                        "(expires_at IS NULL OR expires_at > ?)",
                        (time.monotonic(),),
                    ).fetchone()
                    active = int(row[0] if row else 0)
                    return {
                        "type": self.name,
                        "path": self.path,
                        "size": active,
                        "hits": self.hits,
                        "misses": self.misses,
                        "hit_rate": self.hits / (self.hits + self.misses)
                        if (self.hits + self.misses)
                        else 0.0,
                    }
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                raise CacheError(f"DiskCache stats failed: {exc}") from exc

    @staticmethod
    def _row_to_entry(row: Any) -> CacheEntry:
        key, content, model, created_at, expires_at, metadata = row
        try:
            meta = json.loads(metadata) if metadata else {}
        except (TypeError, ValueError):
            meta = {}
        return CacheEntry(
            key=key,
            content=content,
            model=model,
            created_at=float(created_at),
            expires_at=float(expires_at) if expires_at is not None else None,
            metadata=meta,
        )

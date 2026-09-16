"""Runtime helpers: model loading interface and memory tracking."""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Any


class ModelLoader(ABC):
    """Loads models from an identifier into the active hardware backend.

    Status: experimental; provider-specific loaders should subclass.
    """

    @abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> Any:
        """Load a model; returns a provider-specific handle."""


class MemoryTracker:
    """Tracks approximate process memory for reporting.

    Uses the standard library ``resource`` when available; otherwise reports
    0 and marks the measurement as unavailable.
    """

    def __init__(self) -> None:
        self._peak_rss = 0.0
        self._samples: list[float] = []

    def sample(self) -> float:
        """Return current RSS in bytes (0 if unavailable)."""
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                rss *= 1024  # macOS reports kilobytes
            self._samples.append(float(rss))
            self._peak_rss = max(self._peak_rss, float(rss))
            return float(rss)
        except Exception:  # noqa: BLE001
            return 0.0

    def peak_bytes(self) -> float:
        return self._peak_rss

    def reporting(self) -> dict[str, Any]:
        return {
            "peak_rss_bytes": self.peak_bytes(),
            "measurement": "ru_maxrss (OS-level, best effort)",
            "samples_count": len(self._samples),
        }

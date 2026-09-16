"""Hardware detection facade: returns the best available backend."""

from __future__ import annotations

from typing import Any

from opticore.hardware.base import (
    BaseHardwareBackend,
    CPUBackend,
    CUDABackend,
    HardwareInfo,
    ROCmBackend,
    UnknownBackend,
)
from opticore.hardware.base import (
    require_backend as require_backend,
)

_BACKENDS: list[BaseHardwareBackend] = [
    ROCmBackend(),
    CUDABackend(),
    CPUBackend(),
]


def detect_backend(force: str | None = None) -> HardwareInfo:
    """Detect the available hardware backend.

    Order of preference: ROCm (AMD), CUDA (NVIDIA), CPU. ``force`` lets a
    caller override detection with a specific backend name.
    """
    if force:
        for backend in _BACKENDS:
            if backend.name == force:
                return backend.info()
        return UnknownBackend().info()

    for backend in _BACKENDS:
        if backend.available():
            return backend.info()
    return UnknownBackend().info()


def list_backends() -> list[HardwareInfo]:
    """Return capability reports for every registered backend."""
    return [b.info() for b in _BACKENDS]


def describe_detailed() -> dict[str, Any]:
    """Return a nested dict of backend details for the CLI/reports."""
    backends: dict[str, Any] = {}
    for backend in _BACKENDS:
        info = backend.info()
        backends[info.backend] = {
            "available": info.available,
            "status": getattr(backend, "status", "unknown"),
            "description": info.description,
            "details": backend.details(),
        }
    return {
        "detected": detect_backend().backend,
        "backends": backends,
    }

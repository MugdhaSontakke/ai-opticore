"""Hardware backend interface."""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from opticore.exceptions import HardwareBackendUnavailableError


def require_backend(backend: BaseHardwareBackend) -> HardwareInfo:
    """Return backend info or raise ``HardwareBackendUnavailableError``.

    Detection defaults to honest capability reporting (``available`` is
    never faked). This helper is for callers that need a hard failure when
    a backend is unavailable instead of a soft capability report.
    """
    info = backend.info()
    if not info.available:
        raise HardwareBackendUnavailableError(
            f"Hardware backend {backend.name!r} is unavailable: {info.description}"
        )
    return info


@dataclass
class HardwareInfo:
    """Normalized hardware capability description."""

    backend: str
    available: bool
    description: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class BaseHardwareBackend(ABC):
    """Base class for hardware backends.

    Backends report capabilities honestly. If a capability cannot be
    established, ``available`` must be False and ``details`` should say why.
    """

    name = "base"
    status: str = "planned"

    @abstractmethod
    def available(self) -> bool:
        """Return True only when this hardware was actually detected."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Report what the backend can do (never fake values)."""

    def info(self) -> HardwareInfo:
        return HardwareInfo(
            backend=self.name,
            available=self.available(),
            description=self.describe(),
            capabilities=self.capabilities(),
            details=self.details(),
        )

    def describe(self) -> str:
        """Human-readable description of the backend."""
        return self.name

    def details(self) -> dict[str, Any]:
        """Environment-specific detail keyed by backend."""
        return {}


class CPUBackend(BaseHardwareBackend):
    """CPU execution backend (always available)."""

    name = "cpu"
    status = "available"

    def __init__(self) -> None:
        self._cpu_count = None
        self._arch = platform.machine()

    def available(self) -> bool:
        return True

    def capabilities(self) -> dict[str, Any]:
        return {
            "device": "cpu",
            "supports_onnxruntime": False,
            "supports_cuda": False,
            "supports_rocm": False,
        }

    def details(self) -> dict[str, Any]:
        return {
            "architecture": self._arch,
            "python_implementation": platform.python_implementation(),
        }


class CUDABackend(BaseHardwareBackend):
    """CUDA execution backend, detected via PyTorch."""

    name = "cuda"
    status = "available"

    def available(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def capabilities(self) -> dict[str, Any]:
        return {"device": "cuda", "supports_cuda": True, "supports_rocm": False}

    def details(self) -> dict[str, Any]:
        try:
            import torch

            torch_available = torch.cuda.is_available()
            return {
                "torch_version": torch.__version__,
                "device_count": torch.cuda.device_count() if torch_available else 0,
                "device_name": torch.cuda.get_device_name(0) if torch_available else None,
            }
        except ImportError:
            return {"error": "torch not installed", "available": False}


class ROCmBackend(BaseHardwareBackend):
    """ROCm/AMD backend.

    Detection strategies (all must indicate an AMD GPU is reachable):

      1. ``torch.version.hip`` is set (ROCm PyTorch build) and a device is
         visible through ``torch.cuda`` (which ROCm maps to). This does NOT
         imply the hardware accelerates a particular model.
      2. ``rocm-smi`` binary presence is logged as a hint only.

    If none are met, ``available`` is False and capabilities explain why.
    """

    name = "rocm"
    status = "experimental"

    @property
    def non_result(self) -> str:
        return "ROCm not detected; AMD GPU unavailable on this machine"

    def available(self) -> bool:
        try:
            import torch

            hip = getattr(torch.version, "hip", None)
            if not hip:
                return False
            if torch.cuda.device_count() <= 0:
                return False
            name = torch.cuda.get_device_name(0)
            return (
                "amd" in name.lower()
                or "radeon" in name.lower()
                or "instinct" in name.lower()
            )
        except ImportError:
            return False
        except Exception:  # noqa: BLE001
            return False

    def capabilities(self) -> dict[str, Any]:
        return {
            "device": "rocm",
            "supports_rocm": True,
            "supports_cuda": False,
            "accelerated": self.available(),
        }

    def details(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "reason": self.non_result}
        import torch

        return {
            "torch_version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        }


class UnknownBackend(BaseHardwareBackend):
    """Fallback backend when nothing can be determined confidently."""

    name = "unknown"
    status = "unavailable"

    def available(self) -> bool:
        return False

    def capabilities(self) -> dict[str, Any]:
        return {"device": "unknown", "supports_cuda": False, "supports_rocm": False}

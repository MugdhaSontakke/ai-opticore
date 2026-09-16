"""Precision and quantization selection.

Status: planned — this module is a declarative interface. No automatic
quantization is performed; the active provider decides whether to honor a
requested precision level.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationConfig:
    """Declares a requested precision level in bits."""

    bits: int = 16

    def __post_init__(self) -> None:
        if self.bits not in (1, 2, 4, 8, 16, 32):
            raise ValueError("bits must be one of (1,2,4,8,16,32)")

    def description(self) -> str:
        return f"{self.bits}-bit precision (planned)"

    def status(self) -> str:
        return "planned"

"""Batching scheduler interfaces for inference optimization.

Status: experimental — correctness depends on provider batching support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BatchedRequest:
    """One item inside a batch."""

    request_id: str
    payload: dict[str, Any]


@dataclass
class BatchResult:
    """Result produced for one batch item."""

    request_id: str
    response: dict[str, Any]


_ITEM_ECHO: dict[str, Any] = {
    "content": "",
    "model": "",
    "provider": "",
    "input_tokens": 0,
    "output_tokens": 0,
}


class BatchScheduler(ABC):
    """Schedules individual requests into model-call batches."""

    name = "base"
    status = "planned"

    @abstractmethod
    def submit(self, payload: dict[str, Any]) -> str:
        """Queue a request; returns a request_id."""

    @abstractmethod
    def flush(self) -> list[BatchResult]:
        """Execute all queued requests and return results."""

    def stats(self) -> dict[str, Any]:
        return {"scheduler": self.name, "status": self.status}


class MemoryBatchScheduler(BatchScheduler):
    """A simple in-process batch scheduler that aggregates requests.

    It does not parallelize actual provider invocations; it provides the
    scheduling interface so batch-aware pipelines can be built. Marked
    experimental until a real parallel executor is added.
    """

    name = "memory"
    status = "experimental"

    def __init__(self, max_batch_size: int = 16) -> None:
        self.max_batch_size = max_batch_size
        self._pending: list[BatchedRequest] = []
        self._counter = 0

    def submit(self, payload: dict[str, Any]) -> str:
        self._counter += 1
        request_id = f"req-{self._counter}"
        self._pending.append(BatchedRequest(request_id=request_id, payload=payload))
        return request_id

    def flush(self) -> list[BatchResult]:
        drained = self._pending[: self.max_batch_size]
        self._pending = self._pending[len(drained) :]
        return [
            BatchResult(
                request_id=item.request_id,
                response={"request": item.payload, "echo": dict(_ITEM_ECHO)},
            )
            for item in drained
        ]

"""Deterministic retry policy with exponential backoff and jitter.

Design rules (production contract):
- Retries happen ONLY for transient, idempotent-safe failures: timeouts,
  connection failures, HTTP 408/429/5xx.
- Never retry operations that could duplicate a side effect. For pure
  generation requests this is safe; callers opt in by using these helpers.
- Auth/invalid-request/conflict/not-found statuses are non-retryable and are
  surfaced immediately.
- Backoff is exponential with a configurable jitter so thundering-herd
  retries do not stampede a recovering endpoint.
- Limits are explicit (``max_attempts``); there is no unbounded retrying.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from opticore.exceptions import ProviderError

# Transient / infra statuses that are safe to retry.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

# Statuses that must never be retried (would be unsafe or pointless).
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 405, 409, 410, 413, 422}

# Failure kinds that are safe to retry (transient, idempotent).
RETRYABLE_KINDS = frozenset({"timeout", "connection", "rate", "server"})

T = TypeVar("T")

FailureClassifier = Callable[[Exception], str]
ErrorFactory = Callable[[str, Exception], Exception]

SleepFn = Callable[[float], None]


@dataclass
class RetryPolicy:
    """Configuration for retrying transient provider failures.

    ``max_attempts`` is the total number of tries (1 disables retrying).
    ``base_delay_seconds`` is the sleep before the first retry; each
    subsequent retry doubles it up to ``max_delay_seconds``. ``jitter`` is a
    fraction (0..1) of the computed delay added as uniform noise.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be >= 0")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be in [0, 1]")

    def backoff_delay(self, failed_attempt: int) -> float:
        """Seconds to sleep before retrying after ``failed_attempt`` failures.

        Uses ``2 ** failed_attempt`` growth so attempt 0 -> base_delay,
        attempt 1 -> 2*b, attempt 2 -> 4*b, capped at max_delay.
        """
        exponent = max(0, int(failed_attempt))
        delay = min(self.base_delay_seconds * (2 ** exponent), self.max_delay_seconds)
        if self.jitter > 0 and delay > 0:
            delay += random.uniform(0, delay * self.jitter)
        return float(delay)

    def can_retry(self, failed_attempt: int) -> bool:
        """True when another attempt is allowed after ``failed_attempt`` fails."""
        return failed_attempt < self.max_attempts - 1

    def should_retry_status(self, status: int | None, failed_attempt: int) -> bool:
        """True when a status code is transient and budget remains."""
        if not self.can_retry(failed_attempt):
            return False
        if status is None:
            return False
        return status in RETRYABLE_STATUS


def sleep_with_backoff(policy: RetryPolicy, failed_attempt: int) -> float:
    """Sleep ``backoff_delay`` and return how long was slept (for tests)."""
    delay = policy.backoff_delay(failed_attempt)
    if delay > 0:
        time.sleep(delay)
    return delay


def retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    classify: FailureClassifier,
    finalize: ErrorFactory,
    sleep_fn: SleepFn = time.sleep,
) -> T:
    """Run ``fn`` with retries, mapping failures via ``classify``/``finalize``.

    ``classify(exc)`` returns one of the known failure kinds
    (``timeout``, ``connection``, ``rate``, ``server``, ``auth``,
    ``invalid``, ``response``, ``unknown``). Only transient kinds are
    retried. When the budget is exhausted (or the failure is permanent),
    ``finalize(kind, exc)`` builds the :class:`ProviderError` to raise.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except ProviderError:
            # Already a typed, final failure (e.g. ProviderResponseError raised
            # from inside the callable). Never retry or wrap it.
            raise
        except Exception as exc:  # noqa: BLE001 - providers raise arbitrary SDK/HTTP exceptions
            kind = classify(exc)
            if kind not in RETRYABLE_KINDS or not policy.can_retry(attempt):
                raise finalize(kind, exc) from exc
        delay = policy.backoff_delay(attempt)
        if delay > 0:
            sleep_fn(delay)
        attempt += 1

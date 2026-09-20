"""Tests for the shared retry policy and backoff helpers."""

from __future__ import annotations

import pytest

from opticore.exceptions import ProviderError
from opticore.providers.retry import (
    NON_RETRYABLE_STATUS,
    RETRYABLE_STATUS,
    RetryPolicy,
    retry_call,
)


class TestRetryPolicyValidation:
    def test_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_backoff_constraints(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(max_delay_seconds=0.2, base_delay_seconds=1.0)
        with pytest.raises(ValueError):
            RetryPolicy(jitter=1.5)
        with pytest.raises(ValueError):
            RetryPolicy(base_delay_seconds=-1)

    def test_defaults_are_sane(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.max_delay_seconds >= policy.base_delay_seconds


class TestBackoff:
    def test_backoff_grows_exponentially_within_cap(self) -> None:
        policy = RetryPolicy(base_delay_seconds=0.5, max_delay_seconds=8.0, jitter=0)
        assert policy.backoff_delay(0) == pytest.approx(0.5)
        assert policy.backoff_delay(1) == pytest.approx(1.0)
        assert policy.backoff_delay(2) == pytest.approx(2.0)
        assert policy.backoff_delay(4) == pytest.approx(8.0)

    def test_backoff_respects_budget(self) -> None:
        policy = RetryPolicy(max_attempts=3)
        assert policy.can_retry(0) is True
        assert policy.can_retry(1) is True
        assert policy.can_retry(2) is False


class TestStatusClassification:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    def test_retryable_statuses_with_budget(self, status: int) -> None:
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry_status(status, failed_attempt=0) is True

    @pytest.mark.parametrize("status", sorted(NON_RETRYABLE_STATUS))
    def test_non_retryable_statuses(self, status: int) -> None:
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry_status(status, failed_attempt=0) is False

    def test_no_budget_means_no_retry(self) -> None:
        policy = RetryPolicy(max_attempts=1)
        assert policy.should_retry_status(500, failed_attempt=0) is False


class _LastAttemptError(Exception):
    pass


class _TransientError(Exception):
    pass


class _PermanentError(Exception):
    pass


class TestRetryCall:
    def test_retries_transient_then_succeeds(self) -> None:
        calls = {"n": 0}
        sleep_times: list[float] = []

        def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise _TransientError("boom")
            return "ok"

        out = retry_call(
            fn,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.05, jitter=0),
            classify=lambda exc: "server",
            finalize=lambda kind, exc: ProviderError("unreachable"),
            sleep_fn=sleep_times.append,
        )
        assert out == "ok"
        assert calls["n"] == 3
        assert len(sleep_times) == 2

    def test_raises_finalized_error_when_budget_exhausted(self) -> None:
        def fn() -> None:
            raise _TransientError("always failing")

        with pytest.raises(ProviderError, match="exhausted"):
            retry_call(
                fn,
                policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.01, jitter=0),
                classify=lambda exc: "server",
                finalize=lambda kind, exc: ProviderError("retries exhausted"),
                sleep_fn=lambda _: None,
            )

    def test_permanent_failure_surfaces_immediately(self) -> None:
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1
            raise _PermanentError("nope")

        with pytest.raises(ProviderError, match="nope"):
            retry_call(
                fn,
                policy=RetryPolicy(max_attempts=3),
                classify=lambda exc: "auth",
                finalize=lambda kind, exc: ProviderError(str(exc)),
                sleep_fn=lambda _: None,
            )
        assert calls["n"] == 1

    def test_non_transient_classification_is_not_retried(self) -> None:
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1
            raise _LastAttemptError("bad request")

        with pytest.raises(ProviderError):
            retry_call(
                fn,
                policy=RetryPolicy(max_attempts=3),
                classify=lambda exc: "invalid",
                finalize=lambda kind, exc: ProviderError(str(exc)),
                sleep_fn=lambda _: None,
            )
        assert calls["n"] == 1

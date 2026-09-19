"""Provider regression tests against REAL endpoints (PHASE 3).

These tests hit external providers and therefore RUN ONLY when the operator
opt-in is set, so CI (which never sets it) skips them cleanly:

    OPTICORE_TEST_LIVE=1                    # enable live regression tests
    OPTICORE_TEST_OPENAI_API_KEY=sk-...     # optional: OpenAI-compatible
    OPTICORE_TEST_OPENAI_MODEL=gpt-4o-mini  # optional: model name
    OPTICORE_TEST_OLLAMA_URL=...            # optional: Ollama endpoint
    OPTICORE_TEST_OLLAMA_MODEL=llama3.2     # optional: Ollama model

Each test verifies a *behavior*, never a performance claim: response
availability, token measurement, latency recording, cache behavior, routing
decisions, and typed failure handling. No API keys are ever stored.
"""

from __future__ import annotations

import os

import pytest

from opticore import AIClient, OptimizationConfig
from opticore.cache import MemoryCache
from opticore.exceptions import ProviderAuthError
from opticore.providers import get_provider
from opticore.routing.router import ModelRouter, RouteRule

LIVE = os.environ.get("OPTICORE_TEST_LIVE", "0") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set OPTICORE_TEST_LIVE=1 to enable")

OPENAI_KEY = os.environ.get("OPTICORE_TEST_OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPTICORE_TEST_OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_URL = os.environ.get("OPTICORE_TEST_OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OPTICORE_TEST_OLLAMA_MODEL", "llama3.2")


def _online(url: str) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=3):  # noqa: S310 - test-only endpoint probe
            return True
    except Exception:
        return False


@live_only
@pytest.mark.parametrize("name,model,env_key", [
    pytest.param("openai", OPENAI_MODEL, "OPENAI_API_KEY", marks=pytest.mark.skipif(not OPENAI_KEY, reason="no OPENAI key")),
])
def test_openai_compatible_regression(name: str, model: str, env_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env_key, OPENAI_KEY or "")
    provider = get_provider(name, model=model)

    # Baseline: optimization disabled.
    baseline_client = AIClient(provider=provider, optimization=False)
    baseline = baseline_client.generate(
        prompt="What is machine learning? Answer in one sentence.",
        system="You are a helpful assistant.",
        tools=[{"type": "function", "function": {"name": "nop", "description": "no-op", "parameters": {"type": "object", "properties": {}}}}],
    )
    assert baseline.content
    assert baseline.provider == name
    # No optimization ran, so no reduction metrics are fabricated.
    assert baseline.original_tokens == 0

    # Optimized: same request through the pipeline.
    provider2 = get_provider(name, model=model)
    client = AIClient(provider=provider2, optimization=True)
    result = client.generate(
        prompt="What is machine learning? Answer in one sentence.",
        system="You are a helpful assistant.",
        tools=[{"type": "function", "function": {"name": "nop", "description": "no-op", "parameters": {"type": "object", "properties": {}}}}],
    )
    assert result.content
    assert result.original_tokens > 0
    assert result.model_time_ms >= 0
    assert result.request_id


@live_only
def test_live_provider_cache_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY or "")
    if not OPENAI_KEY and not OLLAMA_URL:
        pytest.skip("no live provider configured")
    provider = get_provider("openai", model=OPENAI_MODEL) if OPENAI_KEY else get_provider("ollama", base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    client = AIClient(
        provider=provider,
        config=OptimizationConfig(),
        cache=MemoryCache(),
    )
    r1 = client.generate(prompt="What is the capital of France?")
    r2 = client.generate(prompt="What is the capital of France?")
    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert r2.content == r1.content


@live_only
def test_live_provider_routing_decision_is_observable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY or "")
    if not OPENAI_KEY and not OLLAMA_URL:
        pytest.skip("no live provider configured")
    if OPENAI_KEY:
        provider = get_provider("openai", model=OPENAI_MODEL)
        available = [OPENAI_MODEL]
    else:
        provider = get_provider("ollama", base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        available = [OLLAMA_MODEL]

    router = ModelRouter(
        rules=[RouteRule(model="large", min_complexity=0.0)],
        default_model=available[0],
        fallback_models=available,
        available_models=available,
    )
    client = AIClient(provider=provider, config=OptimizationConfig(), router=router)
    result = client.generate(prompt="Please summarize the attached document in great detail, " * 5)
    assert result.metadata.get("routing") is not None
    assert result.metadata["routing"]["model"]
    if available:
        assert result.metadata["routing"]["model"] in available


@live_only
def test_live_provider_auth_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad key must surface a typed ProviderAuthError, not a raw crash."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key-for-regression")
    if not OPENAI_KEY:
        pytest.skip("no OPENAI key configured to compare against")
    from opticore.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key_env="OPENAI_API_KEY")
    client = AIClient(provider=provider, optimization=False)
    with pytest.raises(ProviderAuthError):
        client.generate(prompt="hello")


@live_only
def test_live_provider_ollama_regression() -> None:
    if not OLLAMA_URL:
        pytest.skip("OPTICORE_TEST_OLLAMA_URL not set")
    if not _online(f"{OLLAMA_URL}/api/tags"):
        pytest.skip("Ollama endpoint not reachable")
    provider = get_provider("ollama", base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    result = AIClient(provider=provider, optimization=True).generate(
        prompt="Reply with the single word: ok",
        system="Be extremely brief.",
    )
    assert result.content
    assert result.original_tokens > 0


def test_live_tests_skip_cleanly_outside_ci() -> None:
    """Without the opt-in flag the collection must not fail (this runs always)."""
    if not LIVE:
        # Behavior of the skip marker is enforced by pytest itself; this just
        # documents that missing credentials never break CI.
        assert True

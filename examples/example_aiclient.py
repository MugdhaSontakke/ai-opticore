"""Complete end-to-end example.

Application
  -> AI-OptiCore (optimization pipeline)
  -> cache lookup
  -> model routing
  -> provider
  -> response + per-request metrics

Runs with a deterministic fake provider so it works with zero configuration.
Point it at a real provider by setting OPTICORE_EXAMPLE_PROVIDER and the
matching environment variables; every code path stays identical.

    # local, no API needed:
    python examples/example_aiclient.py

    # real provider (needs OPENAI_API_KEY in the environment):
    OPTICORE_EXAMPLE_PROVIDER=openai python examples/example_aiclient.py
"""

from __future__ import annotations

import os

from opticore import AIClient, ModelRouter, OptimizationConfig
from opticore.providers.base import FakeProviderMixin
from opticore.routing.router import default_rules


class EchoFake(FakeProviderMixin):
    """Deterministic stand-in for a model: echoes input back."""

    name = "echo"


def build_client() -> AIClient:
    provider = os.environ.get("OPTICORE_EXAMPLE_PROVIDER", "echo")

    config = OptimizationConfig(
        # User-supplied example pricing so cost estimates are populated.
        # Remove these lines and estimated_cost reports "N/A" instead.
        pricing_input_per_1k=0.005,
        pricing_output_per_1k=0.015,
    )

    router = ModelRouter(
        rules=default_rules(),
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini", "gpt-4o"],
    )

    if provider == "echo":
        return AIClient(provider=EchoFake(), config=config, router=router)
    if provider in ("openai", "ollama", "huggingface"):
        return AIClient(
            provider=provider,
            config=config,
            router=router,
            model=os.environ.get("OPTICORE_EXAMPLE_MODEL") or None,
        )
    raise SystemExit(f"unknown provider for example: {provider!r}")


def main() -> None:
    client = build_client()

    # 1. A "real" request straight from an application.
    prompt = "Can you    please tell me what   is   machine learning?"

    first = client.generate(prompt=prompt, system="You are a helpful assistant.")
    second = client.generate(prompt=prompt, system="You are a helpful assistant.")  # cache hit

    print("== Response ==")
    print("content:        ", first.content)
    print("request_id:     ", first.request_id, "            (observed in logs too)")
    print("model used:     ", first.metadata.get("model") or getattr(client.provider, "name", "?"))
    print("routing:        ", first.metadata.get("routing"))
    print("cache hit #2:   ", second.cache_hit)

    print("\n== Token profile (request #1) ==")
    print(f"original tokens: {first.original_tokens}  (before optimization)")
    print(f"optimized input: {first.optimized_tokens}  (after optimization)")
    print(f"input tokens:    {first.input_tokens}")
    print(f"output tokens:   {first.output_tokens}")
    print(f"total tokens:    {first.input_tokens + first.output_tokens}")
    print(f"tokens saved:    {first.tokens_saved}")
    print(f"reduction:       {round(first.tokens_saved / first.original_tokens * 100, 1) if first.original_tokens else 0}%")

    print("\n== Timing (per stage, ms, this request) ==")
    print(f"optimizer:       {first.optimizer_time_ms}")
    print(f"cache_lookup:    {first.cache_lookup_ms}")
    print(f"provider:        {first.model_time_ms}")
    print(f"total:           {first.total_time_ms}")

    print("\n== Cost (estimated; only present because pricing is configured) ==")
    print("estimated_cost: ", first.estimated_cost)

    print("\n== Aggregate metrics (this client) ==")
    print(client.metrics.summary()["counters"])

    print("\nRun against a real provider:")
    print("  OPTICORE_EXAMPLE_PROVIDER=openai OPTICORE_EXAMPLE_MODEL=gpt-4o-mini \\")
    print("    python examples/example_aiclient.py")


if __name__ == "__main__":
    main()

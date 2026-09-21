#!/usr/bin/env python3
"""Real end-to-end demo: AI-OptiCore + Ollama (Llama 3.2).

Shows a real local LLM call through the AI-OptiCore pipeline:
- MODE A (BASELINE): direct provider call, no optimization, no cache reuse.
- MODE B (AI-OPTICORE): prompt optimized and cached; first request is a
  cache MISS (generation happens), the identical second request is a HIT
  (no generation, ~zero latency).
- Cost is reported N/A: a local Ollama endpoint has no per-token billing.
- Quality is a HEURISTIC similarity score - not a model-based judge.

Environment variables:
    OLLAMA_BASE_URL  default http://localhost:11434
    OLLAMA_MODEL     default llama3.2

Requires a running Ollama server with the model pulled:
    ollama serve
    ollama pull llama3.2

Exit codes: 0 = demo completed; 1 = missing dependency / server / model.
"""

from __future__ import annotations

import os
import sys

SAMPLE_PROMPTS: list[dict[str, str]] = [
    {
        "title": "Simple factual",
        "prompt": "What is the capital of France?",
        "system": "Answer in one sentence.",
    },
    {
        "title": "Verbose rewrite target",
        "prompt": (
            "Rewrite the following announcement to be clear and concise: "
            "Due to the fact that the meeting has been rescheduled, we would "
            "like to inform all attendees that it will now take place on "
            "Thursday at 10 AM in the main conference room instead of the "
            "originally scheduled time."
        ),
        "system": "Make announcements clear and concise.",
    },
    {
        "title": "Summarization",
        "prompt": (
            "Summarize in one sentence: Cloud computing provides on-demand "
            "compute and storage billed on usage. Providers operate global "
            "regions, replicate data for durability, and price by region, "
            "instance type, and storage class."
        ),
        "system": "Provide a concise one-sentence summary.",
    },
]


def _fail(message: str, hint: str) -> int:
    print(message, file=sys.stderr)
    print(f"suggestion: {hint}", file=sys.stderr)
    return 1


def main() -> int:
    try:
        from opticore import AIClient, OptimizationConfig
        from opticore.cache import MemoryCache
        from opticore.hardware import detect_backend
        from opticore.optimizers.token import Tokenizer
        from opticore.providers import get_provider
    except ImportError as exc:  # pragma: no cover - import guard
        return _fail(
            f"ai-opticore is not installed ({exc})",
            "pip install -e '.[ollama]'",
        )

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    try:
        provider = get_provider("ollama", base_url=base_url, model=model)
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(f"could not initialize Ollama provider: {exc}", "check OLLAMA_BASE_URL")

    print("AI-OptiCore - Real LLM demo with Ollama")
    print(f"Endpoint: {base_url}   Model: {model}")
    print()

    health = provider.health(model=model, probe=True)
    if not health.get("reachable"):
        return _fail(
            f"Ollama server is not reachable at {base_url} "
            f"({health.get('error_kind')}: {health.get('error')})",
            "start it with `ollama serve` (brew services start ollama)",
        )
    if not health.get("model_installed"):
        return _fail(
            f"model '{model}' is not installed",
            f"install it with `ollama pull {model}`",
        )
    probe = health.get("probe") or {}
    if not probe.get("ok"):
        return _fail(
            f"generation probe failed ({health.get('error_kind')})",
            "check the server logs; the model may be still loading",
        )
    print(
        f"Ollama: reachable (v{health.get('ollama_version')}) - "
        f"generation probe OK in {probe.get('elapsed_ms')} ms "
        f"({len(health.get('installed_models') or [])} model(s) installed)"
    )
    print(f"Hardware backend: {detect_backend().backend}")
    print()

    tokenizer = Tokenizer()
    baseline_client = AIClient(
        provider=provider, optimization=False, cache=MemoryCache()
    )
    optimized_client = AIClient(
        provider=provider,
        config=OptimizationConfig(),
        cache=MemoryCache(),
    )

    for i, sample in enumerate(SAMPLE_PROMPTS, start=1):
        prompt = sample["prompt"]
        system = sample["system"]
        print(f"[{i}/{len(SAMPLE_PROMPTS)}] {sample['title']}")
        print(f"  prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        baseline = baseline_client.generate(prompt=prompt, system=system)
        print("  MODE A - BASELINE (direct, no optimization, no cache reuse)")
        print(f"    response: {_preview(baseline.content)}")
        print(f"    latency: {baseline.model_time_ms:.1f} ms")
        print(f"    request_id: {baseline.request_id}")
        print()

        first = optimized_client.generate(prompt=prompt, system=system)
        second = optimized_client.generate(prompt=prompt, system=system)
        print("  MODE B - AI-OPTICORE (optimized prompt + cache, 2 requests)")
        original_tokens = tokenizer.count(prompt)
        reduction = first.reduction_percent
        cache_label = "HIT" if second.cache_hit else "MISS"
        print(f"    request 1: cache MISS - tokens {original_tokens} -> "
              f"{first.optimized_tokens} ({reduction}% reduction)")
        print(f"    request 1 latency: {first.model_time_ms:.1f} ms")
        print(f"    request 2: cache {cache_label} - latency "
              f"{second.model_time_ms:.1f} ms (HIT = no generation)")
        print(f"    response: {_preview(first.content)}")
        print()

    print("Cost: N/A - local Ollama endpoint has no per-token billing")
    print(
        "Quality: N/A here - the pipeline's quality gate is a HEURISTIC "
        "char/token similarity check, not a model-based judge."
    )
    print()
    print("To also benchmark a dataset:")
    print(
        "  ai-opticore benchmark --provider ollama --model llama3.2 "
        "--dataset benchmarks/data/real_llm_dataset.json --repeats 2"
    )
    print("  ai-opticore doctor --provider ollama --model llama3.2")
    return 0


def _preview(text: str, limit: int = 100) -> str:
    if not text:
        return "(empty response)"
    single = " ".join(text.split())
    if len(single) <= limit:
        return single
    return f"{single[:limit]}..."


if __name__ == "__main__":
    sys.exit(main())

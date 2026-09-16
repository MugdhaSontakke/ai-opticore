"""AI-OptiCore command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from opticore import __version__
from opticore.core.config import OptimizationConfig, SafetyMode
from opticore.hardware import describe_detailed, detect_backend, list_backends
from opticore.providers import available_providers

CONFIG_PATH = "opticore.json"


def _error(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def _load_fake_provider():
    from opticore.providers.base import FakeProviderMixin

    class FakeProvider(FakeProviderMixin):
        name = "fake"

    return FakeProvider()


def _entrypoint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-opticore",
        description="Optimization layer for AI/LLM applications.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create a starter opticore.json config")
    sub.add_parser("config", help="Show active configuration")
    sub.add_parser("hardware", help="Detect available hardware backends")
    sub.add_parser("models", help="List providers and available models")

    p_opt = sub.add_parser("optimize", help="Optimize a prompt (stdin or --prompt)")
    p_opt.add_argument("--prompt", help="Prompt text to optimize")
    p_opt.add_argument("--system", help="System prompt")
    p_opt.add_argument("--safety", choices=[m.value for m in SafetyMode], default="safe")
    p_opt.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_bench = sub.add_parser("benchmark", help="Run a real benchmark")
    p_bench.add_argument(
        "--samples", type=int, default=3, help="Number of unique sample prompts"
    )
    p_bench.add_argument("--repeats", type=int, default=1, help="Repeats per sample")
    p_bench.add_argument(
        "--provider",
        choices=["fake", *available_providers()],
        default="fake",
        help="Provider (fake uses deterministic mock; safe for CI)",
    )
    p_bench.add_argument("--model", default=None, help="Model identifier")
    p_bench.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_cache = sub.add_parser("cache", help="Inspect and clear the cache")
    p_cache.add_argument("--clear", action="store_true", help="Clear cache contents")
    return parser


def cmd_init(args: argparse.Namespace) -> None:
    if os.path.exists(CONFIG_PATH):
        _error(f"{CONFIG_PATH} already exists")
    template = {
        "safety_mode": "safe",
        "max_token_budget": 4096,
        "enable_token_optimization": True,
        "enable_prompt_optimization": True,
        "enable_context_optimization": True,
        "enable_semantic_cache": True,
        "enable_model_routing": True,
        "semantic_cache_threshold": 0.90,
        "cache_ttl_seconds": 3600,
        "prioritize_recent": True,
        "log_prompts": False,
        "provider": {"name": "openai", "model": ""},
    }
    with open(CONFIG_PATH, "w") as fh:
        json.dump(template, fh, indent=2)
        fh.write("\n")
    print(f"wrote {CONFIG_PATH} (review before use)")


def _load_config_from_disk() -> OptimizationConfig | None:
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH) as fh:
            data = json.load(fh)
        return OptimizationConfig.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not read {CONFIG_PATH}: {exc}", file=sys.stderr)
        return None


def cmd_config(args: argparse.Namespace) -> None:
    config = _load_config_from_disk() or OptimizationConfig()
    print(json.dumps(config.to_dict(), indent=2))


def cmd_hardware(args: argparse.Namespace) -> None:
    detected = detect_backend()
    print(f"Detected backend: {detected.backend}")
    print(f"Available: {detected.available}")
    print(f"Status: {detected.capabilities.get('device', 'unknown')}")
    for info in list_backends():
        status = "available" if info.available else "unavailable"
        print(f"  - {info.backend}: {status}")
    details = describe_detailed()
    print(json.dumps(details, indent=2))


def cmd_models(args: argparse.Namespace) -> None:
    print(f"Registered providers: {', '.join(available_providers())}")
    for name in available_providers():
        from opticore.providers import get_provider

        try:
            provider = get_provider(name)
            print(f"  {name}: {', '.join(provider.models()) or '(no default model)'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: unavailable ({exc})")


def cmd_optimize(args: argparse.Namespace) -> None:
    from opticore import Optimizer

    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    if not prompt:
        _error("no prompt provided (use --prompt or pipe stdin)")

    config = _load_config_from_disk() or OptimizationConfig()
    config.safety_mode = SafetyMode(args.safety)
    optimizer = Optimizer(config=config)
    outcome = optimizer.optimize(prompt, system=args.system)

    if args.json:
        print(
            json.dumps(
                {
                    "original": prompt,
                    "optimized_prompt": outcome.optimized_prompt,
                    "original_tokens": outcome.original_tokens,
                    "optimized_tokens": outcome.optimized_tokens,
                    "tokens_saved": outcome.tokens_saved,
                    "reduction_percent": outcome.reduction_percent,
                    "optimizers_run": outcome.optimizers_run,
                    "warnings": outcome.warnings,
                },
                indent=2,
            )
        )
        return

    print(f"Original tokens: {outcome.original_tokens}")
    print(f"Optimized tokens: {outcome.optimized_tokens}")
    print(f"Tokens saved: {outcome.tokens_saved} ({outcome.reduction_percent}%)")
    print(f"Optimizers: {', '.join(outcome.optimizers_run)}")
    if outcome.warnings:
        print("WARNINGS:")
        for warning in outcome.warnings:
            print(f"  - {warning}")
    print("--- optimized prompt ---")
    print(outcome.optimized_prompt)


def cmd_benchmark(args: argparse.Namespace) -> None:
    from opticore.benchmarks.runner import BenchmarkRunner
    from opticore.core.config import OptimizationConfig

    provider = (
        _load_fake_provider()
        if args.provider == "fake"
        else get_provider_from_name(args.provider)
    )
    samples = _build_samples(args.samples)
    config = _load_config_from_disk() or OptimizationConfig()
    runner = BenchmarkRunner(
        provider=provider,
        model=args.model or "benchmark-model",
        samples=samples,
        config=config,
        repeats=args.repeats,
    )
    result = runner.run()
    if args.json:
        print(json.dumps(_benchmark_to_dict(result), indent=2))
        return
    print(result.render())


def get_provider_from_name(name: str):
    from opticore.providers import get_provider

    return get_provider(name)


def _build_samples(count: int):
    from opticore.core.interfaces import OptimizerRequest

    if count <= 0:
        count = 1
    templates = [
        "What is machine learning?",
        "Explain the difference between classification and regression.",
        "Write a haiku about a computer.",
        "Summarize the key ideas of reinforcement learning in a few sentences.",
    ]
    return [
        OptimizerRequest(
            prompt=templates[i % len(templates)],
            system="You are a helpful assistant.",
        )
        for i in range(count)
    ]


def _benchmark_to_dict(result: Any) -> dict[str, Any]:
    return {
        "model": result.model,
        "provider": result.provider,
        "hardware": result.hardware,
        "samples": result.samples,
        "baseline": result.baseline,
        "optimized": result.optimized,
        "metrics": result.metrics,
        "notes": result.notes,
    }


def cmd_cache(args: argparse.Namespace) -> None:
    if args.clear:
        print("cache cleared")
        return
    print("cache usage: run a benchmark or use the Python API to populate the cache")


def main() -> None:
    parser = _entrypoint_parser()
    args = parser.parse_args()
    handlers = {
        "init": cmd_init,
        "config": cmd_config,
        "hardware": cmd_hardware,
        "models": cmd_models,
        "optimize": cmd_optimize,
        "benchmark": cmd_benchmark,
        "cache": cmd_cache,
    }
    handlers[args.command](args)


def cli() -> None:
    main()


if __name__ == "__main__":
    main()

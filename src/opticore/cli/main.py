"""AI-OptiCore command-line interface.

Every command:
- has ``--help``
- returns useful errors (exit code 1 on operational failure, 2 on CLI misuse)
- never dumps sensitive data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from opticore import __version__
from opticore.core.config import OptimizationConfig, SafetyMode, load_config
from opticore.exceptions import OptiCoreError
from opticore.hardware import describe_detailed, detect_backend, list_backends
from opticore.providers import available_providers

CONFIG_PATH = "opticore.yaml"


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

    sub.add_parser("init", help="Create a starter opticore.yaml config")

    p_config = sub.add_parser("config", help="Show or validate the active configuration")
    p_config.add_argument(
        "action",
        nargs="?",
        choices=["show", "validate"],
        default="show",
        help="show=print resolved config; validate=check a config file for errors",
    )
    p_config.add_argument("--config", default=None, help="Path to config file")
    p_config.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    sub.add_parser("hardware", help="Detect available hardware backends")
    sub.add_parser("models", help="List providers and available models")

    p_opt = sub.add_parser("optimize", help="Optimize a prompt (stdin or --prompt)")
    p_opt.add_argument("--prompt", help="Prompt text to optimize")
    p_opt.add_argument("--system", help="System prompt")
    p_opt.add_argument(
        "--safety", choices=[m.value for m in SafetyMode], default=None,
        help="Override safety mode (default: from config / safe)",
    )
    p_opt.add_argument("--config", default=None, help="Path to optics config file")
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
    p_bench.add_argument("--config", default=None, help="Path to optics config file")
    p_bench.add_argument("--dataset", default=None, help="Path to a JSON benchmark dataset")
    p_bench.add_argument("--output", default=None, help="Write JSON report to a file")
    p_bench.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_cache = sub.add_parser("cache", help="Explain cache behavior or show live stats")
    p_cache.add_argument(
        "action",
        nargs="?",
        choices=["explain", "stats"],
        default="explain",
        help="explain=how caching is configured; stats=live backend statistics",
    )
    p_cache.add_argument("--config", default=None, help="Path to optics config file")
    p_cache.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def cmd_init(args: argparse.Namespace) -> None:
    from opticore.core.config import config_template

    if os.path.exists(CONFIG_PATH):
        _error(f"{CONFIG_PATH} already exists")
    template = {
        **config_template(),
        "provider": {"name": "openai", "model": ""},
    }

    def _dump(obj: Any, indent: int = 0) -> str:
        lines: list[str] = []
        for key, value in obj.items():
            prefix = " " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump(value, indent + 2))
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    with open(CONFIG_PATH, "w") as fh:
        fh.write("# AI-OptiCore configuration. Never put API keys in here.\n")
        fh.write("# Keep secrets in environment variables (e.g. OPENAI_API_KEY).\n")
        fh.write(_dump(template))
        fh.write("\n")
    print(f"wrote {CONFIG_PATH} (review before use)")


def _load(config_path: str | None) -> OptimizationConfig:
    """Load config, defaulting to built-in defaults when no file is used.

    An explicit ``--config`` path must load successfully (errors surface).
    The implicit ``opticore.yaml`` is only used when it exists and is valid;
    if it exists but is broken we fail loudly instead of silently guessing.
    """
    path = config_path or (CONFIG_PATH if os.path.exists(CONFIG_PATH) else None)
    try:
        return load_config(path)
    except OptiCoreError as exc:
        if config_path:
            _error(str(exc))
        if path:
            _error(f"default config {CONFIG_PATH} is invalid: {exc}")
        return OptimizationConfig()


def cmd_config(args: argparse.Namespace) -> None:
    if args.action == "validate":
        _cmd_config_validate(args)
        return
    config = _load(getattr(args, "config", None))
    print(json.dumps(config.to_dict(), indent=2))


def _cmd_config_validate(args: argparse.Namespace) -> None:
    """Validate a config file strictly; exit non-zero when it is broken."""
    path = args.config
    if path is None and os.path.exists(CONFIG_PATH):
        path = CONFIG_PATH
    report: dict[str, Any] = {"status": "ok", "path": path, "errors": []}
    exit_code = 0
    if path is None:
        report["note"] = "no config file found; validating built-in defaults"
    else:
        try:
            config = load_config(path)  # raises ConfigurationError on any problem
            report["config"] = config.to_dict()
        except OptiCoreError as exc:
            exit_code = 1
            report["status"] = "invalid"
            report["errors"] = [str(exc)]
    if args.json:
        print(json.dumps(report, indent=2))
    elif exit_code == 0:
        src = path or "built-in defaults"
        print(f"config valid: {src}")
        print(json.dumps(report.get("config", {}), indent=2))
    else:
        print(f"error: config invalid: {report['errors'][0]}", file=sys.stderr)
    sys.exit(exit_code)


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
        except OptiCoreError as exc:
            print(f"  {name}: unavailable ({exc})")


def cmd_optimize(args: argparse.Namespace) -> None:
    from opticore import Optimizer

    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    if not prompt:
        _error("no prompt provided (use --prompt or pipe stdin)")

    config = _load(args.config)
    if args.safety:
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
                    "changes": outcome.changes,
                    "quality_verified": outcome.quality_verified,
                    "optimization_time_ms": outcome.optimization_time_ms,
                    "accepted": outcome.accepted,
                    "rejection_reason": outcome.rejection_reason,
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
    print(f"Optimization time: {outcome.optimization_time_ms} ms")
    if not outcome.accepted:
        print(f"REJECTED: {outcome.rejection_reason or 'quality gate failed'} (used original prompt)")
    if outcome.changes:
        print("Changes:")
        for change in outcome.changes:
            print(f"  - {change}")
    if outcome.warnings:
        print("WARNINGS:")
        for warning in outcome.warnings:
            print(f"  - {warning}")
    print("--- optimized prompt ---")
    print(outcome.optimized_prompt)


def cmd_benchmark(args: argparse.Namespace) -> None:
    from opticore.benchmarks.runner import BenchmarkRunner, load_benchmark_dataset

    provider = (
        _load_fake_provider()
        if args.provider == "fake"
        else get_provider_from_name(args.provider)
    )
    if args.dataset:
        dataset_name = os.path.basename(args.dataset)
        samples = load_benchmark_dataset(args.dataset)
    else:
        dataset_name = "builtin"
        samples = _build_samples(args.samples)
    config = _load(args.config)
    runner = BenchmarkRunner(
        provider=provider,
        model=args.model or "benchmark-model",
        samples=samples,
        config=config,
        repeats=args.repeats,
        provider_notes=(
            ["fake provider: token savings are real, latency is not representative"]
            if args.provider == "fake"
            else []
        ),
        dataset=dataset_name,
    )
    result = runner.run()
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
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


def cmd_cache(args: argparse.Namespace) -> None:
    if args.action == "stats":
        _cmd_cache_stats(args)
        return
    config = _load(getattr(args, "config", None))
    print("Cache is in-memory by default and process-local.")
    if config.enable_semantic_cache:
        print(f"Semantic cache: enabled (threshold={config.semantic_cache_threshold})")
    else:
        print("Semantic cache: disabled")
    print(f"TTL: {config.cache_ttl_seconds}s")
    print(
        "Use the Python API (AIClient) to populate and inspect the cache, or "
        "run `ai-opticore cache stats`."
    )


def _cmd_cache_stats(args: argparse.Namespace) -> None:
    """Report live statistics for the cache backend the config selects."""
    import os as _os

    from opticore.cache import DiskCache, MemoryCache

    config = _load(getattr(args, "config", None))
    report: dict[str, Any] = {"config": {"cache_backend": config.cache_backend}}
    base_stats: dict[str, Any]
    if config.cache_backend == "disk":
        cache = DiskCache(
            path=config.cache_disk_path
            or _os.path.join(_os.getcwd(), ".opticore", "cache.sqlite")
        )
        base_stats = cache.stats()
    else:
        base_stats = MemoryCache().stats()
    report["base"] = base_stats

    if config.enable_semantic_cache:
        report["semantic"] = {
            "enabled": True,
            "threshold": config.semantic_cache_threshold,
            "note": "semantic cache requires an embedding_fn at runtime "
            "(AIClient); no live statistics available in CLI",
        }
    else:
        report["semantic"] = {"enabled": False}

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Backend: {base_stats['type']}")
    print(f"Size (live entries): {base_stats['size']}")
    print(f"Hits: {base_stats['hits']}  Misses: {base_stats['misses']}")
    print(f"Hit rate: {base_stats['hit_rate']:.3f}")
    if "evictions" in base_stats:
        print(f"Evictions: {base_stats['evictions']}  Invalidations: {base_stats['invalidations']}")
    if config.enable_semantic_cache:
        print(
            f"Semantic cache: enabled "
            f"(threshold={config.semantic_cache_threshold})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _entrypoint_parser()
    args = parser.parse_args(argv)
    handlers = {
        "init": cmd_init,
        "config": cmd_config,
        "hardware": cmd_hardware,
        "models": cmd_models,
        "optimize": cmd_optimize,
        "benchmark": cmd_benchmark,
        "cache": cmd_cache,
    }
    try:
        handlers[args.command](args)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except OptiCoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())

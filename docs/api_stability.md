# API Stability Classification

This document classifies every public symbol in `opticore` so that consumers
know what they may rely on and what may change.

The list mirrors `opticore/__init__.py`. Anything not listed here is
**internal** and can change at any time.

## Stable

Backward compatible. Changes require a deprecated period and a minor/major
version bump.

| Symbol | Notes |
| --- | --- |
| `Optimizer` | `optimize()` contract unchanged since v0.1. |
| `AIClient` | `generate()` / `OptimizeOutcome` contract. |
| `OptimizationConfig` | New fields added in v0.2 are additive and optional; existing fields unchanged. |
| `SafetyMode` | Enum values stable. |
| `AIRequest`, `AIResponse` | Canonical request/response model. |
| `OptimizerRequest`, `OptimizerResult` | Interfaces used by optimizers and the pipeline. |
| `GenerationResult` | Result of `AIClient.generate()`. |
| Exceptions | The `opticore.exceptions` hierarchy: `OptiCoreError`, `ConfigurationError`, `OptimizationError`, `EvaluationError`, `QualityEvaluationError`, `CacheError`, `ProviderError` and subclasses (`ProviderAuthError`, `ProviderRateError`, `ProviderTimeoutError`, `ProviderUnavailableError`, `ProviderResponseError`), `RoutingError`, `UnsupportedModelError`, `TokenizerNotFoundError`, `HardwareBackendUnavailableError`. Subclasses are themselves stable; the set of subclasses may grow. |
| `ModelRouter`, `RouteRule`, `RoutingDecision` | Routing public surface. |

Guarantees:

- `AIClient.generate()` never raises a raw SDK/`requests` exception; concurrency
  and transport failures surface as typed `ProviderError` subclasses.
- Result and request types are JSON-serializable via `to_dict()`.

## Experimental

May change without notice within the 0.x series. Consume at your own risk and
pin your dependency.

| Symbol | Notes |
| --- | --- |
| `OptimizationPipeline` | Registered optimizers, ordering, and custom stages are in flux. |
| `BaseCache`, `MemoryCache`, `DiskCache`, `SemanticCache`, `cache_key` | Disk key format and TTL semantics may change as backends evolve (Redis planned). |
| `CostEstimator`, `CostBreakdown` | Pricing model may gain tiers/caching. |
| `Tokenizer` | May add tokenizer families; counting rules for edge cases may change. |
| `ProviderConfig` | Backoff/SSRF knobs added in v0.2 (`backoff_*`, `allow_private_networks`, `allowed_hosts`). Behavior may be tightened. |
| `hardware` and `costing` public helpers | Backend detection details. |

## Internal

Not part of the public contract. No compatibility promises. Importing these
guarantees nothing except that the module name exists today.

- `opticore.benchmarks.metrics`, `opticore.benchmarks.runner.*` helpers
  (the `BenchmarkRunner`/`BenchmarkResult` shapes are experimental, not internal).
- `opticore.cache.disk` SQLite internals, key hashing internals.
- `opticore.core.model.*` (canonical model internals), `opticore.core.config`
  internals, `opticore.core.pipeline` stage helpers.
- `opticore.logging` internals (redaction filter). `opticore.security` helper
  functions are experimental.
- Provider modules under `opticore.providers.*` beyond the documented
  `generate()`/`name` surface and `ProviderConfig`.
- CLI internals under `opticore.cli` (the `ai-opticore` console script is the
  supported interface).

## Adding new symbols

- New stable symbols follow this policy: documented in the package docstring,
  covered by tests, and reviewed before being promoted from experimental.
- The first release that promotes a symbol from experimental to stable will
  note it in `CHANGELOG.md`.
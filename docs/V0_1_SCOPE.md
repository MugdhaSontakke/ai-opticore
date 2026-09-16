# AI-OptiCore — v0.1.0 Scope

This document is the **contract** for the first production-quality release.
It states what v0.1.0 promises and what it explicitly does not.
A feature that is not listed here is *experimental* or *planned*, not promised.

## Version goal

| | |
| --- | --- |
| Current | 0.1.0 (Early Alpha as of 2026-09-16 audit) |
| Target | 0.1.x — trustworthy for supervised early production use |
| Versioning | Semantic Versioning; no 1.0 claim |

## The minimum stable core (promised)

| Component | What v0.1.0 promises |
| --- | --- |
| Request abstraction | `AIRequest` (provider-independent) with `prompt`, `system`, `messages`, `model`, `max_tokens`, `temperature`, `tools`, `metadata`. `OptimizerRequest` remains as an alias. |
| Response abstraction | `AIResponse` with `content`, `model`, `usage`, `metadata`. `ProviderResponse` remains as an alias. |
| Optimizer interface | `BaseOptimizer(ABC)` with `name`, `order`, `optimize(request, config) -> OptimizerResult`. Optimizers never mutate the original request. |
| Optimization pipeline | Sequential pipeline, per-optimizer enable flags, deterministic order, original request preserved. |
| Token analysis | Real tokenizer (`tiktoken`), tokenizer abstraction, explicit exact/fallback/custom modes. No `chars/4` heuristics. |
| Prompt optimization | SAFE (default) / BALANCED / AGGRESSIVE modes; only verbatim-duplicate and structural whitespace removal; no content deletion on word-count heuristics alone. |
| Context optimization | Token budget, recent-message priority (not true relevance), duplicate-message removal, system-message preservation (never delete in SAFE). |
| Metrics | Canonical keys: `original_input_tokens`, `optimized_input_tokens`, `output_tokens`, `total_tokens`, `tokens_saved`, `reduction_percentage`. |
| Configuration | `opticore.yaml` (or `opticore.json`) + environment overrides; validated; `ConfigurationError` on invalid values. |
| Provider abstraction | `BaseProvider` with `generate`, `models`, `health`, timeouts, auth/rate/timeout errors, fake provider for CI. |
| Basic provider integration | OpenAI-compatible endpoints. Ollama and Hugging Face remain stubs usable via interface, validated against fakes only. |
| Benchmark framework | Baseline vs optimized, optimizer overhead measured separately, JSON export, reproducible metadata (timestamp, versions, OS, model, provider, hardware, config, tokenizer). No fabricated numbers — `N/A` when unknown. |
| Quality/evaluation framework | Request-level quality gate wired into the pipeline; similarity evaluates prompt *and* conversation `messages`; `reject_on_failure` falls back to the original request with an explicit `rejection_reason`; unknown evaluator names raise `QualityEvaluationError` (fail closed, never fail open); "quality not verified" stated explicitly when no evaluator is configured. |
| CLI | `ai-opticore init|config|hardware|models|optimize|benchmark` with `--help`, useful errors, exit codes, no sensitive data. |
| Python API | `from opticore import Optimizer, AIClient` — small public surface; public / internal / experimental separation documented. |
| Security basics | No secrets committed; secret values never logged (functional redaction); no external telemetry; prompt content not logged by default. |
| Tests | Meaningful unit + integration tests for the core; property-based checks where useful; no automated tests require paid APIs or GPUs. |
| CI | Lint, type check, tests, coverage gate, package build, dependency scan, secret scan on every push/PR. |
| Documentation | README matches reality; getting-started; configuration; exceptions; creating an optimizer/provider; benchmark methodology. |

## Explicitly experimental (do not rely on)

- ROCm acceleration (detection + foundation only; performance unverified).
- Inference optimization interfaces (batching, quantization, loaders, memory
  tracking).
- Model routing beyond the default rules.
- Hugging Face / Ollama providers against live endpoints.

## Explicitly NOT in v0.1 (postponed)

- SaaS / cloud / auth / payments / plugin marketplace.
- Distributed inference platform.
- Dashboard with live metrics (build only, no fake charts).
- Redis / SQLite / disk cache backends.
- Real batched inference executor.
- Semantic-cache *background* re-embedding / vector stores.
- Advanced AMD (ROCm) performance optimization.
- Prompt-templates / instruction-preserving semantic rewrites.

## What "production-quality core" means for v0.1

1. The safe fallback is validated: a failing optimization returns the original
   request, not a degraded one.
2. Costs are visible: token change, latency change, and optimizer overhead are
   reported separately.
3. The cache cannot cross isolation boundaries (`model`, `system`,
   `temperature`, `namespace`) and is bounded in memory (bounded index with
   oldest-entry eviction, optional serve-stale duration via `CachePolicy`).
4. Errors are typed and actionable; no silent swallowing.
5. Nothing is claimed in docs that is not implemented and tested.
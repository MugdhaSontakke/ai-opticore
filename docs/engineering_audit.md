# AI-OptiCore — Engineering Audit

> Scope: complete codebase review before the v0.2 production-engineering
> hardening pass. Evidence date 2026-09-20. Baseline verification:
> **178 passed, 5 skipped**, `ruff` clean, `mypy` clean (39 source files), CI
> green at commit `2bfbdbb`. This document records findings; fixes are applied
> in the later stages of the v0.2 plan (see `docs/next_roadmap.md`).

## 1. Architecture

### Core abstractions

| Abstraction | Module | Role |
| --- | --- | --- |
| `AIRequest` / `AIResponse` | `core/model.py` | Canonical provider-independent request/response pair; schema-preserving fields (`tools`, `metadata`) pass through verbatim |
| `OptimizerResult` | `core/interfaces.py` | Result of running optimizers; carries original/optimized request, token counts, changes, timing, accepted/rejection |
| `Optimizer` (Protocol) + `BaseOptimizer` | `core/interfaces.py`, `optimizers/base.py` | Pluggable optimizer contract (`name`, `order`, `optimize`) |
| `OptimizationPipeline` | `core/pipeline.py` | Runs optimizers in `order` sequence, aggregates tokens, applies the quality gate with safe fallback |
| `BaseCache` / `CacheEntry` / `CachePolicy` | `cache/base.py` | Storage-agnostic cache contract; TTL + stale policy |
| `MemoryCache`, `DiskCache`, `SemanticCache` | `cache/` | In-memory (bounded, thread-safe), SQLite (stdlib-only, bounded), and exact+embedding semantic caches |
| `BaseProvider` + `FakeProviderMixin` | `providers/base.py` | Provider contract (`generate`, `models`, `health`); deterministic fake for tests/CI |
| `OpenAIProvider`, `OllamaProvider`, `HuggingFaceProvider` | `providers/` | Concrete providers, lazy dependencies, env-only keys |
| `ModelRouter` / `RouteRule` / `RoutingDecision` | `routing/router.py` | Deterministic, inspectable routing with availability validation |
| `SimilarityEvaluator`, `ResponseEvaluator`, `QualityGate` | `evaluation/evaluator.py` | Pluggable similarity and quality evaluation; request- and response-level |
| `MetricsCollector` | `benchmarks/metrics.py` | Thread-safe counters + series + custom aggregators |
| `CostEstimator` / `CostBreakdown` | `costing.py` | User-priced estimated cost accounting; `None` when unconfigured |
| `Tokenizer` / `TokenOptimizer` / `TokenAnalyzer` | `optimizers/token.py` | Real tokenizer counting with explicit fallback modes |
| `Optimizer` (facade) / `AIClient` / `GenerationResult` | `api.py` | Two public entry points: pure optimization and end-to-end generation |
| `OptimizationConfig` / `ProviderConfig` | `core/config.py` | Eagerly validated, typed configuration |
| CLI | `cli/main.py` | `init/config/hardware/models/optimize/benchmark/cache` |
| Dashboard | `dashboard/` | React viewer; REAL vs DEMO distinction |

### Public APIs
`opticore/__init__.py` intentionally exports ~20 symbols (`AIClient`,
`Optimizer`, `OptimizationConfig`, `SafetyMode`, model/result types, and the
exception hierarchy). Docstring states "everything else is internal or
experimental." There is no explicit `docs/api_stability.md` yet and no
per-symbol stability classification. Routed/`router` functionality is not
exported at the top level.

### Provider interfaces
Contract: `generate(request: dict) -> AIResponse`. Providers read API keys
from env vars only. OpenAI reuses one client instance; Ollama falls back to
HTTP; Hugging Face loads lazily and reuses the model + tokenizer.

### Optimizer interfaces
`optimize(request, config) -> OptimizerResult`. Optimizers are sequential,
order-aware, and each reports its own token impact and an auditable `changes`
list. Enabled-ness derives from config flags at pipeline level.

### Cache interfaces
`BaseCache.get/set/delete/clear/stats`. `cache_key()` provides deterministic
isolation over prompt/system/model/temperature/max_tokens/namespace.
`SemanticCache` layers the semantic index above a `BaseCache` store.

### Routing
`ModelRouter` rules are evaluated in order; availability validated against
`available_models` with fallback resolution; `RoutingError` when nothing is
eligible. Not wired into the CLI; `AIClient` supports an optional `router`.

### Evaluation
Character similarity (default), token-jaccard, and response-level evaluators
are deterministic and dependency-free. The quality gate compares original vs
optimized request text; unknown evaluator names fail loudly.

### Metrics
`MetricsCollector` is thread-safe (RLock), with counters, series, and
`summary()`. `accumulate_tokens`/`token_metrics` standardize token keys.

### Configuration
`OptimizationConfig` validates eagerly (typed errors). YAML/JSON loader
`load_config(path, use_env=True)` + `OPTICORE_*` env overrides. Precedence:
CTOR values < env overrides < ... actually CTOR then env applied last in
`load_config`, so env wins over CTOR/file; explicit CLI `--safety` wins after.
Precedence is **file base then env override**, which matches "env > file > 
default", and explicit Python args are the file load + env; not fully
documented as a precedence table. Unknown keys silently stored in `extra`
(backward-compatible but can hide typos).

### CLI
Argparse; exit codes 0/1 (operational), 2 (argparse misuse), 130 (interrupt).
`--json` present for `optimize` and `benchmark`. No `config validate`, no
`cache stats`.

## 2. Code quality findings

### Duplicated logic
- `_request_text` (pipeline) and `_messages_to_text` (Ollama) and
  `count_tokens` (token.py) each flatten `messages`; three near-identical
  serializations of request text. Minor but consistent.
- Cache key derivation logic appears in `cache_key` (base) and
  `SemanticCache._derive_key` — identical field lists. Risk of drift.

### Coupling
- `api.AIClient` hard-codes the cache pick order (semantic-if-embedding,
  elif disk, else memory). Acceptable but the decision could move into a small
  factory so adding `redis` is a one-line change there and in `config`.
- `benchmarks/runner.py` re-implements optimization + caching instead of
  reusing `AIClient`-style orchestration. It intentionally measures separate
  phases (baseline vs optimized), so duplication is partly by design; the
  cache-hit path duplicates `SemanticCache.lookup` semantics on `MemoryCache`.

### Overly large modules
- `api.py` (511 lines) mixes `AIClient`, `Optimizer`, `GenerationResult`, and
  embedding plumbing. Acceptable as the "public API" module but routes state
  through many local variables.
- `benchmarks/runner.py` (570 lines) is the largest module; scenario/dataset
  internals are testable but the metrics assembly could be split.

### Inconsistent naming
- `ProviderAuthError` vs `ProviderAuthenticationError` (prompt's example);
  existing name is fine and documented.
- `ProviderResponse` alias (`core/model.py`) vs `base.py` re-export vs
  `providers/__init__.py` re-export — three places, fetch risk.

### Unsafe defaults / missing validation
- `ollama.models()` swallows `Exception` and returns `[]` (line 137) — a
  connection failure is indistinguishable from "no models." Low risk: callers
  treat empty as "none", never fabricate.
- `ProviderConfig.base_url` accepts any string (SSRF surface); no validation.
- `_apply_env_overrides` mutates a *possibly shared* `OptimizationConfig` in
  place (env override path only) instead of returning a copy — benign today
  because `load_config` builds a fresh instance.
- `huggingface._count_tokens` and `tokenizer` exceptions are swallowed to
  return 0/lists — deliberate "never fabricate" but unobservable. Acceptable
  with metadata flags present.

### Swallowed exceptions / error handling inconsistencies
- `api.generate` catches `Exception` around cache lookup (intentional,
  documented) — good; logs with `request_id`.
- `benchmarks/runner.py` catches `Exception` on cache write during benchmark —
  logs generic message without context.
- OpenAI provider re-raises unmapped `openai` exceptions **raw**
  (`raise` in the generic except) — an `openai.APIConnectionError`,
  `APIStatusError`, or `BadRequestError` escapes the typed `ProviderError`
  family. **This is the biggest error-contract gap** and will be fixed in the
  provider-reliability stage.
- OpenAI `choices[0]` indexing will raise `IndexError` on malformed responses
  (empty choices) — raw, not `ProviderResponseError`.
- Ollama/TFS have **no retry/backoff** at all; OpenAI relies on SDK
  `max_retries`. No `backoff` config knob exists.

### Dead / placeholder code
- `inference/` (`batching.py`, `quantization.py`, `runtime.py`) exposes
  interfaces with honest `NotImplementedError`/empty stubs and is advertised
  as "experimental/planned" — keep but document labels (no misrepresentation).
- `evaluation.QualityGate` and `TaskSuccessFunc` are not wired into the
  pipeline (pipeline uses its own gate logic); `DefaultResponseEvaluator` also
  unwired. Response-level quality is a documented gap.
- `CachePolicy`/`CacheResponse` are used lightly (policy in MemoryCache;
  `CacheResponse` wrapper is built nowhere — dead surface).
- `routing` `max_latency_ms` field is documented as a soft/no-op — okay, but
  should be labeled.
- `TokenAnalyzer` is unused by the pipeline (public utility).

### TODOs / noise
- No stray `TODO`s found in source.
- `config_template()` has one mis-aligned literal (`"log_prompts": False,`).
- `.mypy_cache` exists under `src/` locally but is git-ignored (not tracked).

## 3. Production risks

### Concurrency
- `MemoryCache`, `DiskCache`, `SemanticCache`, `MetricsCollector` use RLock —
  good. **No concurrency tests exist yet** for cache/MetricsCollector at
  scale (10/50/100 threads).
- `OpenAIProvider._get_client` lazy init is not synchronized; two threads may
  build two clients (harmless but wasteful; race is benign).
- `Tokenizer._load` is not synchronized; benign double-load possible.
- `AIClient` instance `metrics`/`cache` shared across threads: counters and
  cache ops are locked; `SemanticCache` exact-match path reads
  `self._store.get` (locked in MemoryCache). Acceptable.
- Request IDs: `uuid4().hex[:12]` — collisions negligible.

### Cache consistency
- TTL semantics: `time.monotonic` everywhere (good: immune to wall-clock
  jumps); disk TTL persists across restarts using monotonic values, which are
  meaningless after a restart (entries may have out-of-range `expires_at`).
  **Risk: after process restart, a disk entry with a very large monotonic
  `expires_at` may look fresh forever.** Noted for cache-hardening stage.
- DiskCache eviction: `DELETE ... NOT IN (SELECT key ... ORDER BY created_at
  DESC LIMIT n)` after each insert — O(n log n) per insert at scale.
- DiskCache: metadata `json.dumps` with `sort_keys`; corrupted rows degrade to
  `{}` silently; corrupt DB file raises `CacheError` (wired to fall back to
  model in `AIClient`) — good.
- Semantic index (`_entries`) and the underlying exact store can drift: `set`
  with metadata inserts to store AND index; deleting a semantic entry removes
  from both. Bounded at `max_entries`.

### Provider timeouts / retries
- Timeout: only OpenAI SDK timeout and Ollama HTTP timeout exist. Hugging Face
  has none (local model). Ollama python-client path has no timeout.
- Retries: OpenAI SDK `max_retries`; **no retry/backoff for Ollama HTTP or
  HF; no non-retryable classification** (auth/invalid request would be retried
  by the SDK). Deterministic `RetryPolicy` needed.

### Malformed responses
- Empty `completion.choices` → `IndexError` (raw). Missing `usage` handled
  gracefully. `ollama` python client returns dicts; `_generate_via_http`
  assumes `data` is a dict — a non-JSON/error body raises `KeyError` raw.

### Configuration failures
- Invalid `cache_backend` raises an actionable `ConfigurationError` — good.
- Unknown top-level keys silently accepted (`extra`). Precedence across
  CTOR/file/env/CLI is *implemented* (env > file, CLI `--safety` > config) but
  not *documented* as a table.
- `ProviderConfig` lacks `backoff_base`, `backoff_max`, and any SSRF scope.

### Resource leaks / memory growth
- `MemoryCache` bounded. `SemanticCache` exact store bounded via MemoryCache's
  own `max_entries=10_000` default, semantic index bounded. **DiskCache uses
  max_entries eviction** — bounded.
- `MetricsCollector.series` grow unbounded per key (lists append forever on a
  long-running `AIClient`). **Memory growth risk in long-running processes.**
- HTTP clients: OpenAI SDK creates its own session (reused); Ollama
  `requests.post` per call (no persistent session) — acceptable; no explicit
  close paths for HF model (process-lifetime by design).

### Secret leakage / unsafe logging
- Config files reject sensitive keys (nested too) — strong.
- `RedactingFilter` applied to all `get_logger` handlers; `OPTICORE_LOG_LEVEL`
  can force DEBUG but prompt content still gated by `log_prompts`.
- `AIClient.generate` line ~287: `logger.warning(fallback_reason)` — built
  string; fine.
- **Dashboard** is client-side; demo data marked DEMO; no secrets.
- `exception` messages from providers are included in raised errors — no keys
  by construction (SDK redactions aside); document explicitly.

### Dependency risks
- Runtime deps: `tiktoken`, `pyyaml`, `requests`/`urllib3`
  (interpreter-gated pins for py>=3.10). Low. `pip-audit` in CI.
- Optional deps: `openai`, `ollama`, `transformers+torch`, `sentence-transformers`.
  Heavy deps are optional — correct. Docker currently installs `[all]` (pulls
  torch/transformers) — **image is huge**; trim for the service image.

## 4. Test-suite gaps (evidence-based)

Present today: property tests (Hypothesis), cache isolation + disk cache,
evaluation suite from `test_cases.json`, provider error mapping tests
(`test_provider_errors.py`), quality gate, redaction, CLI, benchmark honesty.

Missing / weak:
- No mock-provider **retry/backoff**, timeout linearization, 429, 5xx,
  malformed-response, auth-failure tests at the provider layer (partially in
  `test_provider_errors.py`).
- No concurrency tests (cache/MetricsCollector/router/request-ID).
- No `BaseCache` conformance test shared by memory/disk/semantic.
- No SSRF/security tests.
- Live provider tests are env-gated (correct) and untested in CI (correct).
- No test for disk-cache monotonic-TTL-across-restart behaviour.

## 5. Constraints and conventions to respect

- Python 3.9 runtime floor; no `zip(strict=)`, no 3.10+ syntax.
- "Measured, not claimed": never fabricate metrics; `N/A` for unmeasured.
- Statuses are evidence-based (DONE/PARTIAL/...), no arbitrary percentages.
- Secrets: env-only; never logged; config files reject secret keys.
- Behavior changes require tests; run ruff + mypy + full suite.
- One clean logical commit per milestone; push only when asked.

## 6. Recommended v0.2 change set (by audit finding)

1. **Error contract** (Stage 2/3): add `ProviderUnavailableError`,
   `ProviderResponseError` (+ map `APIConnectionError`, `APIStatusError`,
   malformed `choices`/body); keep `ProviderAuthError`/`ProviderRateError`/
   `ProviderTimeoutError`; document `docs/exceptions.md` + `api_stability`.
2. **Retry policy** (Stage 3): deterministic `RetryPolicy`
   (`max_attempts`, `base_delay`, `max_delay`, retryable-status set),
   non-retryable classification (auth, invalid request, unsupported model),
   applied in OpenAI/Ollama HTTP; unit-tested against mocked transports.
3. **Mock-provider suite** (Stage 3): success/timeout/429→backoff→success/
   500→retry→fail/malformed→`ProviderResponseError`/auth, no real keys.
4. **Config precedence + validation** (Stage 2): document precedence table;
   `docs/api_stability.md`; optional `ai-opticore config validate`.
5. **Cache hardening** (Stage 4): fix monotonic-TTL-across-restart; bounded
   metrics series (ring buffer / sampler) or document retention; concurrency
   tests (N=10/50/100); `CacheResponse` dead surface removed or documented;
   unify `_derive_key`/`cache_key`.
6. **Optimization safety** (Stage 5): add explicit pipeline `strict`/
   `fail_open` semantics; document optimizer contract; keep per-step timing.
7. **Observability** (Stage 6): emit structured lifecycle events
   (`request_started → ... → request_completed`) with request_id correlation;
   bounded metrics/percentiles (p50/p95/p99).
8. **Security + SSRF** (Stage 7): SSRF validator for `base_url`
   (allow loopback/private for local Ollama/vLLM by config; document
   behavior); secret-redaction tests; dependency audit retained.
9. **CLI/Docker** (Stage 8): `config validate`, `cache stats`; documented exit
   codes; Docker: non-root user, healthcheck, minimal deps image (core, not
   `[all]`), compose only if justified (a Redis compose is *not* justified).
10. **CI/release** (Stage 9): keep matrix; add build+examples+security jobs;
    organize new tests into `tests/unit|integration|provider|performance|
    security`; version → 0.2.0; `docs/release_checklist.md`.
11. **Docs/examples** (Stage 10): the full `docs/` set from the 26-point doc
    list; runnable `examples/` set.
12. **Real-provider validation** (Stage 11): env-gated; run per release.
13. **Final assessment** (Stage 12): `docs/production_readiness.md` etc.

No Kubernetes/microservices/Kafka/Celery/PostgreSQL/distributed tracing are
needed by this codebase; a shared in-process `MemoryCache`/`DiskCache`
suffices for v0.2. Redis remains an interface-level extension point, not a
new dependency.
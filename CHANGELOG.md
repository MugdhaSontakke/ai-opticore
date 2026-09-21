# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-09-21

### Added — Real end-to-end LLM evaluation with Ollama + Llama 3.2

- Ollama `health()` now performs a real, staged connectivity check
  (server reachable → installed models → requested model present → optional
  small generation probe) with distinct `error_kind` results
  (`not_running`, `timeout`, `http:<status>`, `malformed_response`,
  `model_not_found`). Never raises for an offline server.
  (`src/opticore/providers/ollama.py`)
- New `ai-opticore doctor` command: environment diagnostics (package, config,
  provider connectivity) with `--json`, `--provider`, `--model`, `--no-probe`;
  exits 0 only when all critical checks pass.
  (`src/opticore/cli/main.py`)
- `python -m opticore` is now a working alias for the CLI
  (`src/opticore/__main__.py`).
- `OLLAMA_MODEL` environment variable honored as the default Ollama model.
- Benchmark report additions: `provider_info` (Ollama version / endpoint /
  installed-model count), `tokens_per_sec_baseline` / `tokens_per_sec_optimized`
  (on responses with real measurable latency), heuristic
  `quality_response_similarity_avg` + `quality_response_token_jaccard_avg`
  comparing baseline vs optimized *responses*, and a `sample_responses` list
  preserved on the result object but never serialized to JSON.
  (`src/opticore/benchmarks/runner.py`)
- Real-LLM benchmark dataset: 26 deterministic prompts across 9 categories
  (`benchmarks/data/real_llm_dataset.json`).
- `examples/ollama_demo.py`: real end-to-end demo (MODE A baseline vs
  MODE B optimized+cached, cache MISS→HIT, cost N/A for local Ollama, typed
  suggestions when the server/model is missing).
- Mock-based tests for the Ollama HTTP paths and health probe
  (`tests/test_ollama_health.py`); live opt-in health regression test added to
  `tests/test_providers_live.py` (runs only with `OPTICORE_TEST_LIVE=1`).
- Docs: `docs/ollama.md` real-LLM guide; README "Real LLM evaluation with
  Ollama" section; `.env.example` documents `OLLAMA_MODEL`.

### Changed

- urllib3 `NotOpenSSLWarning` (LibreSSL on macOS) is suppressed at the Ollama
  `requests` import site so CLI output stays clean.

## [Unreleased]

### Added — v0.2 hardening pass (SSRF guardrails, retry policy, resilience)

- SSRF guardrails: `opticore.security.validate_base_url` blocks link-local /
  metadata / private-network hosts by default (`allow_private_networks=False`)
  while keeping local Ollama/vLLM working; `allowed_hosts` hard-allowlist
  overrides; applied in the OpenAI, Ollama, and HuggingFace providers.
- Request redaction: `opticore.security.redact_text` / public
  `opticore.logging.redact_text` scrub API keys, bearer tokens, and
  authorization headers from every provider error message and log.
- Retry policy: `opticore.providers.retry` with exponential backoff
  (`PercentileReachedError`-free), retryable statuses {408,425,429,500,502,503,504},
  and generic `retry_call` that never retries or wraps already-typed
  `ProviderError` failures.
- `ProviderConfig` knobs: `backoff_base_seconds`, `backoff_max_seconds`,
  `allow_private_networks`, `allowed_hosts`, and derived `max_attempts`
  (= `max_retries + 1`). Config constraints raise `ConfigurationError`.
- Providers retry themselves around a bounded budget instead of relying on SDK
  built-in retries (`max_retries=0`); SDK/HTTP timeout, connection, rate, and
  server errors map to typed `ProviderError` subclasses with redacted messages;
  malformed/non-dict provider responses raise `ProviderResponseError`.
- Pipeline resilience: `strict_optimizers` flag; by default a failing optimizer
  logs and is skipped (metadata records `optimizer_errors`), with
  `strict_optimizers=True` it re-raises `OptimizationError`.
- Disk cache: wall-clock TTLs survive process restarts (converted to a
  monotonic timeline on read); `stats()` now reports `evictions` and
  `invalidations` counters.
- Metrics: `MetricsCollector` series are bounded (`max_series_len=1000`) and
  `summary()` reports p50/p95 latency when enough samples exist.
- CLI: `ai-opticore config validate` (exit-code checked) and `config show`,
  `cache stats` / `cache explain`, `--json` output on config/cache/benchmark,
  and `benchmark --dataset <file>` (category→scenario mapping, sample
  metadata preserves `id`/`expected_keywords`).
- Benchmark JSON (`BenchmarkResult.to_dict`) now includes median/p95 latency,
  dataset, environment fingerprint, overhead/cost/quality fields; dashboard
  renders them and shows DATA UNAVAILABLE until a real `--json` file is loaded.
- Docker: non-root `Dockerfile` (minimal install, build-time `config validate`
  smoke check, HEALTHCHECK), `.dockerignore`, `docker-compose.yml` with an
  optional local `ollama` service. Build uses the CLI as the container entry
  point (`docker compose run --rm app <command>`).
- Docs: `docs/api_stability.md` classifies Stable / Experimental / Internal
  symbols; version bumped to 0.2.0.
- 258 runtime tests (incl. Hypothesis), ruff + mypy clean, ~78% coverage.

### Added

- MVP module architecture: pipeline, optimizers, cache, providers, routing,
  hardware, inference, benchmarks, evaluation, metrics, CLI.
- Token analysis with real tokenizer implementations (`tiktoken` default,
  custom counters supported) and canonical token metrics
  (`original_input_tokens`, `optimized_input_tokens`, `output_tokens`,
  `total_tokens`, `tokens_saved`, `reduction_percentage`).
- Typed exception hierarchy (`opticore.exceptions`: `OptiCoreError`,
  `ConfigurationError`, `OptimizationError`, `QualityEvaluationError`,
  `CacheError`, `ProviderError` + `ProviderAuthError`/`ProviderRateError`/
  `ProviderTimeoutError`, `UnsupportedModelError`,
  `TokenizerNotFoundError`, `HardwareBackendUnavailableError`).
- Canonical request/response model (`AIRequest`/`AIResponse`) that preserves
  `system`, `temperature`, `tools`, and `namespace` through the pipeline.
- Quality gate wired into the pipeline with safe fallback to the original
  request on rejection, plus per-optimizer `changes` audit trail.
- Config hardening: eager validation, YAML/JSON loading, `OPTICORE_*`
  env overrides, `quality` block flattening, secret-key rejection in files.
- Cache isolation: cache keys now include `system`, `temperature`,
  `max_tokens`, and `namespace`; semantic cache scoped per request.
- Optimizer overhead measurement (`optimizer_overhead_ms_avg`,
  `optimizer_time_ms`/`model_time_ms`/`total_time_ms`).
- Benchmark environment fingerprint and JSON export (`--output`,
  `BenchmarkResult.to_dict`).
- CLI: default `opticore.yaml`, clean error handling (typed exit codes, no
  tracebacks), benchmark `--output`, honest `cache` diagnostics.
- Hardware `require_backend()` helper raising
  `HardwareBackendUnavailableError` for callers needing hard failures.
- Build verification (`python -m build`) and hardening extras (`yaml`).
- 151 runtime tests (incl. Hypothesis property tests) with a coverage gate.
- Pipeline reports per-optimization timing and outcome: `optimization_time_ms`,
  `accepted`, `rejection_reason` on `OptimizerResult`/`OptimizeOutcome`, plus
  `optimizer_time_by_step_ms` in pipeline metadata.
- Quality gate now evaluates conversation history: `messages` are included in
  the comparison text; unknown `quality_evaluator` names raise
  `QualityEvaluationError` instead of failing open.
- Context optimizer: `max_token_budget == 0` now drops all messages explicitly.
- Cache hardening: `MemoryCache(policy=CachePolicy)` can serve stale entries
  when the policy allows; `SemanticCache` is bounded by `max_entries`
  (default 10_000) with oldest-entry eviction.
- Opt-in prompt logging: `log_prompt_content()` gated by `LOG_PROMPTS` /
  `config.log_prompts` (still off by default and redacted).
- Providers: OpenAI client is re-used across calls (lazy init in `_get_client`);
  Ollama maps timeouts to `ProviderTimeoutError` and HTTP errors to typed
  `ProviderError`s; Hugging Face uses real tokenizer token counts (metadata
  reports `token_counts: tokenizer|unmeasured`) and honors `temperature` via
  `do_sample`.
- CLI hardening: the implicit `opticore.yaml` is actually loaded (and fails
  loudly with exit code 1 when it exists but is invalid); `optimize` output
  (text + JSON) includes `optimization_time_ms`, `accepted`, `rejection_reason`;
  CLI misuse exits with argparse's code 2.
- `safety_mode` validation raises `ConfigurationError` (not a leaked
  `ValueError`) and `from_dict` wraps invalid config values consistently.
- Benchmarking now measures real cache hit/miss counts and a `cache_hit_rate`
  percentage when the semantic cache is enabled (`N/A` when disabled).
- Production-validation pass: observability, cost, disk cache, routing safety,
  provider regression tests, per-scenario benchmarks, honest docs.
  - Per-request observability: `request_id`, end-to-end `total_time_ms`,
    `cache_lookup_ms`, routing decision, quality verdict, `fallback_reason`,
    and metrics series (`optimizer_time_ms`, `cache_lookup_ms`,
    `model_time_ms`, `estimated_cost_per_request`) plus counters
    (`cache_read_errors`, `cache_write_errors`, `quality_gate_rejections`,
    `fallbacks`, `routing_failures`).
  - Costing: `opticore.CostEstimator` with user-configured pricing
    (`pricing.input_per_1k` / `output_per_1k`, aliases supported). Costs are
    always labeled `estimated`; `N/A` when pricing is unconfigured. Wired into
    `AIClient` (per-request `estimated_cost`), benchmark output, and metrics.
  - Disk cache backend: `opticore.cache.DiskCache` (SQLite, stdlib-only, TTL,
    bounded by `max_entries`, typed `CacheError` on corruption), selected via
    `cache_backend="disk"` + `cache_disk_path`. Cache read/write failures fall
    back to the model.
  - Model routing safety: `RoutingError`, availability validation against
    `available_models`, fallback resolution (preferred → fallbacks → default),
    decision logging and per-request `metadata.routing`.
  - `tools` passthrough in `pipeline.run(...)` and the benchmark runner.
  - Benchmarking: median + p95 latency, `net_latency_change_ms`,
    `provider_latency_ms_avg`, `optimization_overhead_tokens` (0 — local
    deterministic optimizers, stated honestly), estimated cost, per-scenario
    (A–H) summaries, JSON dataset loader (`load_benchmark_dataset`), and
    honest notes when latency increases.
  - Provider regression tests: `tests/test_providers_live.py` (env-gated,
    skips cleanly in CI) + optional manual `provider-live` GitHub workflow.
  - Deterministic quality-regression dataset `tests/evaluation/test_cases.json`
    and `tests/test_evaluation_suite.py`.
  - Dashboard distinguishes REAL measured data from DEMO (no fabricated claims).
  - Docs: `docs/production_readiness.md` (matrix),
    `docs/production_checklist.md` (scorecard), `docs/next_roadmap.md`,
    README + configuration updates (pricing, disk cache, live tests).
  - 178 runtime tests (incl. Hypothesis property tests), coverage ~79%.

### Fixed

- YAML config loading works out of the box: `PyYAML` is now a required
  dependency. Previously it was a separate extra, so a minimal
  `pip install ai-opticore` + `ai-opticore init` + `optimize` failed to read
  the default `opticore.yaml`.
- `RedactingFilter` now actually redacts secrets (real filter implementation).
- Silent cache-write exceptions are now logged and counted in metrics.
- Provider timeouts and auth failures map to typed provider errors.
- `datetime.UTC` avoided for Python 3.9 compatibility; `requires-python`
  corrected to `>=3.9`; project URLs point at the real repository.
- README installation extras and quick-start API verified against the code.

### Security

- Redacting log filter for sensitive fields.
- Prompt/content logging disabled by default.

## [0.1.0] - 2026-09-16

### Added

- Project scaffolding: pyproject.toml, package layout, docs, GitHub
  templates, CI workflow, LICENSE, SECURITY.md, CODE_OF_CONDUCT.md.
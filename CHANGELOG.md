# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
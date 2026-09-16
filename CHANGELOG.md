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
- 131 runtime tests (incl. Hypothesis property tests) with a coverage gate.

### Fixed

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
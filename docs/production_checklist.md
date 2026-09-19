# AI-OptiCore — Production Checklist

Statuses are evidence-based, not aspirational. "DONE" means the behavior is
implemented *and* covered by a passing test (or directly verifiable end-to-end).
No percentage scores are used: a 60%-covered module is not "60% done".

Legend: **DONE** · **PARTIAL** · **EXPERIMENTAL** · **NOT STARTED**

## Reliability / correctness

| Item | Status | Evidence |
| ---- | ------ | -------- |
| Optimizers never mutate caller data | DONE | `test_pipeline_result_immutability` |
| Unhandled optimizer exception cannot break the app | DONE | pipeline catches `OptimizationError`, records it, continues; `test_optimization_failure_is_not_application_failure` |
| Quality gate rejects → original request restored | DONE | `test_quality_gate_rejection_restores_original_request` |
| Unknown quality evaluator fails loudly | DONE | `QualityEvaluationError` raised |
| Unknown provider fails loudly | DONE | `get_provider` raises `ProviderError` (tested) |
| Cache read/write failure → continue to model | DONE | `test_aiclient_survives_cache_lookup_failure` / write |
| Cache isolation by system/model/temperature/max_tokens/namespace | DONE | `test_cache_isolation.py` |
| Semantic cache bounded with eviction | DONE | `test_v01_core_hardening.py` |
| Disk cache persists across processes; corrupt DB handled | DONE | `test_cache_disk.py` |
| Tokenizer missing → typed error | DONE | `TokenizerNotFoundError` tested |
| Config invalid values fail fast | DONE | `ConfigurationError` eager validation tested |
| Secrets rejected from config files | DONE | config loader tests |
| Model selection never silent/unavailable | DONE | `ModelRouter` availability validation + `RoutingError` |

## Observability

| Item | Status | Evidence |
| ---- | ------ | -------- |
| Per-request record (request_id, provider, model, tokens, timings) | DONE | `GenerationResult` + `metadata` |
| Optimizer/cache/provider/end-to-end latency split | DONE | `optimizer_time_ms`, `cache_lookup_ms`, `model_time_ms`, `total_time_ms` |
| Token reduction measured, not assumed | DONE | `token_metrics()`; benchmark from real counts |
| Routing decision observable per request | DONE | `metadata.routing` + metrics counter |
| Quality verdict observable per request | DONE | `metadata.quality` |
| Fallback reason observable | DONE | `fallback_reason` |
| Estimated cost labeled `estimated` | DONE | `CostEstimator` |
| Secret-safe logging | DONE | `RedactingFilter`, opt-in prompt logging |

## Performance / benchmarking

| Item | Status | Evidence |
| ---- | ------ | -------- |
| Baseline vs optimized measured | DONE | `BenchmarkRunner` |
| Median + p95 reported, not just mean | DONE | `latency_median_ms` / `latency_p95_ms` |
| Net latency change reported honestly (incl. slowdowns) | DONE | `net_latency_change_ms` + notes |
| Optimization overhead vs provider latency separated | DONE | `optimizer_latency_ms_avg`, `provider_latency_ms_avg` |
| Scenarios A–H with per-scenario summaries | DONE | `benchmarks/runner.py` + dataset loader |
| Cost estimated from user pricing only | DONE | `pricing_*` config |

## Providers

| Item | Status | Evidence |
| ---- | ------ | -------- |
| OpenAI-compatible provider (reusable client, typed errors) | DONE | code + unit tests |
| Ollama provider (HTTP fallback, typed timeouts) | DONE | code + unit tests |
| Hugging Face provider (real tokenizer, honors temperature) | DONE | code + unit tests |
| Real-endpoint regression tests | DONE (opt-in) | `tests/test_providers_live.py` skips without `OPTICORE_TEST_LIVE=1` |
| No API keys in repo/CI | DONE | env-only; gitleaks in CI |

## Security

| Item | Status | Evidence |
| ---- | ------ | -------- |
| Secrets redacted in logs | DONE | logging tests |
| Config files cannot carry secrets | DONE | loader tests |
| Dependency vulnerability scan | DONE | `pip-audit` job |
| Secret scan + CodeQL | DONE | gitleaks + CodeQL jobs |
| SSRF guardrails for custom `base_url` | PARTIAL | documented risk; no endpoint allow-list yet |
| Prompt/response content not logged by default | DONE | `log_prompts` false by default |

## CI / packaging / doc

| Item | Status | Evidence |
| ---- | ------ | -------- |
| Fresh install + tests in CI (3.9/3.11/3.12) | DONE | `ci.yml` |
| Lint + mypy + coverage gate | DONE | `ruff`, `mypy`, `--cov-fail-under=70` |
| Build sdist+wheel and `twine check` | DONE | `build` job |
| CLI smoke (init/hardware/optimize/benchmark/cache) | DONE | `cli-smoke` job |
| Live-provider manual workflow | DONE | `provider-live.yml` (manual dispatch) |
| Dashboard separates REAL vs DEMO | DONE | `dashboard/src/App.tsx` |
| Docker image | PARTIAL | `Dockerfile`; no compose/healthcheck yet |
| Docs updated with this milestone | DONE | README + `docs/` |
| Redis cache backend | NOT STARTED | extension point on `BaseCache` |

## Overall readiness estimate

Collected evidence: **178 tests passing** (5 opt-in live-provider skips),
`ruff` clean, `mypy` clean, coverage ~79%,
dashboard production build succeeds, CI jobs defined for test/build/cli-smoke/
security/codeql/live.

- **Development-ready: DONE.** Core functionality works and is tested locally.
- **Beta-ready: DONE** for the documented scope — external users can test it with
  the documented limitations (live-provider paths, routing off by default,
  Docker is basic).
- **Production-ready: NOT YET.** Remaining blockers are operational, not
  structural: per-release live-provider validation, SSRF guardrails, a
  production-grade deployment story (compose, healthcheck), and optional
  Redis-backed cache. Each is tracked in `docs/next_roadmap.md`.

**Project stage (honest label):** *Alpha — hardening toward Beta / production
validation release.* It is not marketed as production-ready.
# AI-OptiCore — Production Readiness Matrix

> Evidence date: 2026-09-19 · Verification: `pytest` (178 passing, 5 skipped —
> the env-gated live-provider tests skip cleanly), `ruff` clean, `mypy` clean,
> dashboard production build clean, CLI benchmark smoke OK.
> Baseline commit: `1ac316a`; this matrix tracks the current working tree.
> Every cell below is backed by repository evidence (tests, source, CI), not by
> the existence of an interface.

| Area                  | Current Status | Evidence | Risk | Required Work |
| --------------------- | -------------- | -------- | ---- | ------------- |
| Core pipeline         | DONE | `OptimizationPipeline` (`core/pipeline.py`) runs optimizers by order, preserves the original request, aggregates tokens, records per-step timing, surfaces `accepted`/`rejection_reason`, and never mutates caller data (immutability tests). | Low | Optional: custom pipeline hooks / plugins. |
| Prompt optimization   | DONE | SAFE/BALANCED/AGGRESSIVE modes, verbatim duplicate removal only, severity hints, tools/system preserved verbatim (`test_prompt_optimizer.py`, evaluation suite). | Low | LLM-based rewrite stage (explicitly experimental). |
| Context optimization  | DONE | Duplicate-message removal, oldest-first truncation under `max_token_budget`, `prioritize_recent`; budget `0` is an explicit no-op. | Medium (semantics of aggressive summarization are heuristic) | Relevance-based selection (research, not shipped). |
| Semantic cache        | DONE | `SemanticCache` exact-match short-circuit + cosine lookup, threshold, TTL, bounded (`max_entries`) with eviction, full isolation keys (system/model/temperature/max_tokens/namespace). | Medium (semantic threshold is user-tuned) | Optional real embedding test vectors; more embedding backends. |
| Cache backends        | DONE (memory + disk), INTERFACE (redis) | `MemoryCache` (thread-safe, TTL, stale policy) and `DiskCache` (SQLite, stdlib-only, TTL, bounded, user-selectable via `cache_backend`) share `BaseCache`; `AIClient` falls back to the model on any cache read/write failure. | Low | Redis backend behind `BaseCache` (explicitly later/optional). |
| Model routing         | PARTIAL (deterministic + validated) | `ModelRouter` validates against `available_models`, prefers fallback models then default, raises `RoutingError` when nothing is eligible; decisions are logged and exposed in `GenerationResult.metadata.routing`. | Low (not on by default; `enable_model_routing=False`) | Latency/cost-aware routing signals; provider capability checks per call. |
| Provider integrations | PARTIAL (openai/ollama/hf + fakes) | OpenAI SDK reuse, typed auth/rate/timeout errors, HTTP fallback for Ollama; Hugging Face reports real tokenizer counts. Real-endpoint regression tests are env-gated (`tests/test_providers_live.py`) — untested paths exist by design until a developer runs them. | Medium (live paths unverified in CI) | Run live suite per release; mock-based unit tests for SDK paths. |
| Quality gate          | DONE | Unknown evaluator raises `QualityEvaluationError`; rejection restores the original request with `accepted=False`/`rejection_reason`; conversation messages included in similarity; safe fallback verified by tests. | Medium (set-based evaluators are coarse) | Embedding/proxy evaluator plugins; response-level gate in the CLI. |
| Metrics               | DONE | Per-request record with request_id, optimizer/cache/provider/end-to-end times, token reduction, routing decision, quality verdict, fallback reason, estimated cost. `MetricsCollector` counters + series. | Low | Live dashboard consumption of these metrics. |
| Benchmarking          | DONE | Baseline vs optimized, per-scenario (A–H) dataset, median + p95 latency, `net_latency_change_ms`, optimizer/provider latency split, honest cache-hit rate, `estimated_cost` (only when priced) — nothing fabricated; negative net gains are reported in notes. | Low | More real-provider points in the matrix; a dedicated perf regression suite. |
| Cost analysis         | DONE (estimated, user-priced) | Configurable `pricing.input_per_1k` / `pricing.output_per_1k`; `CostEstimator` returns costs labeled `estimated` and `None` when unconfigured; wired into requests, benchmark and metrics. | Low (never provider-quoted) | Vendor price catalogs (explicitly out of scope to avoid stale claims). |
| Security              | DONE (baseline), PARTIAL (advanced) | Secret-safe scoped logging + `RedactingFilter`; config files reject secret keys; keys read only from env; CI runs `pip-audit`, gitleaks, CodeQL. | Low–Medium | SSRF guardrails for custom `base_url`; optional TLS pinning; signed artifacts. |
| Testing               | DONE (core), GATED | 178 tests (5 opt-in skips) incl. property tests, cache-disk, evaluation suite, cache-failure fallback, routing safety, failure/fallback paths; CI enforces coverage ≥70% (currently ~79%). | Low | Live provider suite in a scheduled run; fuzz larger prompts. |
| CI/CD                 | DONE | Matrix tests (3.9/3.11/3.12), lint, mypy, build + `twine check`, CLI smoke (incl. benchmark), security audit, CodeQL, optional manual live-provider workflow. | Low | Nightly live-provider job; release automation. |
| Dashboard             | DONE (viewer, honest) | Loads real `--json` output, hard-coded demo data is explicitly labeled DEMO with N/A claims; shows a REAL/DEMO badge. | Low | Live metrics endpoint (later). |
| Docker/deployment     | PARTIAL | `Dockerfile` present; install path == documented quickstart; disk cache supports stateful containers. | Low | Compose example with volume for disk cache; healthcheck. |
| Documentation         | PARTIAL | README + `docs/` (architecture, config, exceptions, benchmark methodology, readiness, checklist, roadmap). Some claims still to be freshly audited each release. | Low | Release-notes ↔ docs coupling; API reference. |

## Definition of the statuses used

- **DONE** — the behavior exists, is tested, and its honesty properties are verified.
- **PARTIAL** — the capability exists with documented limitations or is not exercised by default.
- **INTERFACE** — the abstraction point exists but the concrete backend is not shipped.
- **NOT STARTED / EXPERIMENTAL** — see `docs/production_checklist.md`.

## Delivery readiness

- **Development-ready:** yes — install, `pytest`, CLI, benchmark, examples all work locally.
- **Beta-ready:** yes for documented scope — external users can benchmark with fake or real
  providers; live-provider tests are opt-in; limitations are documented.
- **Production-ready:** not claimed. Remaining gaps are operational (live-provider validation
  per release, SSRF guardrails, dashboard live metrics, Redis backend), not structural.

See `docs/production_checklist.md` for the item-by-item scorecard and
`docs/next_roadmap.md` for the plan.
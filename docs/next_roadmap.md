# AI-OptiCore — Next Roadmap

Milestone: **Beta / Production Validation Release**. The v0.1 hardening pass
(delivered in `docs/PRODUCTION_READINESS.md` + this repo at v0.1.0) and the
production-validation pass (2026-09-19) are complete. What follows is the
honest, ordered plan.

## Completed (verified in this repository)

- **v0.3.0 — real end-to-end LLM evaluation with Ollama + Llama 3.2**
  (2026-09-21): live `health()` probe with staged diagnostics feeding
  `ai-opticore doctor`; `python -m opticore`; heuristic baseline-vs-optimized
  response similarity, tokens/sec, and provider info in benchmark reports; a
  26-prompt deterministic real-LLM dataset; `examples/ollama_demo.py`;
  mock-based test coverage of the Ollama HTTP/probe paths (no network in CI);
  live opt-in Ollama regression test (`OPTICORE_TEST_LIVE=1`). See
  `docs/ollama.md`. Run the real benchmark on a machine with Ollama:
  `ai-opticore benchmark --provider ollama --model llama3.2
  --dataset benchmarks/data/real_llm_dataset.json --json`.
- Quality gate wired into the pipeline with safe fallback and loud failure on
  unknown evaluators.
- Optimizer-overhead vs provider-latency decomposition; median/p95 latency;
  `net_latency_change_ms`; per-scenario (A–H) benchmark summaries.
- Estimated cost accounting (user-configured pricing, always labeled
  `estimated`, `N/A` when unconfigured).
- Disk cache backend (SQLite, stdlib-only) behind `BaseCache`; cache
  read/write failures fall back to the model.
- Model routing availability validation, fallback ordering, typed
  `RoutingError`, observable decisions.
- Per-request observability: `request_id`, routing + quality + fallback +
  cost metadata, timing series in `MetricsCollector`.
- Provider regression test framework, gated by `OPTICORE_TEST_LIVE=1`,
  skipped cleanly in CI; optional manual live-provider GitHub workflow.
- Deterministic quality regression dataset (`tests/evaluation/test_cases.json`)
  plus baseline-vs-optimized and restoration tests.
- Dashboard distinguishes REAL measured data from DEMO (no fabricated claims).
- `docs/production_readiness.md`, `docs/production_checklist.md`.

## Next (needed to leave "Alpha")

1. **Live-provider validation per release** — run `OPTICORE_TEST_LIVE=1` against
   at least one OpenAI-compatible + one Ollama endpoint on a real model; record
   the matrix row in `docs/production_readiness.md` with measured numbers.
2. **SSRF guardrails for custom `base_url`** — opt-in allow-list of endpoints;
   forbid non-http(s) schemes; document default behavior.
3. **Docker/deployment maturity** — compose file with a disk-cache volume,
   healthcheck, and a non-root user.
4. **Routing capability checks** — validate provider `models()` + context
   limits before a routing decision is accepted.
5. **Mock-based unit tests for SDK paths** — Ollama HTTP/health paths are now
   covered (`tests/test_ollama_health.py`); openai/huggingface request
   building and error mapping without network remain (raises CI coverage
   meaningfully).

## Later

- Redis cache backend behind `BaseCache` (no forced dependency).
- Response-level quality gate in the CLI/`AIClient` (needs a judge/similarity
  model; keep out of the default deterministic path).
- Embedding/proxy evaluator plugins for the quality gate.
- Dashboard live metrics endpoint feeding the existing viewer (real, not demo).
- Release automation (tag → build → publish to PyPI + GitHub release).

## Optional / experiments

- Model-in-the-loop prompt rewrite optimizer (explicitly experimental;
  would change `optimization_overhead_tokens` today fixed at 0).
- Relevance-based context selection (research, not heuristic).
- ONNX Runtime inference backends.
- Real batched inference executor.

## Explicitly NOT planned

- Vendor price catalogs baked into the package (they go stale; pricing stays
  user-configured and labeled `estimated`).
- Unsupported SSRF-style open endpoint forwarding by default.
- Any claim of measured improvement without an included `--json` benchmark
  result from a real provider.
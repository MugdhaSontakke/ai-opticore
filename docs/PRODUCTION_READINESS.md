# AI-OptiCore — Production Readiness Report

> Status: **Internal audit** — date: 2026-09-16
> Evidence basis: actual repository contents, `pytest` (75 passing), `ruff`,
> `mypy`, and a `pytest --cov` run on commit `1546da0`.

This report is a factual assessment of what actually exists in the repository.
It does not assume that a feature works because a file or interface exists.

---

## CORE FUNCTIONALITY

**Status:** Implemented (foundation) / Partial (data-model unification)

**Evidence**

- `OptimizerRequest` (`src/opticore/core/interfaces.py`) is the normalized
  request object flowing through the pipeline. Optimizers receive a copy and
  never mutate caller data (`test_pipeline_result_immutability`).
- `OptimizationPipeline` (`src/opticore/core/pipeline.py`) runs optimizers in
  ascending `order`, respects per-optimizer enable flags, preserves the
  original request, and aggregates token savings.
- `OptimizerResult` captures original/optimized requests and token counts, but
  there is **no per-optimization audit trail** (no `changes` list) and the
  request model lacks `temperature`, `tools`, and `namespace` fields.
- The request/response model is **split across two structures**: optimizers
  use `OptimizerRequest`; providers use plain `dict` requests and return
  `ProviderResponse`. There is no single provider-independent `AIRequest` /
  `AIResponse` pair.

**Problems**

- Two overlapping request representations (optimizer-level and provider-level)
  make the public API larger than it needs to be.
- No unified audit trail describing what each optimization changed.
- `OptimizerResult` does not carry a documented `metrics` dict with canonical
  token keys (`original_input_tokens`, `optimized_input_tokens`, `tokens_saved`,
  `reduction_percentage`).

**Required work**

1. Add `AIRequest` / `AIResponse` as the canonical, provider-independent data
   model; keep current names as aliases for backward compatibility.
2. Add `changes` (audit trail) and canonical `metrics` to `OptimizerResult`.
3. Add optional `temperature`, `tools`, `namespace` fields that optimizers
   preserve verbatim.

**Production risk:** Medium

---

## OPTIMIZATION

**Status:** Implemented (SAFE / BALANCED / AGGRESSIVE prompt + context)

**Evidence**

- `PromptOptimizer` (`src/opticore/optimizers/prompt.py`): SAFE does lossless
  whitespace collapse + repeated-line removal; BALANCED additionally dedupes
  verbatim sentences; AGGRESSIVE additionally dedupes paragraphs and reports a
  severity hint. Behavior is driven by `SafetyMode` and is covered by tests.
- `ContextOptimizer` (`src/opticore/optimizers/context.py`): duplicate-message
  removal, oldest-first truncation under `max_token_budget` with
  `prioritize_recent`, conservative system-prompt truncation outside SAFE mode.
  Coverage of the module is only **51%**.
- `TokenOptimizer` (`src/opticore/optimizers/token.py`) is a measurement step.

**Problems**

- There is **no quality gate / fallback** in the pipeline: an optimizer's result
  is never validated, so an aggressive rewrite is unconditionally accepted.
- "Relevance-based selection" for context is not implemented; only recency +
  token budget are used. (README should not claim relevance selection.)
- No tool/function/structured-output schema protection guarantees (messages are
  typed `list[dict[str, str]]`).

**Required work**

1. Route pipeline output through a quality gate; on failure, fall back to the
   original request.
2. Ensure tool schemas and structured-schema material carried in
   request fields are preserved verbatim (add explicit tests).
3. Correct README claims about relevance-based selection.

**Production risk:** Medium

---

## QUALITY / EVALUATION

**Status:** Partial (interfaces exist, not wired into the pipeline)

**Evidence**

- `src/opticore/evaluation/evaluator.py` provides `SimilarityEvaluator`,
  `CharacterSimilarity`, `TokenJaccardSimilarity`, `ResponseEvaluator`,
  `DefaultResponseEvaluator`, `QualityGate`, `TaskSuccessFunc`. Tests cover the
  gate's pass/warn behavior.
- **The evaluators are not connected to the optimization pipeline or
  `AIClient`.** Nothing measures whether an optimization preserved request
  semantics before the request is sent to a provider.

**Problems**

- The system currently cannot claim "quality preserved" — it never measures it
  in the default path.
- `MinimumScore` / `reject_on_failure` style configuration does not exist.

**Required work**

1. Add config: `quality.enabled`, `quality.minimum_score`,
   `quality.reject_on_failure`.
2. Wire a request-level quality check into the pipeline; emit an explicit
   "quality not verified" note when no evaluator is configured.

**Production risk:** High (honesty requirement)

---

## CACHE

**Status:** Implemented (memory + semantic), Partial (isolation policy)

**Evidence**

- `MemoryCache` (`src/opticore/cache/base.py`): thread-safe, TTL, LRU-style
  eviction by oldest, delete/clear/stats. TTL expiry tested.
- `SemanticCache` (`src/opticore/cache/semantic.py`): exact-match short-circuit,
  cosine-similarity semantic lookup, threshold, TTL, lazy expiry, model-aware
  over `_derive_key(text, model)`. Module coverage **73%**.
- `AIClient` uses a MemoryCache when no embedding function is provided and a
  SemanticCache when an embedding function is given.

**Problems**

- Semantic keys do **not** isolate `system`, `temperature`, `max_tokens`, or a
  `namespace`. A cached response computed for one system prompt / temperature
  can be served for another.
- `AIClient._store_response` swallows all cache write errors with `except
  Exception: pass` (silent failure).
- The `cache` CLI command is a stub: `--clear` prints "cache cleared" but clears
  nothing, and there is no `CachePolicy` abstraction for time-sensitivity.

**Required work**

1. Include `system`, `temperature`, `max_tokens`, `namespace` in cache keys and
   semantic derivation.
2. Replace silent swallowing of cache errors with warning logging; introduce
   `CacheError`.
3. Add `CacheEntry` TTL policy metadata + a `CachePolicy` (serve-stale policy
   explicitly disabled by default).

**Production risk:** Medium

---

## PROVIDERS

**Status:** Implemented (3 providers), Partial (error taxonomy, timeouts)

**Evidence**

- `BaseProvider` (`src/opticore/providers/base.py`) defines `generate`,
  `models`, `health`, plus `ProviderError`, `ProviderAuthError`,
  `ProviderRateError`. `FakeProviderMixin` provides a deterministic test/branch
  fake.
- `OpenAIProvider` reads the key from the environment, lazy-imports the SDK,
  maps auth/rate errors, sets timeout and max_retries from `ProviderConfig`.
- `OllamaProvider` / `HuggingFaceProvider` follow the same interface.
  Coverage: openai **31%**, ollama **26%**, huggingface **28%** — the real
  network paths are untested (correctly, since tests must not require APIs), but
  no mocked/faked unit tests exist for them either.

**Problems**

- No `ProviderTimeoutError`; timeouts surface as generic SDK exceptions.
- Provider layer error taxonomy is incomplete vs. the target set
  (`ProviderError`, `ProviderTimeoutError`, `UnsupportedModelError`, ...).
- External-API surface has no mocked tests; only `FakeProviderMixin` is tested.

**Required work**

1. Add `ProviderTimeoutError` and map SDK timeout exceptions.
2. Add lazy-import/mock-based unit tests for openai/ollama/huggingface
   ("external API tests optional", "use mocks/fakes").

**Production risk:** Low–Medium

---

## CLI

**Status:** Implemented (7 commands), Partial (error handling)

**Evidence**

- `src/opticore/cli/main.py` covers `init`, `config`, `hardware`, `models`,
  `optimize`, `benchmark`, `cache`, `--version`, `--help`, `--json` outputs.
- `tests/test_cli.py` (11 tests) invokes the CLI as a subprocess and verifies
  init/config/hardware/models/optimize (text + JSON), benchmark (text + JSON),
  help for every command, and unknown-command failure. (Coverage reports 0% for
  `cli/main.py` solely because tests run it in subprocesses.)

**Problems**

- Uncaught exceptions in handlers (e.g., unknown `--provider`) will produce a
  Python traceback and a misleading exit behavior.
- `cache` command is not backed by the actual cache subsystem.
- No `--output` flag to write benchmark JSON to a file.

**Required work**

1. Wrap handler dispatch in a single error boundary → clean `error: ...` on
   stderr + exit code 1.
2. Make `cache` honest (report config/capability, or a real clear).
3. Add `benchmark --output <file>`.

**Production risk:** Low

---

## BENCHMARKING

**Status:** Implemented, Partial (reproducibility metadata, overhead)

**Evidence**

- `BenchmarkRunner` (`src/opticore/benchmarks/runner.py`) measures baseline vs
  optimized input tokens, output tokens, and latency with repeats, reports `N/A`
  for unavailable metrics, and never fabricates numbers. CLI `--json` export
  works (tested).
- Overhead is **not** measured: `optimizer_time_ms`, `model_time_ms`,
  `total_time_ms` do not exist, so a token-saving run that is slower overall is
  invisible.
- Results carry no environment fingerprint (version, Python, OS, timestamp), so
  they are not reproducible.

**Required work**

1. Measure optimizer overhead separately from model latency.
2. Record timestamp, `opticore_version`, `python_version`, OS, model, provider,
   hardware, config, tokenizer in the result.
3. Add `benchmark --output results.json`.

**Production risk:** Medium (an overclaim risk because overhead is currently
reported as pure latency benefit)

---

## HARDWARE

**Status:** Implemented (CPU / CUDA), Experimental (ROCm), honest

**Evidence**

- `src/opticore/hardware/base.py` + facade: `CPUBackend` always available;
  `CUDABackend` detected via PyTorch; `ROCmBackend` detected via
  `torch.version.hip` + device vendor, explicitly marked `experimental` with an
  honest non-result message. Tests assert detection never lies.
- `HardwareBackendUnavailableError` is not defined; nothing raises it.

**Required work**

1. Add `HardwareBackendUnavailableError` for callers that require a backend.
2. Keep ROCm as foundation (detection + docs + tests), ROCm *performance
   optimization* explicitly out of v0.1 scope.

**Production risk:** Low (honest by construction)

---

## SECURITY

**Status:** Basic (no secrets committed, key redaction intent), Partial

**Evidence**

- API keys are read from environment variables only; `.env.example` documents
  this; `.env` is gitignored; tests never require keys.
- `src/opticore/logging.py` provides a `RedactingFilter` and prompt logging is
  off by default (`OPTICORE_LOG_PROMPTS`).

**Problems**

- `RedactingFilter` is effectively a no-op: it only replaces the literal string
  `api_key=<redacted>` with itself; it does not scrub `sk-...`/`api_key=...`
  values from messages.
- No dependency scanning, no secret scanning, no CodeQL in CI.

**Required work**

1. Fix `RedactingFilter` to actually redact sensitive field values.
2. Add dependency scanning (e.g., `pip-audit`) + a secret-scan step in CI.

**Production risk:** Medium

---

## TESTING

**Status:** Good start, Partial (coverage 68%)

**Evidence**

- 75 tests pass on Python 3.9 (project targets 3.11+; tests run with a
  `conftest.py` sys.path shim because the local pip cannot editable-install
  hatchling packages).
- `ruff` clean, `mypy` clean (35 files).
- Coverage total **68%**. Weak modules: `cli/main.py` (0% — subprocess artifact),
  `context.py` (51%), `semantic.py` (73%), `logging.py` (66%), provider
  implementations (~26–31%).
- No property-based tests; several required edge cases are untested
  (empty/long input, invalid config, missing tokenizer, provider timeout,
  cache collision, quality rejection, tool-calling schemas).

**Required work**

1. Add edge-case + fidelity tests (empty, very long, invalid config,
   unsupported model, missing tokenizer, timeout, cache collisions, quality
   rejection, schema preservation).
2. Add Hypothesis property tests (no mutation, determinism, non-negative
   counts, TTL behavior).
3. Enforce a coverage floor in CI (core package).

**Production risk:** Medium

---

## CI/CD

**Status:** Implemented (basic), Missing (build/security)

**Evidence**

- `.github/workflows/ci.yml`: matrix tests on Python 3.11/3.12 (ruff, mypy,
  pytest) plus a CLI smoke job. No paid APIs / GPUs required.

**Problems**

- No job builds the wheel or verifies `pip install ai-opticore` in a clean env.
- No dependency/secret scanning; no coverage gate; no CodeQL.

**Required work**

1. Add a build job (`python -m build` + install wheel into a fresh venv +
   `ai-opticore --version`).
2. Add coverage gate, `pip-audit`, and a secret-scan workflow (optionally
   CodeQL).

**Production risk:** Low

---

## DOCUMENTATION

**Status:** Good, Partial (accuracy)

**Evidence**

- README with quick-start, install, features, benchmarking, providers,
  hardware, roadmap, contributing, license. Contributor guides exist under
  `docs/` (architecture, getting-started, performance benchmark methodology,
  hardware/rocm, creating-an-optimizer).
- pyproject `[project.urls]` point at `ai-opticore/ai-opticore` (wrong org —
  should be `MugdhaSontakke`) and a fake readthedocs URL.
- Some feature-table claims are ahead of reality (e.g., "recent/relevance
  prioritization").

**Required work**

1. Fix URLs; trim overclaims; document quality gate, tokenizer behavior
   (exact/fallback), and experimental status explicitly.
2. Add `docs/exceptions.md` and `docs/configuration.md`.

**Production risk:** Low–Medium

---

## PACKAGING

**Status:** Implemented (metadata), Unverified (build)

**Evidence**

- `pyproject.toml`: hatchling backend, `ai-opticore` console script, extras
  (`openai`, `ollama`, `huggingface`, `semantic`, `benchmark`, `dev`, `all`),
  license/readme/classifiers set, requires-python `>=3.11`.
- `python -m build` has **never been run** locally; no missing extras (`yaml`,
  `redis`); no `pydantic` usage despite being a hard dependency.

**Required work**

1. Verify `python -m build` and wheel install.
2. Add `yaml` / `redis` extras; drop or use `pydantic`.

**Production risk:** Medium (unverified artifact)

---

## OBSERVABILITY

**Status:** Partial

**Evidence**

- `MetricsCollector` (counters + series + custom aggregators) is used in
  `AIClient` and the benchmark runner.
- Logging via `opticore.logging.get_logger`; prompt logging opt-in.

**Problems**

- Optimization overhead is not in metrics; canonical token keys are not used
  consistently; `RedactingFilter` ineffective (see Security).

**Required work**

1. Emit `optimizer_time_ms` / `model_time_ms` / `total_time_ms` from
   `AIClient` and the benchmark.
2. Standardize token metric keys.

**Production risk:** Medium

---

## OPEN-SOURCE READINESS

**Status:** Implemented (scaffolding), Missing (starter issues)

**Evidence**

- README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, LICENSE (MIT),
  `CHANGELOG.md`, 6 issue templates + config, PR template, `docs/`.
- No starter issues exist on the repository; no custom label set configured.

**Required work**

1. Create ~10 meaningful starter issues grounded in the gaps listed above.
2. Add custom labels (optimization, benchmark, research, hardware, AMD, ROCm,
   CUDA, ...).

**Production risk:** Low

---

# Distance from production v0.1.0

**Current stage:** Early Alpha

**Target:** Production-quality v0.1.0

The project is genuinely usable for prompt/context optimization and
benchmarking, and the architecture is sound. It is not yet safe to market as
"quality-preserving" because the quality gate is not wired in, overhead is not
reported, and several error paths are silent.

## What already works

- Pipeline + SAFE/BALANCED/AGGRESSIVE optimizers (tested, deterministic).
- Real token counting via `tiktoken` with a tokenizer abstraction.
- Memory + semantic cache with TTL, eviction, stats (mostly tested).
- Provider interface with fake provider for CI; OpenAI/Ollama/HF stubs wired.
- CLI (7 commands) with subprocess tests.
- Baseline vs optimized benchmark runner with JSON export and `N/A` honesty.
- Hardware detection that never fakes a result (CPU/CUDA/ROCm-experimental).
- Lint + typecheck clean; 75 tests; basic CI.

## What blocks v0.1.0 (critical)

1. Quality gate is not enforced anywhere; no safe fallback to the original
   request. (Honesty + safe-by-default hole.)
2. Optimization overhead is not measured, so benchmark "latency change" can
   mislead.
3. Semantic cache can serve responses across `system`/`temperature` — a
   correctness risk.
4. Ineffective secret redaction in logs; silent `except Exception: pass` in the
   cache path.
5. CLI can print raw tracebacks on unexpected errors.

## What does NOT block v0.1.0 (high priority, but post-contract)

- ROCm performance optimization (foundation only for v0.1).
- Real batched inference executor, quantization, model-routing improvements.
- Redis/disk cache backends.
- Real dashboard live metrics.

## What should be postponed

- SaaS/cloud/platform features, auth, payments, plugin marketplace.
- Dozens of model providers.
- Advanced AMD optimization.

---

## Final verdict

The core is ~halfway to a trustworthy v0.1.0. The blocking items are
**quality-gate enforcement + safe fallback**, **overhead measurement**,
**cache isolation**, and **error/logging honesty** — each an engineering change,
not a research problem. After those land (with tests), the project can honestly
claim "Early production use under supervision" rather than "Early Alpha".
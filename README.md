# AI-OptiCore

> An open-source optimization layer for AI/LLM applications that reduces unnecessary token usage, latency, memory consumption, and inference cost while preserving response quality.

AI-OptiCore sits between your AI application and your model provider. It analyzes every request and passes it through a modular **optimization pipeline** (token analysis, prompt optimization, context management, semantic cache, model routing, inference optimization, hardware backends) before the request reaches the model — then measures and evaluates the result.

It is built from day one for open-source collaboration: every optimizer, provider, and hardware backend is a pluggable component with clear interfaces, tests, and documentation.

---

## Problem statement

LLM applications waste tokens, time, and money on:

- redundant prompt text and duplicated instructions,
- bloated conversational context,

- repeated identical or near-identical requests,
- oversized models handling trivial requests,
- latent device capability that goes unused.

Most apps have **no visibility** into how much of their spend is waste. AI-OptiCore gives you a measurable optimization layer that you can turn on, tune, benchmark, and evaluate — without rewriting your application.

## Why AI-OptiCore exists

- **Measured, not claimed.** Every optimization is real, tested, and benchmarkable. We never fabricate results.
- **Modular.** Add an optimizer, a provider, or a hardware backend without touching the core.
- **Safe.** Different safety modes (SAFE / BALANCED / AGGRESSIVE) control how aggressive transformations are; a quality gate evaluates each optimized request against the original (prompt + conversation history) and falls back to the original on failure.
- **Provider-agnostic.** OpenAI-compatible APIs, Ollama, and local Hugging Face models all implement one provider interface.
- **AMD-ready.** ROCm support is designed as an independent, testable backend from day one — no vendor lock-in.
- **Honest.** Benchmarking reports real measured values including optimizer overhead and cache hit rate — or `N/A` when a metric cannot be measured. Never fabricated.

## Architecture

```mermaid
flowchart TD
    App[Application] --> Core[AI-OptiCore]
    Core --> RA[Request Analysis]
    RA --> Pipe[Optimization Pipeline]
    Pipe --> TO[Token Optimization]
    Pipe --> PO[Prompt Optimization]
    Pipe --> CO[Context Optimization]
    Pipe --> SC[Semantic Cache]
    Pipe --> MR[Model Routing]
    Pipe --> IO[Inference Optimization]
    Pipe --> HB[Hardware Backend]
    MR --> Model[AI Model / Provider]
    IO --> Model
    HB --> Model
    Model --> Resp[Response]
    Resp --> Metrics[Metrics + Evaluation]
```

Optimizers are sequential, independent components:

```mermaid
flowchart LR
    Req[Request] --> P[OptimizerPipeline]
    P --> O1[Optimizer 1]
    O1 --> O2[Optimizer 2]
    O2 --> O3[Optimizer 3]
    O3 --> MP[Model Provider]
    MP --> Resp2[Response]
```

## Features

| Area | Description |
| --- | --- |
| Token analysis | Real tokenizer implementations (`tiktoken`), per-component token counts, reduction % |
| Prompt optimization | Removes obvious redundancy (repeated instructions, whitespace, duplicate context) with configurable safety modes |
| Context optimization | Token budgets, recent-message priority (not true relevance), duplicate removal; never mutates your data |
| Semantic cache | Exact + embedding similarity, configurable threshold, TTL, invalidation, metadata, model/provider/namespace isolation, hit/miss metrics; bounded memory or SQLite disk backend |
| Model routing | Complexity/budget-based selection with availability validation + fallback; disable fully when you don't want it |
| Hardware abstraction | CPU / CUDA / ROCm backends with honest capability detection |
| Inference optimization | Interfaces for batching, quantization, loading, memory tracking (experimental/planned statuses are explicit) |
| Benchmarking | Real measurements, baseline vs optimized, median/p95, net latency change, per-scenario breakdown, `ai-opticore benchmark` |
| Cost | Optional user-configured pricing; costs are always labeled `estimated` and `N/A` when unconfigured |
| Evaluation | Request/response similarity, quality gates, pluggable evaluators; rejection restores the original request |
| Metrics | Unified `MetricsCollector` that is extensible |
| CLI | `init`, `optimize`, `benchmark`, `cache`, `models`, `hardware`, `config`, `doctor` |
| Dashboard | Lightweight React dashboard (see `dashboard/`) |

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/MugdhaSontakke/ai-opticore.git
cd ai-opticore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Minimum install (core, no provider SDKs):

```bash
pip install -e .
```

Optional feature extras:

```bash
pip install -e ".[openai]"      # OpenAI-compatible APIs
pip install -e ".[ollama]"      # local Ollama models
pip install -e ".[huggingface]" # local Hugging Face models
pip install -e ".[semantic]"    # sentence-transformers embeddings
pip install -e ".[yaml]"        # YAML config file support
pip install -e ".[benchmark]"   # psutil, numpy for benchmarks
pip install -e ".[dev]"         # pytest, hypothesis, ruff, mypy, pyyaml
```

Set your API key in the environment (never commit keys):

```bash
cp .env.example .env
# edit .env, then:
set -a && source .env && set +a
```

## Quick start

Optimize a prompt without calling any model:

```python
from opticore import Optimizer

optimizer = Optimizer()

result = optimizer.optimize(
    prompt="Can you please tell me what is   machine learning?",
    system="You are a helpful assistant.",
)

print(result.optimized_prompt)   # cleaned prompt
print(result.tokens_saved)       # real tokens saved
print(result.reduction_percent)
```

Optimize and generate through your provider:

```python
from opticore import AIClient

client = AIClient(
    provider="openai",           # or "ollama", "huggingface", or any BaseProvider
    optimization=True,
)

response = client.generate(prompt="Explain caching in LLM applications.")
print(response.content)
print(response.tokens_saved)
print(response.cache_hit)
print(response.request_id)       # per-request observability
print(response.metadata["routing"])   # when a ModelRouter is attached
print(response.estimated_cost)   # only populated when pricing is configured
```

Turn optimization off for a baseline comparison:

```python
baseline_client = AIClient(provider="openai", optimization=False)
```

Optional disk cache (survives process restarts; SQLite, stdlib-only):

```python
client = AIClient(
    provider="ollama",
    config=OptimizationConfig(cache_backend="disk", cache_disk_path="/var/cache/opticore.sqlite"),
)
```

## Benchmarking

```bash
ai-opticore benchmark                    # uses fake provider; safe for CI
ai-opticore benchmark --provider openai  # real provider (needs OPENAI_API_KEY)
ai-opticore benchmark --provider ollama
ai-opticore benchmark --samples 10 --repeats 3
```

Output contains **real measured values** (or `N/A` when a metric cannot be measured). It reports
baseline vs optimized tokens and latency, median and p95 latency, the net latency
change (including when optimization makes a workload *slower*), optimizer overhead vs
provider latency split, cache hit rate, per-scenario breakdowns, and estimated cost
only when you configure pricing:

```yaml
# opticore.yaml
pricing:
  input_per_1k: 0.005   # USD per 1k input tokens (example, user-supplied)
  output_per_1k: 0.015
```

> Cost figures are always labeled `estimated` — AI-OptiCore never bakes in vendor
> prices that can go stale, and never reports a cost figure when pricing is not
> configured.

> AI-OptiCore provides tools for measuring and reducing unnecessary token usage. We do not publish fixed reduction claims — run your own benchmarks on your own workloads.

## Configuration & CLI reference

Configuration precedence: **defaults → `opticore.yaml` (or `--config`) → `OPTICORE_*` environment variables → explicit API arguments**.

```bash
ai-opticore init                 # write a starting opticore.yaml (never contains keys)
ai-opticore config               # show the effective configuration
ai-opticore config validate      # exit 0 if valid, non-zero + reason if not
ai-opticore config --json        # machine-readable config output
ai-opticore cache explain        # describe how caching behaves for this setup
ai-opticore cache stats          # hits/misses/evictions for the on-disk cache
ai-opticore cache stats --json
ai-opticore models               # providers and (for Ollama) installed models
ai-opticore doctor --provider ollama --model llama3.2   # connectivity diagnostics
```

`doctor` runs staged environment checks (package, config, then provider
connectivity) and answers, for Ollama: *is the server reachable, which models
are installed, is the requested model present, and can a small generation
complete?* It exits non-zero when a critical check fails and reports distinct
reasons for an offline server vs a missing model. `python -m opticore` is an
alias for the `ai-opticore` CLI.

Provider endpoints and credentials come from the environment (see
[`.env.example`](.env.example)):

| Variable | Used by | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI-compatible provider | — (required) |
| `config.base_url` | OpenAI-compatible provider (`AIClient(provider=..., config=ProviderConfig(base_url=...))`; e.g. Azure/vLLM/llama.cpp servers) | OpenAI API |
| `OLLAMA_BASE_URL` | Ollama provider | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama provider default model | `llama3.2` |

Provider URLs are validated before any connection (SSRF guardrails; see
[`SECURITY.md`](SECURITY.md)). A URL pointing at cloud-metadata or link-local
addresses is rejected outright; local model servers like Ollama work because
private/loopback networks are enabled by default. If you accept provider URLs
from untrusted users, disable `allow_private_networks` or set `allowed_hosts`.

Benchmarking accepts your own scenario file:

```bash
ai-opticore benchmark --dataset benchmarks/data/fixtures.json --json
# dataset schema: list of {"prompt", "system"?, "messages"?, "tools"?,
# "category"? (→ scenario tag), "id"?, "expected_keywords"?}
```

## Docker

A non-root, minimal image with a built-in healthcheck. The container entry
point is the CLI, so run one-off commands with `docker compose run`:

```bash
docker build -t ai-opticore .
docker compose run --rm app config validate
docker compose run --rm app optimize --prompt "Hello    World"
docker compose run --rm app benchmark --provider openai --model gpt-4o-mini
```

An optional `ollama` service is declared in `docker-compose.yml` — start it
only if you want a local model endpoint:

```bash
docker compose up -d ollama
docker compose run --rm app benchmark --provider ollama --model <model>
```

Secrets are injected via environment (`OPENAI_API_KEY=${OPENAI_API_KEY:-}`);
nothing is baked into the image.

## Provider regression testing

```bash
# Run real-endpoint regression tests (optional/opt-in; skips cleanly in CI):
OPTICORE_TEST_LIVE=1 \
OPTICORE_TEST_OPENAI_API_KEY=sk-... \
pytest -q tests/test_providers_live.py
```

See `tests/test_providers_live.py` for all supported variables (OpenAI-compatible,
Ollama), plus the manual `provider-live` GitHub workflow.

## Real LLM evaluation with Ollama

The fastest way to exercise the full pipeline end-to-end against a **real local
LLM** is Ollama + Llama 3.2. Install Ollama (https://ollama.com), then:

```bash
# 1. Start the server and pull a model (one-time)
ollama serve &                 # or: brew services start ollama
ollama pull llama3.2

# 2. Verify connectivity: reachable? models installed? generation probe OK?
ai-opticore doctor --provider ollama --model llama3.2

# 3. Single real request through the pipeline (optimized prompt + cache MISS→HIT)
python examples/ollama_demo.py

# 4. Full benchmark against a real-LLM dataset (26 prompts, 9 categories)
ai-opticore benchmark \
  --provider ollama --model llama3.2 \
  --dataset benchmarks/data/real_llm_dataset.json \
  --repeats 2 --json          # add --output report.json to save

# 5. Live-regression tests (opt-in; real network, skipped in CI)
OPTICORE_TEST_LIVE=1 OPTICORE_TEST_OLLAMA_URL=http://localhost:11434 \
  pytest -q tests/test_providers_live.py
```

What the report tells you:

- **Token reduction** — measured prompt tokens before vs after optimization.
- **Latency** — baseline vs optimized median and p95, plus optimizer overhead
  split out from provider latency; a *slower* net result is reported honestly.
- **Cache hit rate** — repeated requests are served from cache (MISS then HIT).
- **Quality** — a HEURISTIC character/token similarity of the baseline vs
  optimized *responses* (`quality_response_similarity_avg`). It is
  deterministic and dependency-free, and it is explicitly **not** a
  model-based judge; an optional judge/evaluator is a planned milestone.
- **Cost** — `N/A` for a local Ollama endpoint (no per-token billing).
- **Provider info** — Ollama version, model, installed-model count.
- **Tokens/sec** — generation throughput on responses with real measurable
  latency.

AI-OptiCore does **not** train or fine-tune an LLM. It optimizes the *text you
send* to a model you already run; the Ollama integration here is provider
connectivity plus measurement, not model training.

> Full guide: [`docs/ollama.md`](docs/ollama.md).

## Supported providers

Register an existing provider or add your own.

| Provider | Module | Notes |
| --- | --- | --- |
| OpenAI-compatible | `providers.openai` | Also Azure/vLLM/llama.cpp server via `base_url` |
| Ollama | `providers.ollama` | Local REST API |
| Hugging Face (local) | `providers.huggingface` | `transformers` |

## Hardware support

- **CPU** — always available.
- **CUDA** — detected via PyTorch.
- **ROCm / AMD** — detected via `torch.version.hip` + device vendor. Status: experimental. See [`docs/hardware/rocm.md`](docs/hardware/rocm.md) for tested configurations and known limitations.

Detection never fakes results. If a backend is unavailable, a clear capability message is returned.

## Roadmap

- [x] Initial hardening toward v0.1 (quality gate with safe fallback, overhead measurement, cache isolation, secret-safe logging)
- [x] v0.1 hardening pass 2 (real provider analytics, per-step timing, honest timing/caching, `ai-opticore` exit codes)
- [x] v0.1.0 — Disk cache backend (SQLite), estimated-cost accounting, per-scenario benchmarks, provider regression test framework, observability (request IDs, routing/quality/fallback metadata), dashboard REAL-vs-DEMO labeling
- [x] v0.2.0 — SSRF guardrails (`validate_base_url`), provider retry policy with typed error mapping, request redaction, config `validate`, cache `stats`, benchmark `--dataset`, dashboard DATA UNAVAILABLE state + median/p95/overhead/cost rendering, Docker (non-root image, compose with optional Ollama), `docs/api_stability.md`
- [x] v0.3.0 — Real end-to-end LLM evaluation with Ollama + Llama 3.2: live `doctor` health probe (`ai-opticore doctor`), `python -m opticore`, response-quality heuristic vs optimized responses, tokens/sec, provider info in reports, 26-prompt real-LLM dataset, `examples/ollama_demo.py`, mock-tested Ollama HTTP paths, live-opt-in regression tests
- [ ] v0.2 — Live-provider matrix validated per release; mock-based unit tests for provider SDK request/error paths
- [ ] v0.3 — Redis cache backend; real batched inference executor
- [ ] v0.3 — Response-level quality gate with a judge/similarity model (optional path)
- [ ] v0.4 — Dashboard with live metrics (see `dashboard/`)
- [ ] v0.4 — ONNX Runtime hardware backends

Full plan: [`docs/next_roadmap.md`](docs/next_roadmap.md) · Scorecard:
[`docs/production_checklist.md`](docs/production_checklist.md) · Matrix:
[`docs/production_readiness.md`](docs/production_readiness.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome: optimizers, providers, hardware backends, benchmarks, docs, bug fixes.

## License

MIT — see [LICENSE](LICENSE).
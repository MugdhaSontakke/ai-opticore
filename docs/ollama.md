# Real LLM evaluation with Ollama + Llama 3.2

This guide walks through running the AI-OptiCore optimization layer against a
**real local LLM** (Ollama + Llama 3.2) end-to-end: diagnostics, a single
request through the pipeline, a full benchmark over a categorized dataset, and
opt-in live regression tests.

## 1. Install and start Ollama

- Install Ollama from https://ollama.com.
- Start the server (one of):
  - `ollama serve` in a terminal, or
  - `brew services start ollama` (macOS service).
- Pull the model (one-time):

```bash
ollama pull llama3.2
```

Verify the server is up:

```bash
curl -s http://localhost:11434/api/tags
```

## 2. Install AI-OptiCore with the Ollama extra

```bash
pip install -e ".[ollama]"
```

The `ollama` extra installs the official Python SDK. AI-OptiCore **also works
without it**: the provider falls back to the REST API via `requests`
(Python 3.9+ with the core install already pulls `requests`).

## 3. Check connectivity — `ai-opticore doctor`

The `doctor` command runs staged checks and answers, for Ollama:

1. Is the server **reachable**? (`GET /api/version` → reports the version)
2. Which **models are installed**? (`GET /api/tags`)
3. Is the **requested model** installed?
4. Can a small **generation** complete? (tiny `/api/generate` probe;
   disabled with `--no-probe`)

```bash
ai-opticore doctor --provider ollama --model llama3.2
ai-opticore doctor --provider ollama --json     # machine-readable
```

Exit code is `0` only when every critical check passes. Distinct failure kinds:

| Problem | `error_kind` |
| --- | --- |
| Ollama not installed / server not running | `not_running` |
| Request timed out | `timeout` |
| Server returned an error (e.g. 404/500) | `http:<status>` |
| Unparseable response body | `malformed_response` |
| Server is up but the model is not pulled | `model_not_found` |

The `health()` API powering `doctor` never raises for an offline server; it
returns the structured report so your own tooling can render the same checks.

## 4. Single real request — `examples/ollama_demo.py`

```bash
python examples/ollama_demo.py
```

Runs three prompts through:

- **MODE A (BASELINE)** — direct provider call: no optimization, no cache reuse.
- **MODE B (AI-OPTICORE)** — optimized prompt + cache: request 1 is a cache
  MISS (real generation), the identical request 2 is a HIT (no generation,
  ~zero latency).

If the server or model is missing you get a typed message with the exact fix
(`ollama serve` / `ollama pull llama3.2`) instead of a stack trace.

## 5. Full benchmark — real-LLM dataset

```bash
ai-opticore benchmark \
  --provider ollama \
  --model llama3.2 \
  --dataset benchmarks/data/real_llm_dataset.json \
  --repeats 2 \
  --json --output report.json
```

`benchmarks/data/real_llm_dataset.json` has 26 deterministic prompts across
9 categories: simple factual, summarization, rewriting, coding, explanation,
structured output, long context, multi-part, and repetitive. The schema is the
standard `--dataset` schema (list of `{prompt, system?, category?, id?,
expected_keywords?}`), so any category maps to a per-scenario report row.

What the report contains (all measured, or explicitly `N/A`):

- **Token reduction** (%), baseline vs optimized input tokens.
- **Latency** — baseline vs optimized avg / median / p95; optimizer overhead
  split from provider latency; a *slower* net result is reported honestly.
- **Cache hit rate** — percentage of optimized requests served from cache.
- **Quality** — `quality_response_similarity_avg` and
  `quality_response_token_jaccard_avg`: a **heuristic** character/token
  similarity between the baseline and optimized *responses*. This is
  deterministic and dependency-free, and it is **not** a model-based judge.
- **Tokens/sec** — generation throughput on samples with real measurable
  latency (`tokens_per_sec_baseline` / `tokens_per_sec_optimized`).
- **Provider info** — Ollama version, endpoint, installed-model count,
  requested model, reachability.
- **Cost** — `N/A` for a local Ollama endpoint (no per-token billing).

## 6. Live regression tests (opt-in)

These hit the real endpoint and are skipped unless you opt in, so CI stays
hermetic:

```bash
OPTICORE_TEST_LIVE=1 \
OPTICORE_TEST_OLLAMA_URL=http://localhost:11434 \
OPTICORE_TEST_OLLAMA_MODEL=llama3.2 \
pytest -q tests/test_providers_live.py
```

Covers: response availability, token measurement, cache MISS→HIT, routing
observability, typed auth failures, and the `health()` round-trip.

## 7. Environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | Ollama endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model when none is passed | `llama3.2` |

Absolute CLI precedence: explicit `--model` > `ProviderConfig.model` >
`OLLAMA_MODEL` > `llama3.2`.

## FAQ

**Does AI-OptiCore train or fine-tune Llama 3.2?**
No. It optimizes the *text* you send to a model you already run, and measures
the result. No model weights are modified or created.

**Why is cost `N/A`?**
A local Ollama server has no per-token billing. Cost estimation only appears
when you configure `pricing.input_per_1k` / `output_per_1k`, and it is always
labeled `estimated`.

**Why is the quality score labeled heuristic?**
`character_similarity` / `token_jaccard` are deterministic string metrics. They
detect *drastic* meaning drift cheaply but are not a semantic judge. Wiring an
embedding-based or LLM judge is a planned milestone
(`docs/next_roadmap.md`); until then the report tells you exactly what was
measured and never implies it is model-based QA.
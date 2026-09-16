# Architecture

## Overview

AI-OptiCore is a layer between an application and an AI model/provider:

```text
Application
    |
    v
AI-OptiCore
    |
    v
Request Analysis
    |
    v
Optimization Pipeline
    |--- Token Optimization
    |--- Context Optimization
    |--- Semantic Cache
    |--- Model Routing
    |--- Inference Optimization
    |--- Hardware Backend
    |
    v
AI Model / Provider
    |
    v
Response
    |
    v
Metrics + Evaluation
```

## The pipeline

The heart of the system is `OptimizationPipeline` (`core/pipeline.py`). It
accepts an `OptimizerRequest`, runs each registered optimizer in ascending
`order`, threads the output request forward, and aggregates token savings.

```python
from opticore import OptimizationPipeline
from opticore.optimizers.token import TokenOptimizer
from opticore.optimizers.prompt import PromptOptimizer

pipeline = OptimizationPipeline([
    TokenOptimizer(),
    PromptOptimizer(),
])
```

Rules:

- Optimizers receive the *previous* optimizer's output request.
- Optimizers **never mutate** the input. Always return a new
  `OptimizerRequest`.
- Every optimizer reports original/optimized token counts so the pipeline can
  aggregate `tokens_saved` and `reduction_percent`.
- Each optimizer can be disabled via a config flag
  (e.g. `enable_prompt_optimization: false`).

## Core data types

| Type | Purpose |
| --- | --- |
| `OptimizerRequest` | The normalized request (prompt, system, messages, max_tokens, model, metadata) |
| `OptimizerResult` | Outcome of an optimizer or the full pipeline (both requests, token counts) |
| `OptimizationConfig` | Safety mode, token budgets, per-optimizer enable flags |
| `ProviderResponse` | Normalized provider output (content, tokens, latency, metadata) |

## Plugin system (optimizers)

All optimizers subclass `BaseOptimizer` in `optimizers/base.py`. A new
optimizer:

1. Implements `optimize(request, config) -> OptimizerResult`.
2. Sets a unique `name` and pipeline `order`.
3. Optionally maps to a config flag for enable/disable.

See `docs/developers/creating-an-optimizer.md`.

## Provider system

All providers implement `BaseProvider` (`providers/base.py`) and return a
normalized `ProviderResponse`. SDKs are imported lazily so the core package
stays importable without any optional provider SDK installed. API keys are
read from environment variables only.

```python
from opticore.providers.base import BaseProvider, ProviderResponse

class MyProvider(BaseProvider):
    name = "myprovider"

    def generate(self, request: dict) -> ProviderResponse: ...
```

New providers are registered in `providers/__init__.py` and appear in the
`models`/`benchmark` CLI commands.

## Cache subsystem

Two cache types:

- `MemoryCache` — exact-match thread-safe in-process cache with TTL,
  eviction, and stats.
- `SemanticCache` — exact-match plus embedding similarity lookup with a
  configurable threshold, TTL, and stats. Degrades gracefully to exact-match
  when no embedding function is provided.

Both implement `BaseCache`. The semantic cache never assumes similar questions
have identical answers; the threshold is configurable with a conservative
default (0.90).

## Hardware abstraction

`BaseHardwareBackend` defines `available()`, `capabilities()`, and
`details()`. Backends report honestly; if detection cannot confirm hardware,
`available()` returns `False` and `details()` explains why.

## Metrics

`MetricsCollector` tracks counters and value series with an extensible custom
aggregator API. The pipeline and the `AIClient` record into collectors that
can be merged and rendered in benchmarks.

## Modules

```
src/opticore/
  core/             pipeline, config, interfaces
  optimizers/       token, prompt, context optimizers
  cache/            memory + semantic cache
  providers/        openai, ollama, huggingface
  routing/          model router
  inference/        batching, quantization, runtime
  hardware/         cpu, cuda, rocm backends
  benchmarks/       runner + metrics
  evaluation/       quality/safety evaluation
  cli/              command-line interface
```
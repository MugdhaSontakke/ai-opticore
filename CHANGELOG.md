# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MVP module architecture: pipeline, optimizers, cache, providers, routing,
  hardware, inference, benchmarks, evaluation, metrics, CLI.
- Token analysis with real tokenizer implementations (`tiktoken` default,
  custom counters supported).
- Prompt optimizer with SAFE / BALANCED / AGGRESSIVE safety modes.
- Context optimizer with token budgets and duplicate-message removal.
- Memory cache and semantic cache (exact + embedding similarity, TTL,
  metadata, stats).
- Model provider interface with OpenAI-compatible, Ollama, and Hugging Face
  providers (SDKs lazy-loaded).
- Model router with configurable rules and a disable switch.
- Hardware abstraction with CPU / CUDA / ROCm (experimental) backends and
  honest detection.
- Inference interfaces: batching (experimental), quantization (planned),
  memory tracking.
- Real benchmark runner (`ai-opticore benchmark`), metrics collection, and
  evaluation/quality gates.
- CLI commands: `init`, `optimize`, `benchmark`, `cache`, `models`,
  `hardware`, `config` — each with `--help`.
- Python API: `Optimizer` and `AIClient`.
- 75 runtime tests; linting (Ruff) and mypy clean.

### Security

- Redacting log filter for sensitive fields.
- Prompt/content logging disabled by default.

## [0.1.0] - 2026-09-16

### Added

- Project scaffolding: pyproject.toml, package layout, docs, GitHub
  templates, CI workflow, LICENSE, SECURITY.md, CODE_OF_CONDUCT.md.
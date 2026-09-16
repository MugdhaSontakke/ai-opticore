# Contributing to AI-OptiCore

Thanks for wanting to contribute. AI-OptiCore is designed for community collaboration. All code, docs, tests, and benchmarks are welcome.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Project layout](#project-layout)
- [Environment setup](#environment-setup)
- [Installation](#installation)
- [Running tests](#running-tests)
- [Formatting and linting](#formatting-and-linting)
- [Type checking](#type-checking)
- [Creating a branch](#creating-a-branch)
- [Submitting a PR](#submitting-a-pr)
- [Writing tests](#writing-tests)
- [Adding a new optimizer](#adding-a-new-optimizer)
- [Adding a provider](#adding-a-provider)
- [Adding a hardware backend](#adding-a-hardware-backend)
- [Benchmarks](#benchmarks)

## Code of conduct

Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Be respectful, constructive, and welcoming.

## Project layout

```text
src/opticore/
  core/         pipeline, config, interfaces
  optimizers/   token, prompt, context optimizers
  cache/        memory + semantic cache
  providers/    openai, ollama, huggingface
  routing/      model router
  inference/    batching, quantization, runtime
  hardware/     cpu, cuda, rocm backends
  benchmarks/   runner + metrics
  evaluation/   quality/safety evaluation
  cli/          command-line interface
tests/          pytest suite (no paid APIs required)
benchmarks/     benchmark scripts
examples/       usage examples
docs/           architecture, performance, hardware docs
dashboard/      React dashboard (WIP)
```

## Environment setup

Requirements: Python 3.11+, git.

```bash
git clone https://github.com/ai-opticore/ai-opticore.git
cd ai-opticore
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Installation

Development install from the repository root:

```bash
pip install -e ".[all]"
```

Optional extras are documented in the README.

## Running tests

Tests never require a paid API:

```bash
pytest
```

Run a single file or test:

```bash
pytest tests/test_cache.py
pytest tests/test_token.py::test_tokenizer_counts_with_real_tokenizer -vv
```

## Formatting and linting

We use Ruff:

```bash
ruff check src/ tests/
ruff check src/ tests/ --fix
```

Code style: line length 100, type hints everywhere, docstrings on public
interfaces, meaningful variable names, no giant functions.

## Type checking

We use mypy (best-effort):

```bash
mypy src/opticore --ignore-missing-imports
```

## Creating a branch

```bash
git checkout -b feature/your-descriptive-name
```

Use prefixes: `feature/`, `fix/`, `docs/`, `bench/`, `refactor/`.

## Submitting a PR

1. Make changes on a branch.
2. Add tests for new behavior.
3. Run `pytest`, `ruff`, and `mypy`.
4. Commit with a descriptive message.
5. Push and open a pull request.
6. Fill out the PR template.

Review checklist:

- [ ] Tests added/updated for new behavior
- [ ] `pytest` passes
- [ ] `ruff check` passes
- [ ] `mypy` (best-effort) passes
- [ ] No secrets, keys, or fabricated benchmark numbers
- [ ] Docs updated when user-facing behavior changes

## Writing tests

Put tests in `tests/` mirroring the module layout. Use fakes/mocks for
external providers — never require a real API key in the default suite.

```python
def test_something() -> None:
    assert 1 + 1 == 2
```

## Adding a new optimizer

An optimizer is anything that transforms an `OptimizerRequest` and reports
token impact. Subclass `BaseOptimizer`:

```python
""".../src/opticore/optimizers/custom.py"""

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.optimizers.base import BaseOptimizer
from opticore.optimizers.token import Tokenizer, count_tokens


class CustomOptimizer(BaseOptimizer):

    name = "custom"       # used for enable flags and logs
    order = 50            # ascending order within the pipeline

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()

    def optimize(self, request, config):
        original = count_tokens(request, self.tokenizer)
        new_request = OptimizerRequest(
            prompt=transform(request.prompt),
            system=request.system,
            messages=request.messages,
            max_tokens=request.max_tokens,
            model=request.model,
            metadata=dict(request.metadata),
        )
        optimized = count_tokens(new_request, self.tokenizer)
        return OptimizerResult(...)  # never mutate request
```

Then register it in the pipeline (see `core/pipeline.py`), add a config flag,
and add unit tests. Read the existing optimizers for the exact contract —
especially: **never mutate the input request; return a new object.**

## Adding a provider

Subclass `BaseProvider` in `src/opticore/providers/`, then register it in
`providers/__init__.py`.

Key contract:

```python
from opticore.providers.base import BaseProvider, ProviderResponse

class MyProvider(BaseProvider):
    name = "myprovider"

    def generate(self, request: dict) -> ProviderResponse:
        ...
        return ProviderResponse(
            content=...,
            model=...,
            provider=self.name,
            input_tokens=...,
            output_tokens=...,
            latency_ms=...,
        )
```

- Never store API keys in code; read them from environment variables.
- Lazy-import third-party SDKs so the core stays importable without them.
- Add a fake/mocked unit test plus an optional integration test marked
  `@pytest.mark.integration` (skipped by default).

## Adding a hardware backend

Subclass `BaseHardwareBackend` in `src/opticore/hardware/` and register the
instance in `hardware/__init__.py`.

- `available()` must only return `True` when the hardware was actually
  detected in this environment.
- `capabilities()` describes what the backend supports.
- Never claim a capability you cannot verify. If a check cannot run, report
  how in `details()`.

Write tests with mocks/stubs when the hardware is not present. See
`docs/hardware/rocm.md` for the AMD/ROCm reference.

## Benchmarks

`ai-opticore benchmark` measures **real** values. Never commit fabricated
benchmark results. If a metric cannot be measured it renders as `N/A`. See
`docs/performance/benchmark-methodology.md`.

## Questions

Open an issue with the `help wanted` or `good first issue` label. See the
issue templates in `.github/ISSUE_TEMPLATE/`.
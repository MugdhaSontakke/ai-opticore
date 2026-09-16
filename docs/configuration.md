# Configuration

AI-OptiCore supports three configuration layers, applied in order:

1. Built-in defaults
2. Config file (YAML or JSON) passed to `load_config` / the CLI
3. `OPTICORE_*` environment variable overrides

Every value is validated eagerly: invalid input raises
`opticore.exceptions.ConfigurationError` at load time rather than failing
later at runtime.

## Python API

```python
from opticore import OptimizationConfig

config = OptimizationConfig(
    safety_mode="safe",                 # safe | balanced | aggressive
    max_token_budget=4096,
    enable_token_optimization=True,
    enable_prompt_optimization=True,
    enable_context_optimization=True,
    enable_semantic_cache=True,
    enable_model_routing=True,
    semantic_cache_threshold=0.90,
    cache_ttl_seconds=3600.0,
    prioritize_recent=True,
    quality_enabled=True,
    quality_minimum_score=0.85,
    quality_reject_on_failure=False,
    quality_evaluator="character",
    log_prompts=False,
)

from opticore.core.config import load_config  # same loader used by the CLI
```

## Config file

`ai-opticore init` writes an example `opticore.yaml` in the current
directory. A tracked `opticore.yaml` is the default config file for the
CLI.

```yaml
safety_mode: safe
max_token_budget: 4096
enable_token_optimization: true
enable_prompt_optimization: true
enable_context_optimization: true
enable_semantic_cache: true
enable_model_routing: true
semantic_cache_threshold: 0.90
cache_ttl_seconds: 3600
prioritize_recent: true
quality:
  enabled: true
  minimum_score: 0.85
  reject_on_failure: false
  evaluator: character
log_prompts: false
```

Notes:

- The `quality` block is flattened into `quality_enabled`,
  `quality_minimum_score`, `quality_reject_on_failure`, and
  `quality_evaluator`.
- JSON config files are accepted too (`.json`); anything else is rejected.
- Reading YAML requires the optional `yaml` extra:
  `pip install "ai-opticore[yaml]"`.
- **Secrets are never allowed in config files.** Keys whose names look
  sensitive (`api_key`, `token`, `secret`, `password`, ...) cause
  `ConfigurationError`. API keys must come from environment variables.

## Environment overrides

Environment overrides are applied on top of the file (or defaults):

| Variable | Type | Maps to |
| --- | --- | --- |
| `OPTICORE_SAFETY_MODE` | str | `safety_mode` |
| `OPTICORE_MAX_TOKEN_BUDGET` | int | `max_token_budget` |
| `OPTICORE_SEMANTIC_CACHE_THRESHOLD` | float | `semantic_cache_threshold` |
| `OPTICORE_CACHE_TTL_SECONDS` | float | `cache_ttl_seconds` |
| `OPTICORE_QUALITY_MINIMUM_SCORE` | float | `quality_minimum_score` |
| `OPTICORE_ENABLE_SEMANTIC_CACHE` | bool | `enable_semantic_cache` |
| `OPTICORE_ENABLE_TOKEN_OPTIMIZATION` | bool | `enable_token_optimization` |
| `OPTICORE_ENABLE_PROMPT_OPTIMIZATION` | bool | `enable_prompt_optimization` |
| `OPTICORE_ENABLE_CONTEXT_OPTIMIZATION` | bool | `enable_context_optimization` |
| `OPTICORE_ENABLE_MODEL_ROUTING` | bool | `enable_model_routing` |
| `OPTICORE_QUALITY_ENABLED` | bool | `quality_enabled` |
| `OPTICORE_QUALITY_REJECT_ON_FAILURE` | bool | `quality_reject_on_failure` |
| `OPTICORE_LOG_PROMPTS` | bool | `log_prompts` |

Boolean values are parsed from `1`, `true`, `yes`, `on` (case-insensitive).

## Provider configuration

```python
from opticore.core.config import ProviderConfig

ProviderConfig(
    provider="openai",
    model="gpt-4o-mini",       # optional; SDK picks a default otherwise
    api_key_env="OPENAI_API_KEY",  # NAME of the env var, never the secret itself
    base_url=None,             # OpenAI-compatible endpoints (vLLM, llama.cpp...)
    timeout_seconds=60.0,      # must be > 0
    max_retries=3,             # must be >= 0
)
```

API keys are read from the environment by the provider at call time.

## Validation rules

- `safety_mode` must be one of `safe`, `balanced`, `aggressive`.
- `max_token_budget`, `cache_ttl_seconds >= 0`.
- `semantic_cache_threshold`, `quality_minimum_score` must be in `(0, 1]`.
- `provider` must be non-empty; `timeout_seconds > 0`; `max_retries >= 0`.
- Invalid values raise `ConfigurationError` with an actionable message.
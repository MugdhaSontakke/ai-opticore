# Exceptions

AI-OptiCore uses a typed exception hierarchy so callers can distinguish
config problems, provider failures, cache issues, and optimizer errors
without inspecting messages.

All public exceptions inherit from `OptiCoreError`.

```python
from opticore.exceptions import (
    OptiCoreError,
    ConfigurationError,
    OptimizationError,
    QualityEvaluationError,
    CacheError,
    ProviderError,
    ProviderAuthError,
    ProviderRateError,
    ProviderTimeoutError,
    UnsupportedModelError,
    TokenizerNotFoundError,
    HardwareBackendUnavailableError,
)
```

## Hierarchy

```
OptiCoreError
├── ConfigurationError
├── OptimizationError
├── QualityEvaluationError
├── CacheError
├── UnsupportedModelError
├── TokenizerNotFoundError
├── HardwareBackendUnavailableError
└── ProviderError
    ├── ProviderAuthError
    ├── ProviderRateError
    └── ProviderTimeoutError
```

## Descriptions

| Exception | When raised | Suggested action |
| --- | --- | --- |
| `ConfigurationError` | Invalid or missing configuration; secrets in config files | Fix config; use environment variables for keys |
| `OptimizationError` | An optimizer fails during request transformation | Check optimizer logs; consider disabling the optimizer |
| `QualityEvaluationError` | Quality gate cannot compute similarity | Verify evaluator is configured; check input shape |
| `CacheError` | Non-recoverable cache backend failure | Check backend connectivity; disable caching if not required |
| `ProviderError` | Generic provider failure | Inspect cause; check provider-specific subclass |
| `ProviderAuthError` | Missing or invalid API credentials | Verify the env var referenced by `api_key_env` is set |
| `ProviderRateError` | Provider-side rate limit or quota exceeded | Back off; increase `max_retries` |
| `ProviderTimeoutError` | Provider call exceeded `timeout_seconds` | Increase timeout; reduce request payload |
| `UnsupportedModelError` | Model not supported for requested operation (e.g. tokenizer mode `custom` with unknown model) | Use `tokenizer_mode="tiktoken"` (default) or register a tokenizer |
| `TokenizerNotFoundError` | No tokenizer registered for model/encoding | Same as above |
| `HardwareBackendUnavailableError` | Requested hardware backend is unavailable | Use `HardwareBackend.available()` for detection; fall back to CPU |

## CLI error handling

The CLI catches `OptiCoreError` at the top level and prints a single
`error: <message>` line (no traceback) with exit code **1**. This keeps
user-facing output clean while preserving debuggability in library use
(full tracebacks are available to callers).

KeyboardInterrupt produces exit code **130**.

## Log redaction

The `opticore.logging.RedactingFilter` automatically strips
`key=value` style secrets (including `sk-*` tokens and Bearer headers)
from log records so config bugs never leak API keys into log files.
# Security Policy

AI-OptiCore treats security seriously. If you discover a vulnerability, please
report it privately so it can be fixed before it is disclosed publicly.

## Reporting a vulnerability

**Do not open a public issue.** Email the maintainers at
`security@ai-opticore.dev` with:

- A description of the vulnerability.
- The affected version(s) and configuration.
- A minimal reproduction, if possible.
- Impact assessment (what an attacker could do).

You will receive an acknowledgment within 3 business days. We will work with
you on a coordinated disclosure timeline.

## What the project does for security

- **API keys and secrets are never hard-coded.** Providers read credentials
  from environment variables only. Config files that contain known secret-key
  fields are rejected at load time.
- **SSRF protection for provider URLs.** Every provider validates its base URL
  through `opticore.security.validate_base_url` before any connection is made.
  Cloud-metadata and link-local addresses are always blocked; private-network
  and loopback hosts are blocked by default. Local model endpoints (Ollama,
  vLLM, llama.cpp) keep working because `allow_private_networks=True` is the
  default for the public `ProviderConfig` — **if you accept provider URLs from
  untrusted users, set `allow_private_networks=False` and/or pin
  `allowed_hosts` to an explicit allowlist**, otherwise a malicious URL could
  reach internal services. `allowed_hosts` (when set) overrides all network
  checks and is the recommended deploy-time setting.
- **No secret leakage in errors.** All provider error messages are passed
  through `opticore.security.redact_text` (API keys, bearer tokens,
  authorization headers) before they are raised or logged. A failed provider
  call never surfaces a raw exception body that may echo request headers.
- **Redacting log filter.** The logging layer redacts likely-sensitive fields
  (api_key, token, secret, password, authorization) before writing logs.
- **Prompt/content logging is opt-in.** By default AI-OptiCore does not log
  prompt text or responses. Set `OPTICORE_LOG_PROMPTS=1` or
  `log_prompts: true` in config to enable it explicitly.
- **No fabricated data.** Benchmarks and metrics only report what was
  actually measured.
- **No unsafe redirects followed.** Provider HTTP clients do not follow
  redirects to arbitrary hosts during validation; unreachable/invalid URLs
  surface as typed `ProviderError` (or `ConfigurationError` at construction).

## Data handling

| Data | Default behavior |
| --- | --- |
| Prompt content | Kept in memory for the request lifetime; persisted to cache if caching is enabled; not logged by default |
| API keys | Read from environment; never stored in config files; never logged |
| Telemetry | None sent anywhere; all metrics are local |

When using the semantic cache, embeddings of your prompts reside in the cache
store. If you use the in-memory store, embeddings are process-local. Choose a
suitable store for your data classification needs.

## Sandboxing note

Optimizers operate on text only and never execute prompts or model output.
Providers fetch over the network per your configuration.

## Reporting process expectations

We aim to:

1. Confirm receipt within 3 business days.
2. Assess severity and impact.
3. Ship a fix in the next release cycle (or earlier for critical issues).
4. Credit reporters (if they wish) after the fix is public.

## Scope

This policy applies to the `src/opticore` package, the CLI, and the bundled
dashboard. Third-party dependencies are covered by their own security
policies.
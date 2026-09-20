# AI-OptiCore image — production-lean.
#
# - Runs as a non-root user (no privilege escalation).
# - Installs only the core runtime deps (clients/extras are opt-in extras).
# - Verifies the install with config validate at build and via HEALTHCHECK.
#
# Build:   docker build -t ai-opticore .
# Run:     docker run --rm ai-opticore --help
# Run CLI: docker run --rm -e OPENAI_API_KEY=... ai-opticore benchmark --provider openai

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package first so layer caching is maximized.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Build-time smoke check: the CLI must load and validate config.
RUN ai-opticore config validate >/dev/null

# Non-root runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.opticore \
    && chown -R appuser:appuser /app \
    && chmod 700 /app/.opticore

USER appuser
WORKDIR /app

# HEALTHCHECK: the CLI has no long-running server; validate that the process
# image, config loader, and CLI wiring are operational.
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
    CMD ["ai-opticore", "config", "validate"]

ENTRYPOINT ["ai-opticore"]
CMD ["--help"]
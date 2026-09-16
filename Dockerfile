FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install -e ".[all]"

# Optional: run usage e.g.
# docker run --rm -e OPENAI_API_KEY=... ai-opticore benchmark --provider openai
ENTRYPOINT ["ai-opticore"]
CMD ["--help"]
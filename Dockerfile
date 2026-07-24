FROM ghcr.io/astral-sh/uv:0.11.15 AS uv

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY --from=uv /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

# The serving API needs only locked core dependencies, not notebook extras.
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY best_pipeline_search_runs ./best_pipeline_search_runs

USER 1000

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

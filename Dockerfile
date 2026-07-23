FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    fastapi \
    joblib \
    "pandas>=2.2" \
    pydantic \
    pyyaml \
    "scikit-learn>=1.7.2" \
    uvicorn

COPY src ./src
COPY config ./config
COPY best_pipeline_search_runs ./best_pipeline_search_runs

USER 1000

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

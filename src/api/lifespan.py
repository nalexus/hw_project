"""Application startup and model-loading lifecycle."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.model_loader import build_predictor
from src.api.settings import load_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings and predictor once during FastAPI startup."""

    settings = getattr(app.state, "settings", None) or load_settings()
    app.state.settings = settings

    existing_predictor = getattr(app.state, "predictor", None)
    if existing_predictor is None:
        app.state.model_load_error = None
        try:
            app.state.predictor = build_predictor(settings)
        except Exception as exc:  # pragma: no cover - exercised via readiness path
            app.state.predictor = None
            app.state.model_load_error = str(exc)

    yield

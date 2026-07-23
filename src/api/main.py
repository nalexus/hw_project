"""FastAPI application factory and runtime lifecycle."""

from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.batching import PredictionBatcher
from src.api.routes import router
from src.api.settings import ApiSettings, load_settings
from src.model.predict import PredictorMultiClass


CLASSIFY_DOCUMENT_PATH = "/classify_document"


def build_predictor(settings: ApiSettings) -> PredictorMultiClass:
    """Load the persisted model and attach its runtime OOD policy."""

    pipeline = joblib.load(settings.model_path)
    return PredictorMultiClass(
        pipeline=pipeline,
        threshold=settings.threshold,
        threshold_policy=settings.threshold_policy,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once and manage the prediction batcher lifecycle."""

    settings = getattr(app.state, "settings", None) or load_settings()
    app.state.settings = settings

    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        app.state.model_load_error = None
        try:
            predictor = build_predictor(settings)
            app.state.predictor = predictor
        except Exception as exc:  # pragma: no cover - exercised via readiness path
            app.state.predictor = None
            app.state.model_load_error = str(exc)

    batcher = getattr(app.state, "prediction_batcher", None)
    if predictor is not None and batcher is None:
        batcher = PredictionBatcher(
            predictor=predictor,
            max_delay_ms=settings.batch_max_delay_ms,
            max_batch_size=settings.batch_max_size,
            max_queue_size=settings.batch_queue_size,
        )
        app.state.prediction_batcher = batcher
    if batcher is not None:
        await batcher.start()

    try:
        yield
    finally:
        if batcher is not None:
            await batcher.stop()


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Return a concise message for malformed JSON request bodies."""

    if any(error.get("type") == "json_invalid" for error in exc.errors()):
        return JSONResponse(status_code=422, content={"detail": "Malformed JSON request body"})
    return await request_validation_exception_handler(request, exc)


def register_middleware(app: FastAPI) -> None:
    """Reject non-JSON classification requests before body validation."""

    @app.middleware("http")
    async def enforce_json_content_type(request: Request, call_next):
        if (
            request.method == "POST"
            and request.url.path == CLASSIFY_DOCUMENT_PATH
            and not is_json_content_type(request.headers.get("content-type"))
        ):
            return JSONResponse(
                status_code=415,
                content={"detail": "Content-Type must be application/json"},
            )
        return await call_next(request)


def is_json_content_type(content_type: str | None) -> bool:
    """Return whether a request content type is JSON with optional parameters."""

    if content_type is None:
        return False
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    return media_type == "application/json"


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create and configure the document classification API app."""

    app = FastAPI(
        title="Document Classification API",
        description=(
            "Synchronous API for classifying a single text document with a "
            "preloaded scikit-learn model."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    register_middleware(app)
    app.include_router(router)
    return app


app = create_app()

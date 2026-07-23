"""Request models, app-state dependencies, and API route handlers."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from src.api.batching import PredictionBatchQueueFull, PredictionBatcher
from src.api.settings import ApiSettings, load_settings
from src.model.predict import PredictorMultiClass


class ClassificationRequest(BaseModel):
    """Validated request body for one document classification."""

    model_config = ConfigDict(extra="forbid")

    document_text: StrictStr = Field(
        ...,
        description="Document text to classify.",
        examples=["The team won the championship after extra time."],
    )

    @field_validator("document_text")
    @classmethod
    def validate_document_text(cls, value: str) -> str:
        """Reject empty document text while preserving the submitted content."""

        if not value.strip():
            raise ValueError("document_text must not be empty")
        return value


class ClassificationResponse(BaseModel):
    """Successful classification response with serving-model identity."""

    message: str
    label: str
    model_version: str


class HealthResponse(BaseModel):
    """Liveness or readiness response with serving-model identity."""

    status: str
    model_version: str


def get_settings(request: Request) -> ApiSettings:
    """Return cached application settings, loading them for injected test apps."""

    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = load_settings()
        request.app.state.settings = settings
    return settings


def get_predictor(request: Request) -> PredictorMultiClass:
    """Return the loaded predictor or report that the API is not ready."""

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        detail = getattr(request.app.state, "model_load_error", None) or "Model is not ready"
        raise HTTPException(status_code=503, detail=detail)
    return predictor


def get_prediction_batcher(request: Request) -> PredictionBatcher:
    """Return the app batcher, creating one only for injected test apps."""

    batcher = getattr(request.app.state, "prediction_batcher", None)
    if batcher is not None:
        return batcher
    settings = get_settings(request)
    batcher = PredictionBatcher(
        predictor=get_predictor(request),
        max_delay_ms=settings.batch_max_delay_ms,
        max_batch_size=settings.batch_max_size,
        max_queue_size=settings.batch_queue_size,
    )
    request.app.state.prediction_batcher = batcher
    return batcher


router = APIRouter()


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Reports whether the API process is up.",
)
def health_live(request: Request) -> HealthResponse:
    """Return a liveness response without requiring model readiness."""

    return HealthResponse(status="live", model_version=get_settings(request).model_version)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Reports whether the model has been loaded and the API is ready to serve predictions.",
    responses={503: {"description": "Model is not ready."}},
)
def health_ready(request: Request) -> HealthResponse:
    """Return readiness only after the predictor is available."""

    get_predictor(request)
    return HealthResponse(status="ready", model_version=get_settings(request).model_version)


@router.post(
    "/classify_document",
    response_model=ClassificationResponse,
    summary="Classify one document",
    description="Accepts one document and returns the predicted label.",
    responses={
        413: {"description": "The request document exceeds the configured maximum length."},
        415: {"description": "The request Content-Type is not application/json."},
        422: {"description": "The request body is invalid."},
        500: {"description": "Unexpected inference failure."},
        503: {"description": "The model is not ready."},
    },
)
async def classify_document(
    payload: ClassificationRequest, request: Request
) -> ClassificationResponse:
    """Classify one validated document and return the final label."""

    settings = get_settings(request)
    validate_document_length(payload.document_text, settings.max_document_length)
    label = await predict_label(get_prediction_batcher(request), payload.document_text)
    return ClassificationResponse(
        message="Classification successful",
        label=label,
        model_version=settings.model_version,
    )


def validate_document_length(document_text: str, max_length: int) -> None:
    """Reject documents longer than the configured API limit."""

    if len(document_text) > max_length:
        raise HTTPException(
            status_code=413,
            detail=(
                "document_text exceeds the configured maximum length of "
                f"{max_length} characters"
            ),
        )


async def predict_label(batcher: PredictionBatcher, document_text: str) -> str:
    """Run batched predictor inference for a single document request."""

    try:
        return await batcher.predict(document_text)
    except PredictionBatchQueueFull as exc:
        raise HTTPException(status_code=503, detail="Prediction queue is full") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected inference failure") from exc

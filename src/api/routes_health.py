"""Health-check route handlers."""

from fastapi import APIRouter, Request

from src.api.dependencies import get_predictor, get_settings
from src.api.schemas import HealthResponse


router = APIRouter()


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Reports whether the API process is up.",
)
def health_live(request: Request) -> HealthResponse:
    """Return a liveness response without requiring model readiness."""

    settings_obj = get_settings(request)
    return HealthResponse(status="live", model_version=settings_obj.model_version)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Reports whether the model has been loaded and the API is ready to serve predictions.",
    responses={503: {"description": "Model is not ready."}},
)
def health_ready(request: Request) -> HealthResponse:
    """Return readiness only after the predictor is available."""

    settings_obj = get_settings(request)
    get_predictor(request)
    return HealthResponse(status="ready", model_version=settings_obj.model_version)

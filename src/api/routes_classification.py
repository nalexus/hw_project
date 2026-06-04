"""Document-classification route handlers."""

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from src.api.dependencies import get_predictor, get_settings
from src.api.schemas import ClassificationRequest, ClassificationResponse


router = APIRouter()


@router.post(
    "/classify_document",
    response_model=ClassificationResponse,
    summary="Classify one document",
    description=(
        "Accepts one document, runs model inference during the request, "
        "and returns the predicted label."
    ),
    responses={
        413: {
            "description": "The request document exceeds the configured maximum length."
        },
        415: {"description": "The request Content-Type is not application/json."},
        422: {"description": "The request body is invalid."},
        500: {"description": "Unexpected inference failure."},
        503: {"description": "The model is not ready."},
    },
)
def classify_document(
    payload: ClassificationRequest, request: Request
) -> ClassificationResponse:
    """Classify one validated document and return the predicted label."""

    settings_obj = get_settings(request)
    predictor = get_predictor(request)
    validate_document_length(payload.document_text, settings_obj.max_document_length)
    label = predict_label(predictor, payload.document_text)
    return ClassificationResponse(
        message="Classification successful",
        label=label,
        model_version=settings_obj.model_version,
    )


def validate_document_length(document_text: str, max_length: int) -> None:
    """Reject documents longer than the configured API limit."""

    if len(document_text) <= max_length:
        return
    raise HTTPException(
        status_code=413,
        detail=(
            "document_text exceeds the configured maximum length of "
            f"{max_length} characters"
        ),
    )


def predict_label(predictor, document_text: str) -> str:
    """Run predictor inference for a single document."""

    try:
        texts = np.array([document_text], dtype=object)
        return predictor.predict(texts)[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Unexpected inference failure"
        ) from exc

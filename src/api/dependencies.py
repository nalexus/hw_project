from fastapi import HTTPException, Request

from src.api.settings import ApiSettings, load_settings
from src.model.predict import PredictorMultiClass


def get_settings(request: Request) -> ApiSettings:
    settings_from_state = getattr(request.app.state, "settings", None)
    if settings_from_state is not None:
        return settings_from_state

    loaded_settings = load_settings()
    request.app.state.settings = loaded_settings
    return loaded_settings


def get_predictor(request: Request) -> PredictorMultiClass:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        detail = (
            getattr(request.app.state, "model_load_error", None) or "Model is not ready"
        )
        raise HTTPException(status_code=503, detail=detail)
    return predictor

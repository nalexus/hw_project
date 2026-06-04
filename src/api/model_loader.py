import joblib

from src.api.settings import ApiSettings
from src.model.predict import PredictorMultiClass


def build_predictor(settings: ApiSettings) -> PredictorMultiClass:
    """Load the persisted model and attach the configured reject policy."""

    pipeline = joblib.load(settings.model_path)
    return PredictorMultiClass(
        pipeline=pipeline,
        threshold=settings.threshold,
        threshold_policy=settings.threshold_policy,
    )

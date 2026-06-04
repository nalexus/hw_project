"""Runtime predictor with confidence-based rejection to ``other``."""

import numpy as np

from src.model.predict.validators import normalize_policy, threshold_for_text
from src.model.predict.length import length_bucket, word_count


class PredictorMultiClass:
    """Predict known labels and return ``other`` below confidence threshold."""

    def __init__(self, pipeline, threshold=0.16, threshold_policy=None):
        """Store the sklearn pipeline and normalized reject policy."""

        self.pipeline = pipeline
        self.threshold_policy = normalize_policy(threshold, threshold_policy)

    def predict(self, texts):
        """Return one final label for every input text."""

        probs = self.pipeline.predict_proba(texts)
        max_probs = np.max(probs, axis=1)
        best_class_indices = np.argmax(probs, axis=1)
        predicted_classes = self.pipeline.classes_[best_class_indices]
        return [
            pred if prob >= threshold_for_text(self.threshold_policy, str(text)) else "other"
            for text, pred, prob in zip(texts, predicted_classes, max_probs)
        ]

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.model_loader import build_predictor
from src.api.settings import ApiSettings, load_settings


class StubPredictor:
    """Small API predictor test double."""

    def __init__(self, label="sport", error=None):
        """Store the label or error returned during prediction."""

        self.label = label
        self.error = error

    def predict(self, texts):
        """Return a configured label or raise a configured error."""

        if self.error is not None:
            raise self.error
        return [self.label for _ in texts]


class FixedProbabilityPipeline:
    """Small model-loader test double with predictable probabilities."""

    classes_ = np.array(["food", "sport"])

    def predict_proba(self, texts):
        """Return a confident food prediction for each text."""

        return np.tile(np.array([0.60, 0.40]), (len(texts), 1))


def build_client(predictor=None, max_document_length=50):
    """Create a TestClient with an injected predictor."""

    settings = ApiSettings(
        model_path=Path("unused/model.joblib"),
        threshold=0.16,
        threshold_policy=None,
        runtime_config_path=None,
        max_document_length=max_document_length,
        model_version="baseline",
    )
    app = create_app(settings=settings)
    app.state.settings = settings
    app.state.predictor = predictor
    app.state.model_load_error = None
    return TestClient(app)


def test_classify_document_success():
    client = build_client(predictor=StubPredictor(label="sport"))

    response = client.post("/classify_document", json={"document_text": "match report"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "Classification successful",
        "label": "sport",
        "model_version": "baseline",
    }


def test_classify_document_missing_field():
    client = build_client(predictor=StubPredictor())

    response = client.post("/classify_document", json={})

    assert response.status_code == 422


def test_classify_document_empty_field():
    client = build_client(predictor=StubPredictor())

    response = client.post("/classify_document", json={"document_text": "   "})

    assert response.status_code == 422


def test_classify_document_non_string_field():
    client = build_client(predictor=StubPredictor())

    response = client.post("/classify_document", json={"document_text": 123})

    assert response.status_code == 422


def test_classify_document_malformed_json():
    client = build_client(predictor=StubPredictor())

    response = client.post(
        "/classify_document",
        content='{"document_text": "missing closing brace"',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Malformed JSON request body"


def test_classify_document_rejects_wrong_content_type():
    client = build_client(predictor=StubPredictor())

    response = client.post(
        "/classify_document",
        content='{"document_text": "valid json body"}',
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Content-Type must be application/json"


def test_classify_document_rejects_missing_content_type():
    client = build_client(predictor=StubPredictor())

    response = client.post(
        "/classify_document",
        content='{"document_text": "valid json body"}',
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Content-Type must be application/json"


def test_classify_document_oversized_payload():
    client = build_client(predictor=StubPredictor(), max_document_length=5)

    response = client.post("/classify_document", json={"document_text": "too long"})

    assert response.status_code == 413


def test_classify_document_passes_through_other_label():
    client = build_client(predictor=StubPredictor(label="other"))

    response = client.post("/classify_document", json={"document_text": "ambiguous"})

    assert response.status_code == 200
    assert response.json()["label"] == "other"


def test_classify_document_real_selected_model_returns_other_for_ood():
    """Verify the served selected model can reject an unrelated document."""

    settings = load_settings()
    app = create_app(settings=settings)
    document_text = (
        "A handwritten note lists two unrelated errands: buy printer paper, "
        "move the blue chair, and call the building manager about a hallway light."
    )

    with TestClient(app) as client:
        response = client.post(
            "/classify_document", json={"document_text": document_text}
        )

    assert response.status_code == 200
    assert response.json()["label"] == "other"
    assert response.json()["model_version"] == settings.model_version


def test_readiness_requires_model():
    client = build_client(predictor=None)

    response = client.get("/health/ready")

    assert response.status_code == 503


def test_classify_document_predictor_failure_returns_controlled_error():
    client = build_client(predictor=StubPredictor(error=RuntimeError("boom")))

    response = client.post("/classify_document", json={"document_text": "sample"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Unexpected inference failure"


def test_load_settings_uses_yaml_defaults(tmp_path):
    """Verify basic YAML settings still load without a runtime config."""

    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        "model_path: models/model.joblib\n"
        "threshold: 0.25\n"
        "max_document_length: 123\n"
        "model_version: baseline-v1\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_file)
    project_root = Path(__file__).resolve().parents[2]

    assert settings.model_path == project_root / "models" / "model.joblib"
    assert settings.threshold == 0.25
    assert settings.threshold_policy is None
    assert settings.runtime_config_path is None
    assert settings.max_document_length == 123
    assert settings.model_version == "baseline-v1"


def test_load_settings_env_overrides_yaml(monkeypatch, tmp_path):
    """Verify explicit environment settings keep deployment override behavior."""

    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        "model_path: models/model.joblib\n" "model_version: baseline-v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "external" / "model-v2.joblib"))
    monkeypatch.setenv("MODEL_VERSION", "candidate-v2")

    settings = load_settings(settings_file)

    assert settings.model_path == tmp_path / "external" / "model-v2.joblib"
    assert settings.model_version == "candidate-v2"


def test_load_settings_uses_runtime_config_model_and_policy(tmp_path):
    """Verify selected training runtime config can drive API settings."""

    model_file = tmp_path / "selected_model.joblib"
    model_file.write_text("placeholder", encoding="utf-8")
    runtime_file = tmp_path / "runtime_config.json"
    runtime_file.write_text(
        "{"
        '"policy": "length_bucket",'
        '"model_path": "selected_model.joblib",'
        '"fallback_model_path": "classifier_baseline.joblib",'
        '"default_threshold": 0.40,'
        '"bucket_thresholds": {"ultra_short": 0.20, "medium": 0.50}'
        "}",
        encoding="utf-8",
    )
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"runtime_config_path: {runtime_file.as_posix()}\n"
        "model_path: unused/model.joblib\n"
        "model_version: selected-test\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_file)

    assert settings.model_path == model_file
    assert settings.runtime_config_path == runtime_file
    assert settings.threshold_policy["policy"] == "length_bucket"
    assert settings.threshold_policy["fallback_model_path"] == (
        "classifier_baseline.joblib"
    )
    assert settings.threshold_policy["bucket_thresholds"]["medium"] == 0.50


def test_load_settings_discovers_single_prod_run(tmp_path):
    """Verify default serving config comes from the one PROD run directory."""

    runs_dir = tmp_path / "best_pipeline_search_runs"
    prod_dir = runs_dir / "run_20260101T000000Z_PROD"
    prod_dir.mkdir(parents=True)
    model_file = prod_dir / "best_pipeline_run_20260101T000000Z_PROD.joblib"
    model_file.write_text("placeholder", encoding="utf-8")
    runtime_file = prod_dir / "runtime_config.json"
    runtime_file.write_text(
        "{"
        '"policy": "length_bucket",'
        '"model_path": "best_pipeline_run_20260101T000000Z_PROD.joblib",'
        '"default_threshold": 0.40,'
        '"bucket_thresholds": {"medium": 0.50}'
        "}",
        encoding="utf-8",
    )
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"prod_runs_dir: {runs_dir.as_posix()}\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_file)

    assert settings.runtime_config_path == runtime_file
    assert settings.model_path == model_file
    assert settings.model_version == "run_20260101T000000Z_PROD"
    assert settings.threshold_policy["policy"] == "length_bucket"


def test_load_settings_without_prod_run_points_to_missing_model(tmp_path):
    """Verify no PROD marker leaves startup unable to load a model."""

    runs_dir = tmp_path / "best_pipeline_search_runs"
    runs_dir.mkdir()
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"prod_runs_dir: {runs_dir.as_posix()}\n",
        encoding="utf-8",
    )

    settings = load_settings(settings_file)

    assert settings.runtime_config_path is None
    assert settings.model_path == runs_dir / "NO_PROD_RUN_FOUND.joblib"
    assert settings.model_version == "unpromoted"


def test_build_predictor_applies_runtime_policy(monkeypatch, tmp_path):
    """Verify model loading attaches the selected runtime reject policy."""

    settings = ApiSettings(
        model_path=tmp_path / "model.joblib",
        threshold=0.16,
        threshold_policy={
            "policy": "length_bucket",
            "default_threshold": 0.50,
            "bucket_thresholds": {"ultra_short": 0.70, "medium": 0.50},
        },
        runtime_config_path=None,
        max_document_length=10000,
        model_version="selected-test",
    )
    monkeypatch.setattr(
        "src.api.model_loader.joblib.load", lambda path: FixedProbabilityPipeline()
    )

    predictor = build_predictor(settings)
    predictions = predictor.predict(np.array(["tiny note", " ".join(["word"] * 130)]))

    assert predictions == ["other", "food"]

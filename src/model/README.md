# Clean Parallel Model Package

This package is the canonical implementation of the selected model workflow.

## Structure

- `../../config/model/train.yaml`: default human-edited training settings.
- `../../config/model/tune.yaml`: default human-edited tuning settings.
- `../../config/model/predict.yaml`: default human-edited prediction policy settings.
- `predict/`: runtime prediction wrapper with confidence-based `other` rejection.
- `predict/validators.py`: Pydantic validation, runtime reject-policy parsing, and threshold resolution.
- `predict/length.py`: runtime token counting and length-bucket mapping.
- `data/`: copied provided dataset and copied synthetic examples only.
- `data_preparation/loader.py`: loads copied provided data and copied synthetic examples.
- `data_preparation/splits.py`: deterministic train/validation/test/provided-other splitting.
- `data_preparation/leakage.py`: exact and near-duplicate leakage checks.
- `data_preparation/text.py`: shared tokenization, hashing, and duplicate-check helpers.
- `evaluate/predictions.py`: prediction rows and reject-policy application.
- `evaluate/metrics.py`: overall and grouped evaluation metrics.
- `tune/candidates.py`: TF-IDF/logistic-regression search space and known-class CV.
- `tune/thresholds.py`: global-start plus length-bucket reject threshold tuning.
- `tune/selection.py`: shared tuning objective and ranking helpers.
- `tune/validators.py`: Pydantic validation for tuning objective and threshold-grid configuration.
- `train/validators.py`: Pydantic validation for training paths, split, CV, calibration, and run settings.
- `train/models.py`: sklearn estimator construction and calibrated fitting.
- `experiment/runner.py`: CLI orchestration only.
- `experiment/manifest.py`: runtime config, metadata, and summary payload builders.
- `experiment/writer.py`: model, metrics, predictions, metadata, and log writing.

## Run

```bash
python -m src.model.experiment.runner
```

The runner loads `config/model/train.yaml` and `config/model/tune.yaml` by default. CLI flags override explicit fields when needed.

Outputs are written under:

```text
best_pipeline_search_runs/
```

The copied dataset and synthetic examples live under `data/`, while
the supporting load/split/leakage code lives under `src/model/data_preparation/`.

# Text Classification Case Study

This repository contains a text classification model and a production-style synchronous REST API for single-document inference.

All commands below assume they are run from the project directory that contains this README.

## API Overview

- Framework: FastAPI
- Inference wrapper: `PredictorMultiClass`
- Default selected model artifact: `best_pipeline_search_runs/*_PROD/best_pipeline_*_PROD.joblib`
- Default selected policy: `best_pipeline_search_runs/*_PROD/runtime_config.json`
- Primary endpoint: `POST /classify_document`
- Health endpoints: `GET /health/live`, `GET /health/ready`

The API is intentionally synchronous because the assignment asks for one request to return one classification result immediately.

## Model Performance Summary

The selected serving model is the length-bucket calibrated TF-IDF plus logistic regression artifact:

- artifact: `best_pipeline_search_runs/run_20260531T162610Z_PROD/best_pipeline_run_20260531T162610Z_PROD.joblib`
- runtime policy: `best_pipeline_search_runs/run_20260531T162610Z_PROD/runtime_config.json`
- source run: `best_pipeline_search_runs/run_20260531T162610Z_PROD`
- model version label: `run_20260531T162610Z_PROD`

Latest selected-run metrics:

| Split | Samples | Known balanced accuracy | Known accuracy | OOD accuracy | Overall accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 249 | 0.9480 | 0.9483 | 1.0000 | 0.9639 |
| Test | 269 | 0.9428 | 0.9432 | 0.8710 | 0.9182 |
| Provided `other` final check | 6 | n/a | n/a | 0.8333 | 0.8333 |

The provided `other` result is 5 correct rejections out of 6 documents, so it is useful as a final sanity check but too small to treat as a broad OOD guarantee. Detailed metrics and predictions are stored next to the selected run under `best_pipeline_search_runs/run_20260531T162610Z_PROD/`.

## Project Layout

- `config/`: YAML defaults for API serving, model training, threshold tuning, and prediction policy
- `data/`: copied labeled dataset plus file-backed synthetic samples used by model experiments
- `src/`: application and modeling source code
- `tests/`: API, predictor, loader, and model workflow regression tests
- `kubernetes/`: minimal deployment/service manifests plus local helper scripts
- `best_pipeline_search_runs/`: model experiment outputs, metrics, predictions, serialized artifacts, and the single `_PROD` run served by default
- `src/api/main.py`: FastAPI application factory
- `src/api/lifespan.py`: startup model loading lifecycle
- `src/api/routes*.py`: route aggregation plus focused health/classification handlers
- `src/api/config/settings.py`: API settings loader implementation
- `src/api/settings.py`: compatibility exports for settings imports
- `config/api/settings.yaml`: runtime configuration
- `config/model/`: static model train, tune, and predict defaults
- `src/model/predict/`: classification wrapper with `"other"` reject policies
- `best_pipeline_search_runs/<run_name>_PROD/runtime_config.json`: selected serving policy
- `tests/api/test_api_endpoints.py`: focused API contract tests

## Model Package Flow

The modeling package is organized as a small pipeline. `data_preparation` loads provided and synthetic records, assigns deterministic train/validation/test splits, and checks text metadata. `train` owns validated training config and estimator construction. `tune` compares candidate TF-IDF/logistic-regression configs and selects length-bucket rejection thresholds. `evaluate` builds prediction rows, applies the reject policy, and computes metrics. `experiment` ties those pieces together, writes run artifacts, and optionally promotes a run to `_PROD`.

The serving API does not retrain models. It loads the promoted artifact and runtime policy, then calls `src/model/predict/` through `PredictorMultiClass` for request-time classification.

## Prerequisites

- Python 3.13
- Installed dependencies from `pyproject.toml`

If you are using the local virtual environment already present in the repo, commands below assume `.venv`.

## Run The API

Start the development server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs and schema:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Run With Docker

Build the image:

```bash
docker build -t text-classifier .
```

Run the container:

```bash
docker run --rm -p 8000:8000 text-classifier
```

How model selection works:

1. The image contains `best_pipeline_search_runs/`.
2. The API serves the single run folder whose name contains `_PROD`; that run contains the `.joblib` model and `runtime_config.json`.
3. If `RUNTIME_CONFIG_PATH` is set, the API loads model and policy settings from that JSON file.
4. If `MODEL_PATH` is set, the API loads the model from that path instead of the runtime config model.
5. If `MODEL_VERSION` is set, responses and health checks use that value instead of the `_PROD` run folder name.

Use the baked-in model:

```bash
docker run --rm -p 8000:8000 text-classifier
```

Switch to an external model mounted from the host:

```powershell
docker run --rm -p 8000:8000 `
  -e RUNTIME_CONFIG_PATH=/models/runtime_config.json `
  -e MODEL_PATH=/models/classifier_v2.joblib `
  -e MODEL_VERSION=candidate-v2 `
  -v C:\models:/models `
  text-classifier
```

Switch back to the baked-in model:

```bash
docker run --rm -p 8000:8000 text-classifier
```

Notes:

- the build context intentionally excludes training notebooks, datasets, tests, and local virtualenv files via `.dockerignore`
- the image includes runtime code, top-level `config/`, and `best_pipeline_search_runs`
- if `MODEL_PATH` points to a missing or invalid artifact, startup readiness will fail and inference requests will return `503`

## Kubernetes Manifests

Example Kubernetes manifests are provided in `kubernetes/manifests/`.

They are intentionally minimal:

- `kubernetes/manifests/deployment.yaml`: runs two API pod replicas from the `text-classifier:latest` image
- `kubernetes/manifests/service.yaml`: exposes a stable internal service and load-balances traffic across ready pods
- `MODEL_VERSION` can be set explicitly, otherwise the API uses the `_PROD` run folder name
- readiness probe: calls `/health/ready`, so pods without a loaded model do not receive classification traffic
- liveness probe: calls `/health/live`, so Kubernetes can restart a stuck process

Build the image first:

```bash
docker build -t text-classifier .
```

Apply the manifests:

```bash
kubectl apply -f kubernetes/manifests/
```

For local Docker Desktop Kubernetes, helper scripts are available at:

```powershell
.\kubernetes\local_deploy\win_local.ps1
```

```bash
./kubernetes/local_deploy/linux_local.sh
```

Those scripts import the image into Docker Desktop's local Kubernetes runtime; for a remote cluster, push the image to a registry and update the deployment image instead.

For local testing, forward the service to your machine:

```bash
kubectl port-forward service/text-classifier-api 8000:80
```

Keep the port-forward in the foreground unless you explicitly need file logs. The local helper scripts can start it for you without saving logs:

```powershell
.\kubernetes\local_deploy\win_local.ps1 -PortForward
```

```bash
PORT_FORWARD=true ./kubernetes/local_deploy/linux_local.sh
```

If background file logs are needed for troubleshooting, opt in and write them under `.local/logs/` instead of the project root:

```powershell
.\kubernetes\local_deploy\win_local.ps1 -PortForward -SavePortForwardLogs -PortForwardLogDir .local\logs
```

```bash
PORT_FORWARD=true SAVE_PORT_FORWARD_LOGS=true PORT_FORWARD_LOG_DIR=.local/logs ./kubernetes/local_deploy/linux_local.sh
```

Then call the API at `http://127.0.0.1:8000/classify_document`.

## Configuration

Runtime settings live in `config/api/settings.yaml`:

```yaml
prod_runs_dir: best_pipeline_search_runs
max_document_length: 10000
```

Meaning:

- `prod_runs_dir`: directory containing training runs; exactly one run folder must contain `_PROD` for default serving
- `max_document_length`: request validation limit in characters
- `model_version`: optional explicit version returned in API responses; by default the `_PROD` run folder name is used

Model-serving settings live in the promoted run's `runtime_config.json`:

```json
{
  "model_path": "best_pipeline_run_20260531T162610Z_PROD.joblib",
  "policy": "length_bucket"
}
```

The API resolves that model path relative to the promoted run directory. It does not silently fall back if no `_PROD` run exists, if more than one `_PROD` run exists, or if the promoted model cannot load.

Environment overrides:

- `RUNTIME_CONFIG_PATH`: optional absolute or repo-relative path to a selected training runtime config, bypassing `_PROD` discovery
- `MODEL_PATH`: optional absolute or repo-relative path to a different model artifact
- `MODEL_VERSION`: optional response label for the selected model

## Input Preprocessing And `other`

Clients send raw text in `document_text`. The API validates that the field is present, is a non-empty string, uses JSON content type, and stays under `max_document_length`; it does not perform separate cleaning before prediction.

Text preprocessing is owned by the serialized scikit-learn pipeline. The selected pipeline uses TF-IDF features with lowercasing, English stop-word filtering, unigrams and bigrams, and sublinear term frequency, followed by calibrated logistic regression.

The model is trained on the 10 known categories only. The provided `other` folder is not used for training or model selection. At inference time, `PredictorMultiClass` predicts the highest-probability known class, then returns `other` if that probability is below the configured threshold for the document length bucket:

| Bucket | Threshold |
| --- | ---: |
| ultra_short | 0.34 |
| short | 0.39 |
| medium | 0.53 |
| long | 0.36 |
| extra_long | 0.48 |

That reject policy is why `other` is a successful classification label, not an API error.

## API Contract

### `POST /classify_document`

Request body:

```json
{
  "document_text": "The team won the championship after extra time."
}
```

Success response:

```json
{
  "message": "Classification successful",
  "label": "sport",
  "model_version": "run_20260531T162610Z_PROD"
}
```

Common error cases:

- `422 Unprocessable Entity`: missing, empty, non-string, or malformed `document_text`
- `413 Payload Too Large`: `document_text` exceeds `max_document_length`
- `503 Service Unavailable`: model not loaded yet or failed to load
- `500 Internal Server Error`: unexpected inference failure

### `GET /health/live`

Confirms the process is running.

Example response:

```json
{
  "status": "live",
  "model_version": "run_20260531T162610Z_PROD"
}
```

### `GET /health/ready`

Confirms the model is loaded and the API is ready to serve predictions.

Example response:

```json
{
  "status": "ready",
  "model_version": "run_20260531T162610Z_PROD"
}
```

## Example Requests

PowerShell:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/classify_document" `
  -ContentType "application/json" `
  -Body '{"document_text":"The spacecraft entered lunar orbit after a six-day journey."}'
```

cURL:

```bash
curl -X POST "http://127.0.0.1:8000/classify_document" \
  -H "Content-Type: application/json" \
  -d "{\"document_text\":\"The spacecraft entered lunar orbit after a six-day journey.\"}"
```

## Tests

Run the focused API tests:

```bash
python -m pytest tests/api/test_api_endpoints.py
```

Run the focused predictor policy tests:

```bash
python -m pytest tests/model/unit/test_predictor_policy.py
```

What the tests cover:

- valid classification response
- request validation failures
- oversized payload handling
- `"other"` passthrough behavior
- real selected-model API rejection for an unrelated OOD document
- readiness behavior when no model is available
- controlled `500` on predictor exceptions
- runtime config loading for the selected reject policy
- length-bucket threshold behavior

Most API tests use a stub predictor so they validate the HTTP contract deterministically. The suite also includes one selected-model API integration check for `other`, and separate predictor tests cover the runtime reject-policy behavior.

### OpenAI-Generated Behavior Fixtures

`tests/model/acceptance/data/i1a_known_class_by_length/fixtures.jsonl` contains 50 committed,
independent test-only fixtures generated with the OpenAI API: all 10 known
classes across all 5 runtime length buckets. These fixtures are not training
data, and fixture generation does not use model predictions as an acceptance
criterion.

The optional refresh command is:

```bash
python -m tests.tools.generate_id_by_length_fixtures --fresh --workers 4 --request-timeout 180
```

The committed fixture integrity and behavior tests do not require API access:

```bash
python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_fixtures.py -v
python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_behavior.py
```

Details, validation rules, and the earlier model-gated fixture mistake are
documented in `synthetic_data_generation.md`.

## Operational Notes

- The model is loaded once during application startup and reused across requests in that process.
- For local production-style serving, use multiple worker processes rather than loading the model per request.
- The selected model artifact is about 17 MB, which is small enough to preload in each API worker or pod.
- Inference is CPU-bound scikit-learn work. Use horizontal pod/worker scaling for concurrent request traffic, and measure p50/p95/p99 latency before choosing replica counts.
- For millions of documents, prefer an offline or batch classification job that calls the vectorizer/model on batches. The synchronous API is appropriate for single-document request/response serving, not high-throughput backfills.
- If synchronous throughput becomes the bottleneck, add a benchmark script, test batch prediction, tune worker count against CPU cores, and consider optimized export formats only after profiling shows sklearn inference is the limiting cost.
- Current observability is minimal: Uvicorn access logs plus `/health/live` and `/health/ready`. A production deployment should add structured JSON request logs with request ID, status code, latency, predicted label, and model version, plus metrics for request count, error count, readiness failures, and inference latency.
- The runtime target is Python `3.13`. The selected artifact was loaded and tested with scikit-learn `1.7.2`. `pyproject.toml` and the Dockerfile require `scikit-learn>=1.7.2`; for reproducible production model loading, build with the project dependency set and avoid older scikit-learn versions. Stricter deployments should pin `scikit-learn==1.7.2` when retraining and serializing a new artifact.

## Model Retraining And Rollout

The canonical modeling workflow is the runner under `src/model/`. Training runs are stored under `best_pipeline_search_runs/`.

### Reproducibility Seed

Model experiments use one visible seed, `random_state: 42` from `config/model/train.yaml`, across three stochastic stages: provided-data splitting, candidate cross-validation, and logistic-regression fitting. Using the same configured seed keeps runs reproducible and easy to audit; changing it intentionally creates a different split/CV/model-fit reproducibility path.

### Data Splitting Logic

Provided known-class documents are split deterministically within each `(label, length_bucket)` group, so train/validation/test keep both class and document-length coverage. Exact duplicate texts are grouped before splitting so duplicate content cannot leak across splits. Synthetic examples keep their explicit split hints from `data/synthetic_samples/`, and OOD examples are only allowed in validation/test because `other` is a reject behavior, not a trained class.

Length buckets use the shared tokenizer and fixed ranges: `ultra_short` is 0-19 tokens, `short` is 20-120, `medium` is 121-435, `long` is 436-738, and `extra_long` is 739 or more. The split logic includes these buckets because confidence and OOD rejection behavior differ between very short snippets and long documents. Stratifying by length reduces train/test skew, so validation and test metrics better reflect the range of document lengths the API will serve.

### Candidate Finetuning Logic

Candidate selection first compares TF-IDF/logistic-regression configurations with stratified known-class cross-validation. The best few candidates are then fitted on train data and tuned on validation data with an `other` threshold policy. The default policy is `length_bucket`, which selects one reject threshold per token-count bucket. You can instead select a single threshold for every input length with `--threshold-policy global`.

The threshold policy numbers are selected, not hand-written. A threshold is the minimum model confidence required to keep the best known-class prediction for that length bucket; if confidence is below the threshold, the runtime wrapper returns `other`. The tuner searches the threshold grid from `config/model/tune.yaml` (`0.00` through `1.00` at `threshold_step: 0.01`) on validation predictions. It starts from one global threshold, then greedily updates each length bucket threshold to maximize:

```text
val_score = 0.5 * validation_known_balanced_accuracy + 0.5 * validation_ood_accuracy
```

Validation known balanced accuracy and validation OOD accuracy are used as tie-breakers, and candidates below the configured minimum known balanced accuracy are filtered when feasible. The final test split and provided `other` folder are held back for evaluation after selection.

For the global policy, the same validation objective and grid are used, but the selected runtime config contains one `threshold` value instead of `default_threshold` plus `bucket_thresholds`. There is no separate no-threshold mode; the served predictor always applies a minimum confidence threshold before returning a known class.

Safe update process:

1. Run a new clean experiment without overwriting serving artifacts:

   ```bash
   python -m src.model.experiment.runner
   ```

2. Review the generated run directory under `best_pipeline_search_runs/`.
3. Promote only after the new run improves or preserves the required known-class and OOD behavior:

   ```bash
   python -m src.model.experiment.runner --promote-selected
   ```

   To promote using one optimized threshold for all input lengths:

   ```bash
   python -m src.model.experiment.runner --promote-selected --threshold-policy global
   ```

4. This transfers `_PROD` to the new run folder, renames the run-local model to `best_pipeline_<run_name>_PROD.joblib`, and makes that run the API-serving source.
5. Run tests before deployment:

   ```bash
   python -m pytest
   ```

6. Roll out with Kubernetes rolling update behavior. Readiness probes keep pods without a loaded model out of service.

## Clean Parallel Model Package

`src/model/` contains the canonical selected prediction and training workflow.

- `src/model/predict/`: API-serving predictor wrapper with the same length-bucket `other` reject behavior.
- `src/model/train/`: thin training boundary with model fitting plus training config/validation.
- `src/model/tune/`: candidate CV, threshold tuning, tuning validation, and selection-objective helpers.
- `src/model/evaluate/`: prediction-row construction, policy application, and metrics.
- `src/model/experiment/`: clean experiment runner plus artifact manifest and write logic.
- `data/`: copied dataset and copied synthetic examples used by the clean package.
- `src/model/data_preparation/`: loading, split, leakage, and text-preparation helpers for the clean package.
- `best_pipeline_search_runs/`: output location for canonical experiment runs from the current modeling workflow.
- `config/model/train.yaml`, `config/model/tune.yaml`, and `config/model/predict.yaml`: YAML-first defaults for the clean package.
- `src/model/train/validators.py`, `src/model/tune/validators.py`, and `src/model/predict/validators.py`: per-module Pydantic validators and config loaders.

Run the clean training package with:

```bash
python -m src.model.experiment.runner
```

The clean runner loads YAML config by default and lets explicit CLI flags override selected fields. Each run writes both `runtime_config.yaml` and API-compatible `runtime_config.json`. With `--promote-selected`, it transfers the `_PROD` marker to the new run so the API serves that run.

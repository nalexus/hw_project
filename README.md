# Text Classification Case Study

## Solution At A Glance

This project classifies a document into one of ten known categories:
`business`, `entertainment`, `food`, `graphics`, `historical`, `medical`,
`politics`, `space`, `sport`, and `technologie`.

The deployed decision flow is:

```text
TF-IDF + Logistic Regression
    -> confidence and vocabulary-support gate
    -> predicted known class or other
```

The classifier is trained only on the ten known classes. `other` is reserved
for out-of-distribution (OOD) evaluation and API responses. A prediction is
accepted only when its maximum class probability is high enough and its share
of out-of-vocabulary (OOV) tokens is low enough; otherwise the API returns
`other`.

For a higher-capability production system, rejected documents would be routed
to a fine-tuned zero-shot/NLI transformer. That model could validate the
baseline's candidate topic or return `other`. The transformer experiment in
this repository is exploratory only and is not part of the deployed API.

[`exploration_solution.ipynb`](exploration_solution.ipynb) documents the
dataset analysis, baseline selection, OOD-policy design, and transformer
experiment.

All commands below assume the current directory is `hw_project`.

## Quick Start

### Set up the environment

The project targets Python 3.13. Install the locked project environment,
including notebook and test dependencies:

```powershell
uv sync --all-extras
```

### Audit the dataset

The optional audit reports invalid documents plus exact and near-duplicate
groups. Review its output before changing the configured exclusions.

```powershell
uv run python -m src.model.data_preparation.audit_document
```

To regenerate duplicate-exclusion candidates for review:

```powershell
uv run python -m src.model.data_preparation.audit_document --write-exclusions
```

This writes `config/model/excluded_files.yaml`; inspect the generated list
before using it for training.

### Train and promote a model

```powershell
uv run python -m src.model.tune.runner --promote-selected
```

The command writes a timestamped run under `best_pipeline_search_runs/` and
makes it the single serving run by adding the `_PROD` suffix.

### Run the API

```powershell
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>
- Liveness: `GET /health/live`
- Readiness: `GET /health/ready`

Example request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/classify_document" `
  -ContentType "application/json" `
  -Body '{"document_text":"The spacecraft entered lunar orbit after a six-day journey."}'
```

## Data, Modelling, And Evaluation

The workflow is derived from the notebook and implemented as a reproducible
training command:

1. `DatasetLoader` reads the dataset and removes reviewed duplicate files from
   `config/model/excluded_files.yaml`.
2. `other` is set aside and never used to fit or select the known-class model.
3. Known classes are split deterministically into class-stratified train and
   test sets. Text-length buckets are calculated for analysis, not for the
   production split.
4. Candidate TF-IDF and logistic-regression pipelines are compared with
   three-fold stratified cross-validation on the training split.
5. Out-of-fold predictions from the selected candidate tune the OOD policy:
   a probability threshold and maximum OOV ratio.
6. The selected model and policy are evaluated once on held-out known-class
   data and the reserved `other` documents.

The console and saved artifacts report known-class balanced accuracy, raw
accuracy, coverage, accepted accuracy, OOD rejection and OOD accuracy. Known
class results are also broken down by class and empirical length bucket.

The runner persists the selected pipeline fitted on the training split after
this evaluation. It does not refit the persisted production model on the test
split.

## API Contract

### `POST /classify_document`

Request body:

```json
{
  "document_text": "The team won the championship after extra time."
}
```

Successful response (`200 OK`):

```json
{
  "message": "Classification successful",
  "label": "sport",
  "model_version": "run_<timestamp>_PROD"
}
```

`label` is one of the ten known classes or `other`. `model_version` identifies
the promoted run that produced the prediction.

### Error Responses

| Status | Trigger | Response body |
| --- | --- | --- |
| `413` | `document_text` exceeds `max_document_length` | `{"detail":"document_text exceeds the configured maximum length of 10000 characters"}` |
| `415` | Missing or non-JSON `Content-Type` | `{"detail":"Content-Type must be application/json"}` |
| `422` | Malformed JSON | `{"detail":"Malformed JSON request body"}` |
| `422` | Missing, empty, non-string, or extra request fields | FastAPI validation body, for example `{"detail":[{"type":"missing","loc":["body","document_text"],"msg":"Field required","input":{}}]}` |
| `503` | Model is not ready or the prediction queue is full | `{"detail":"Model is not ready"}` or `{"detail":"Prediction queue is full"}` |
| `500` | Unexpected inference failure | `{"detail":"Unexpected inference failure"}` |

The maximum document length is configured in
`config/api/settings.yaml`; the `413` example reflects its current value.

### Health Endpoints

- `GET /health/live` returns `200` when the API process is running:

  ```json
  {"status":"live","model_version":"run_<timestamp>_PROD"}
  ```

- `GET /health/ready` returns the same shape with `"status":"ready"` after
  the model loads. It returns `503` with `{"detail":"Model is not ready"}`
  when the process is alive but cannot serve predictions.

## Runtime, Artifacts, And Deployment

The API discovers the one directory under `best_pipeline_search_runs/` whose
name ends in `_PROD`. It loads the model and its matching runtime policy; the
API never retrains a model.

Each selected run contains:

```text
model.joblib
runtime_config.json
candidate_cv_results.json
oof_policy_metrics.json
metrics_test.json
metrics_other.json
splits.json
```

`runtime_config.json` stores the model path and the OOD thresholds for that
exact fitted pipeline. The API uses `PredictorMultiClass` to calculate model
confidence and vocabulary support, then returns either the accepted class or
`other`.

The API batches nearby concurrent requests before invoking the predictor. This
keeps the synchronous endpoint simple while reducing repeated model calls under
load.

### Docker

The image packages the application, configuration, and promoted model run.

```powershell
docker build -t text-classifier:latest .
docker run --rm -p 8000:8000 text-classifier:latest
```

`RUNTIME_CONFIG_PATH` can select an explicit runtime configuration and
`MODEL_PATH` can override its model artifact. `MODEL_VERSION` only overrides
the response label; it does not select a model.

### Kubernetes

The manifests run two replicas behind the `text-classifier-api` service.
Readiness probes call `/health/ready`, so pods without a loaded model receive
no inference traffic. Liveness probes call `/health/live`.

After promoting a model, rebuild and redeploy the image so each pod receives
the new `_PROD` run.

For Docker Desktop Kubernetes:

```powershell
.\kubernetes\local_deploy\win_local.ps1 -PortForward
```

For a local Linux environment:

```bash
PORT_FORWARD=true ./kubernetes/local_deploy/linux_local.sh
```

## Tests

| Test type | Purpose | Command |
| --- | --- | --- |
| API | Request validation, error handling, batching, health checks, and OOD response behavior | `uv run python -m pytest tests/api -q` |
| Unit | Small model, data-preparation, evaluation, tuning, and prediction behavior | `uv run python -m pytest tests/model/unit -q` |
| Integration | End-to-end model workflow, persistence, and runtime loading | `uv run python -m pytest tests/model/integration/test_model_workflow.py -q` |
| Acceptance | Expected predictions for known categories across configured text lengths | `uv run python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_fixtures.py -q` and `uv run python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_behavior.py -q` |

The acceptance fixture data, generation prompt, and refresh command are stored
alongside the tests:

```powershell
uv run python -m tests.model.acceptance.data.i1a_known_class_by_length.generate_id_by_length_fixtures --fresh --workers 4 --request-timeout 180
```

## Limitations And Next Step

The current confidence-and-vocabulary gate is a low-cost rejection policy. It
improves safety for inputs that look unlike supported training data, but it is
not a semantic guarantee that every unrelated document will be rejected.

The next production step would route only deferred requests to a fine-tuned
zero-shot/NLI transformer. The transformer would evaluate the baseline's top
candidate topic and decide whether to accept it or return `other`. This adds
latency and compute cost, so it should be used as a fallback rather than for
every request. The notebook includes a zero-shot experiment to evaluate this
direction, but no transformer is used by the deployed API.

## Project Map

```text
exploration_solution.ipynb     Analysis, evaluation, and transformer experiment
config/                        Model and API configuration
src/model/                     Reproducible training, tuning, evaluation, and prediction
src/api/                       FastAPI application and request batching
tests/                         API, unit, integration, and acceptance tests
data/dataset/                  Labeled source documents
best_pipeline_search_runs/     Persisted runs; one directory is marked *_PROD
kubernetes/                    Deployment, service, and local deployment helpers
```

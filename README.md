# Text Classification Case Study

## Quick Start

All commands below assume the current directory is `hw_project`.

### Project Structure

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

### Set up the environment

The project targets Python 3.13. For the complete reviewer environment,
including tests, Jupyter, the zero-shot experiment, and a CPU-only PyTorch
build, run:

```powershell
uv sync --extra jupyter --extra dev --extra torch-cpu
```

This is the default installation path. It works without a CUDA-capable GPU and
is sufficient for every production command, test, and notebook cell.

For local GPU experimentation only, this machine's NVIDIA driver supports the
CUDA 13.0 PyTorch build. Use the CUDA extra instead of the CPU extra:

```powershell
uv sync --extra jupyter --extra dev --extra torch-cu130
```

Start the notebook with the same CPU extras:

```powershell
uv run --extra jupyter --extra torch-cpu jupyter lab exploration_solution.ipynb
```

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

```bash
curl --request POST \
  --url http://127.0.0.1:8000/classify_document \
  --header 'Content-Type: application/json' \
  --data '{"document_text":"The spacecraft entered lunar orbit after a six-day journey."}'
```

### API Runtime

```text
Client request
    -> FastAPI POST /classify_document
    -> per-pod micro-batch queue (up to 100 ms or 64 documents)
    -> TF-IDF model and OOD policy
    -> one response per original request

Docker image (API + configuration + promoted *_PROD run)
    -> Kubernetes Deployment (two API pods)
    -> text-classifier-api Service
```

Each FastAPI pod accepts one document per request. Its local batcher briefly
collects nearby requests, runs one vectorized predictor call, then returns each
result to its original caller. The batch delay, batch size, and queue capacity
are configurable in `config/api/settings.yaml`. Docker packages the serving
application and promoted model; Kubernetes runs two replicas and sends traffic
only to pods whose `/health/ready` check succeeds.

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

The following extension is not deployed, but is the intended next decision
step for requests classified as `other`:

```text
TF-IDF + Logistic Regression predicts other
    -> fine-tuned small zero-shot/NLI transformer (ModernBERT-base-nli)
    -> validates other or the TF-IDF top-1 known class
```

The classifier is trained only on the ten known classes. `other` is reserved
for out-of-distribution (OOD) evaluation and API responses. A prediction is
accepted only when its maximum class probability is high enough and its share
of out-of-vocabulary (OOV) tokens is low enough; otherwise the API returns
`other`.

### Synthetic Golden Dataset

The acceptance tests include a generated, test-only golden dataset. It contains
50 distinct English documents: one document for each of the ten known
classes across all five empirical length buckets. They were generated with
`gpt-5-mini` through the OpenAI Responses API.

The generation prompt supplies the target class, target word range, length
bucket, and a stable seed for each class-bucket cell. It requests an original,
topic-specific document in structured JSON, while prohibiting the literal class
name and meta descriptions. Each fixture records its model, prompt version,
seed, word count, and expected label. Integrity tests verify the complete
class-bucket matrix, token and bucket assignments, fixture uniqueness, and no
exact or near-copy overlap with the loaded source dataset.

The fixtures were not used for fitting, candidate selection, or OOD-policy
tuning. The current promoted model achieves `60.00%` overall and balanced
accuracy on this golden set (`30/50` correct), below the configured `90%`
acceptance target. This materially weakens the apparently strong held-out
dataset result: the baseline and OOD policy are not ready to be described as
real-world production ready.

For a higher-capability production system, rejected documents would be routed
to a fine-tuned zero-shot/NLI transformer. That model could validate the
baseline's candidate topic or return `other`. The transformer experiment in
this repository is exploratory only and is not part of the deployed API.

[`exploration_solution.ipynb`](exploration_solution.ipynb) documents the
dataset analysis, baseline selection, OOD-policy design, and transformer
experiment.

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
| API | Request validation, error handling, batching, health checks, and OOD response behavior | `uv run --extra dev python -m pytest tests/api -q` |
| Unit | Small model, data-preparation, evaluation, tuning, and prediction behavior | `uv run --extra dev python -m pytest tests/model/unit -q` |
| Integration | End-to-end model workflow, persistence, and runtime loading | `uv run --extra dev python -m pytest tests/model/integration/test_model_workflow.py -q` |
| Fixture integrity | Validates the generated dataset's structure, length buckets, and independence from source records | `uv run --extra dev python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_fixtures.py -q` |
| Golden behavior | Evaluates the promoted pipeline and OOD policy against the generated dataset | `uv run --extra dev python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_behavior.py -q` |

The behavior test is intentionally a production-readiness gate, not a test that
currently passes. Its configured thresholds are 90% overall and balanced
accuracy, plus 80% accuracy for every class and length bucket. The current run
scores 60% overall and balanced accuracy, so it fails the gate. This is the
expected result until the model improves.

The fixture data, generation prompt, and refresh command are stored alongside
the acceptance tests. Regeneration uses a 50-cell class-by-length matrix and
records the `gpt-5-mini` model and prompt-level seed in every row:

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

Another experiment is synthetic augmentation with new, disjoint generation
seeds. Additional known-class documents could be reviewed and added to the
training data. Additional OOD documents could improve threshold tuning and
evaluation, and could train the future transformer fallback. They must not be
used as an eleventh `other` training class for the current TF-IDF classifier,
because its contract deliberately trains only on known classes. The committed
50-item golden dataset must remain untouched and separate from any augmentation
corpus. Synthetic data can broaden coverage, but it does not replace externally
collected and labelled production data.

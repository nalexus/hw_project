# Synthetic Data Generation

This document describes how `data/synthetic_samples/` was produced and what logic it is meant to preserve.

## Source

The file-backed synthetic sample tree was generated from the previous in-code synthetic dataset:

- `data/synthetic_examples.py`
- `SYNTHETIC_KNOWN_EXAMPLES`
- `SYNTHETIC_OOD_EXAMPLES`

The old Python file was used as the migration source. The file-backed tree is now the canonical synthetic data source.

## Directory Layout

Synthetic samples now live under:

```text
data/synthetic_samples/<class>/<length_bucket>/<split>_<id>.txt
```

Examples:

```text
data/synthetic_samples/business/ultra_short/train_001.txt
data/synthetic_samples/medical/ultra_short/validation_002.txt
data/synthetic_samples/other/medium/validation_ood_val_weather_001.txt
data/synthetic_samples/other/long/test_ood_test_roof_001.txt
```

The first directory level is the model class:

- Known classes: `business`, `entertainment`, `food`, `graphics`, `historical`, `medical`, `politics`, `space`, `sport`, `technologie`
- OOD class: `other`

The second directory level is the computed length bucket:

```text
ultra_short: 0-19 words
short:       20-120 words
medium:      121-435 words
long:        436-738 words
extra_long:  739+ words
```

These buckets come from `src/model/train/constants.py` and are computed using the shared tokenizer from `src/model/data_preparation/text.py`.

## Split Logic

Split hints are stored in the filename prefix:

```text
train_...
validation_...
test_...
```

Known synthetic examples may use all three splits.

Synthetic OOD examples may only use:

- `validation`
- `test`

They must not use `train`, because `other` rejection is tuned and evaluated, not trained as a normal known class.

## Record ID Preservation

The file-backed loader preserves the same record ID shape as the old Python constants.

Known examples:

```text
synthetic_known:<class>:<split>:<id>
```

Example:

```text
synthetic_known:business:train:001
```

OOD examples:

```text
synthetic_ood:<split>:<id>
```

Example:

```text
synthetic_ood:validation:ood_val_weather_001
```

This keeps split manifests, prediction rows, and metric grouping comparable across the old and new loaders.

## Generation Logic

The conversion logic was:

1. Read every known synthetic example from `SYNTHETIC_KNOWN_EXAMPLES`.
2. For each example, compute its word count using the shared tokenizer.
3. Map the word count to a length bucket.
4. Write the text to:

   ```text
   data/synthetic_samples/<label>/<computed_bucket>/<split>_<index>.txt
   ```

5. Read every synthetic OOD example from `SYNTHETIC_OOD_EXAMPLES`.
6. Compute its length bucket in the same way.
7. Write the text to:

   ```text
   data/synthetic_samples/other/<computed_bucket>/<split>_<original_ood_id>.txt
   ```

The conversion did not rewrite, extend, shorten, paraphrase, or regenerate any sample text. It only moved the existing text examples into files and directories.

## Current Distribution

The generated tree currently contains 368 synthetic text files:

```text
business       20
entertainment  20
food           20
graphics       20
historical     20
medical        20
politics       20
space          20
sport          20
technologie    20
other          168
```

Important caveat: all known-class synthetic examples currently fall into `ultra_short`.

```text
business       ultra_short 20
entertainment  ultra_short 20
food           ultra_short 20
graphics       ultra_short 20
historical     ultra_short 20
medical        ultra_short 20
politics       ultra_short 20
space          ultra_short 20
sport          ultra_short 20
technologie    ultra_short 20
```

The OOD examples are length-balanced more broadly:

```text
other ultra_short 13
other short       44
other medium      68
other long        29
other extra_long  14
```

This means the directory structure is ready for length-aware synthetic known examples, but the current known synthetic source data does not yet provide them.

## Loader Behavior

`DatasetLoader` now reads synthetic examples from `data/synthetic_samples/`.

- Synthetic examples are loaded from files.
- The class comes from the first directory level.
- The declared length bucket comes from the second directory level.
- The split comes from the filename prefix.
- The loader validates that the declared length bucket matches the computed bucket.
- Synthetic `other` files with a `train_` prefix are rejected.

If `data/synthetic_samples/` does not exist and synthetic data is enabled, loading fails with a clear missing-directory error.

## Intended Next Step

Decide whether to add longer known synthetic examples per class and length bucket. Without that, the new layout preserves the old behavior but does not improve known-class length coverage.

## OpenAI-Generated Model Behavior Fixtures

Separate from the training-time `data/synthetic_samples/` tree, the test suite contains
OpenAI-generated behavior fixtures under:

```text
tests/model/acceptance/data/i1a_known_class_by_length/fixtures.jsonl
tests/model/acceptance/data/i1a_known_class_by_length/prompt.md
```

These rows are test-only fixtures for I.1.A. They are not loaded by the training
dataset loader and do not affect model fitting. Their purpose is to check that the
currently promoted `_PROD` model can identify every known class across every
runtime length bucket.

The fixture matrix contains 50 rows:

```text
10 known classes x 5 length buckets = 50 examples
```

The known classes are:

```text
business
entertainment
food
graphics
historical
medical
politics
space
sport
technologie
```

The length buckets use the same runtime tokenizer and ranges as prediction:

```text
ultra_short: 0-19 tokens
short:       20-120 tokens
medium:      121-435 tokens
long:        436-738 tokens
extra_long:  739+ tokens
```

### Generator

The optional generator lives at:

```text
tests/tools/generate_id_by_length_fixtures.py
```

It reads the OpenAI key from either:

- `OPENAI_API_KEY`
- a local `.synth_gen` file at the project root or repository root

The generator uses `gpt-5-mini` by default and writes JSONL rows to
`tests/model/acceptance/data/i1a_known_class_by_length/fixtures.jsonl`. The committed tests do not need
OpenAI API access because they read the generated fixture file.

Run a fresh independent refresh with:

```bash
python -m tests.tools.generate_id_by_length_fixtures --fresh --workers 4 --request-timeout 180
```

If a specific row needs to be replaced for a fixture-quality reason, such as
malformed text or wrong bucket, use its record ID:

```bash
python -m tests.tools.generate_id_by_length_fixtures \
  --workers 1 \
  --request-timeout 240 \
  --max-attempts 8 \
  --replace-record-id i1a:medical:ultra_short
```

On PowerShell, use backticks instead of backslashes for line continuation.

### Validation Rules

Each generated row is accepted only if local validation passes:

- JSON is well formed and contains the required fixture fields.
- `record_id` matches `i1a:<expected_label>:<length_bucket>`.
- `expected_label` is one of the 10 known classes.
- The computed runtime length bucket matches the stored bucket.
- The stored word count matches the shared runtime tokenizer.
- The normalized text is not duplicated in the fixture file.
- The expected class label does not appear as a standalone token in the text.

The generator does not load the promoted model and does not use model
predictions to accept or reject rows. A valid generated row must be kept even if
the current `_PROD` model misclassifies it. In that case, the behavior test
failure is a model finding, not a reason to regenerate the row.

### Why Generation Took Multiple Runs

The first implementation generated the full 50-row matrix serially: one OpenAI
API call per `(known_class, length_bucket)` pair. That approach was valid but
too slow for an interactive run because 20 of the rows are long-form requests:

```text
10 long rows:       470-560 target words
10 extra_long rows: 760-850 target words
```

The first long run also wrote the fixture file only after all rows completed, so
an interruption or timeout could discard otherwise valid rows. A later version
added checkpointing, but it was still serial enough that one slow or malformed
response made the whole process feel stuck.

The generator was changed to avoid that failure mode:

- It keeps valid rows from an existing partial JSONL file.
- It generates missing rows concurrently with a bounded worker count.
- It applies a per-request timeout and a maximum attempt count.
- It checkpoints each successful row immediately.
- It prints progress as rows complete.
- It can regenerate only specific fixture-quality failures by record ID.

An earlier version incorrectly used model-gated regeneration for a few weak
rows. That made the resulting behavior test circular: some inputs were selected
because the current model already passed them. The current fixture workflow
removes that acceptance path. OpenAI generation is allowed to retry only for
local fixture validity failures, not for prediction failures.

### Tests

Fixture integrity tests:

```bash
python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_fixtures.py -v
```

They verify:

- the fixture file has exactly 50 rows;
- every known class and length bucket pair is present;
- tokenizer-derived metadata is correct;
- literal expected-label leakage is absent;
- fixture texts are not exact or near copies of loaded project records.

Behavior tests:

```bash
python -m pytest tests/model/acceptance/test_i1a_known_class_by_length_behavior.py -v
```

They run exactly 50 prediction checks. Each case passes only if the current
promoted `_PROD` predictor returns the expected known class for that independent
fixture row.

Run the related focused suite with:

```bash
python -m pytest tests/model/unit/test_predictor_policy.py tests/model/integration/test_clean_model_package.py tests/model/unit/test_synthetic_samples_loader.py tests/api/test_api_endpoints.py
```

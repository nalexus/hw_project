"""Generate I.1.A known-class length-bucket behavior fixtures."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.model.data_preparation.text import TextProcessor


FIXTURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIXTURE_DIR.parents[4]
FIXTURE_PATH = FIXTURE_DIR / "fixtures.jsonl"
PROMPT_PATH = FIXTURE_DIR / "prompt.md"
TEXT_PROCESSOR = TextProcessor.from_config(PROJECT_ROOT / "config" / "model" / "text.yaml")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5-mini"
PROMPT_VERSION = "i1a-known-class-length-v1"
KNOWN_LABELS = (
    "business",
    "entertainment",
    "food",
    "graphics",
    "historical",
    "medical",
    "politics",
    "space",
    "sport",
    "technologie",
)
BUCKETS = (
    ("ultra_short", 12, 16),
    ("short", 50, 80),
    ("medium", 150, 220),
    ("long", 470, 560),
    ("extra_long", 780, 850),
)
REQUIRED_FIELDS = {
    "record_id",
    "expected_label",
    "length_bucket",
    "word_count",
    "generation_model",
    "prompt_version",
    "generation_seed",
    "text",
}


@dataclass(frozen=True)
class FixtureTask:
    """Generation target for one known label and length bucket."""

    label: str
    bucket: str
    min_words: int
    max_words: int


def main() -> None:
    """Generate and write all fixture rows after local validation."""

    args = parse_args()
    api_key = read_api_key()
    rows = generate_rows(
        api_key,
        args.model,
        args.max_attempts,
        args.output,
        args.workers,
        args.request_timeout,
        args.fresh,
        set(args.replace_record_id),
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} fixture rows to {args.output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible fixture refreshes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--replace-record-id", action="append", default=[])
    return parser.parse_args()


def read_api_key() -> str:
    """Read the OpenAI API key from environment or local config."""

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    for path in (PROJECT_ROOT / ".synth_gen", PROJECT_ROOT.parent / ".synth_gen"):
        key = api_key_from_file(path)
        if key:
            return key
    raise RuntimeError("Set OPENAI_API_KEY or create local .synth_gen with the key.")


def api_key_from_file(path: Path) -> str | None:
    """Extract an API key from a simple local .synth_gen file."""

    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            return line
        name, value = line.split("=", 1)
        if name.strip().upper() in {"OPENAI_API_KEY", "API_KEY"}:
            return value.strip().strip("\"'")
    return None


def generate_rows(
    api_key: str,
    model: str,
    max_attempts: int,
    output: Path,
    workers: int,
    request_timeout: int,
    fresh: bool,
    replace_record_ids: set[str],
) -> list[dict]:
    """Generate one locally valid row for every label and length bucket."""

    rows_by_id = {} if fresh else load_valid_existing_rows(output)
    for record_id in sorted(replace_record_ids):
        rows_by_id.pop(record_id, None)
    missing_tasks = [task for task in expected_tasks() if task_id(task) not in rows_by_id]
    print(f"kept {len(rows_by_id)}/50 valid rows; generating {len(missing_tasks)}", flush=True)
    if missing_tasks:
        generate_missing_rows(
            api_key,
            model,
            max_attempts,
            request_timeout,
            workers,
            output,
            rows_by_id,
        )
    rows = sorted_rows(rows_by_id.values())
    if len(rows) != len(expected_tasks()):
        raise RuntimeError(f"Fixture generation incomplete: {len(rows)}/50 rows")
    return rows


def load_existing_rows(path: Path) -> list[dict]:
    """Load already generated fixture rows so interrupted runs can resume."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_valid_existing_rows(path: Path) -> dict[str, dict]:
    """Return valid checkpoint rows keyed by record ID."""

    rows_by_id: dict[str, dict] = {}
    seen_texts: set[str] = set()
    for row in load_existing_rows(path):
        try:
            validate_row(row, seen_texts)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"discarded invalid row {row.get('record_id', '<missing>')}: {exc}", flush=True)
            continue
        rows_by_id[row["record_id"]] = row
        seen_texts.add(normalized_text(row["text"]))
    return rows_by_id


def expected_tasks() -> list[FixtureTask]:
    """Return the complete I.1.A generation matrix."""

    return [
        FixtureTask(label, bucket, min_words, max_words)
        for label in KNOWN_LABELS
        for bucket, min_words, max_words in BUCKETS
    ]


def generate_missing_rows(
    api_key: str,
    model: str,
    max_attempts: int,
    request_timeout: int,
    workers: int,
    output: Path,
    rows_by_id: dict[str, dict],
) -> None:
    """Generate missing rows concurrently and checkpoint each success."""

    start = time.monotonic()
    failures: list[str] = []
    seen_texts = {normalized_text(row["text"]) for row in rows_by_id.values()}
    tasks = [task for task in expected_tasks() if task_id(task) not in rows_by_id]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(generate_task_row, api_key, model, task, max_attempts, request_timeout): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
                validate_row(row, seen_texts)
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{task_id(task)}: {exc}")
                print(f"failed {task_id(task)}: {exc}", flush=True)
                continue
            rows_by_id[row["record_id"]] = row
            seen_texts.add(normalized_text(row["text"]))
            write_jsonl(output, sorted_rows(rows_by_id.values()))
            count = len(rows_by_id)
            elapsed = int(time.monotonic() - start)
            print(f"completed {count}/50 {row['record_id']} in {elapsed}s", flush=True)
    if failures:
        raise RuntimeError("Fixture generation failures:\n" + "\n".join(failures))


def generate_task_row(
    api_key: str,
    model: str,
    task: FixtureTask,
    max_attempts: int,
    request_timeout: int,
) -> dict:
    """Generate and package one task row."""

    seed = f"{PROMPT_VERSION}:{task.label}:{task.bucket}"
    text = generate_valid_text(
        api_key,
        model,
        task.label,
        task.bucket,
        task.min_words,
        task.max_words,
        seed,
        set(),
        max_attempts,
        request_timeout,
    )
    return build_row(task.label, task.bucket, text, model, seed)


def generate_valid_text(
    api_key: str,
    model: str,
    label: str,
    bucket: str,
    min_words: int,
    max_words: int,
    seed: str,
    seen_texts: set[str],
    max_attempts: int,
    request_timeout: int,
) -> str:
    """Regenerate a single row until validation succeeds or attempts run out."""

    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        prompt = render_prompt(model, label, bucket, min_words, max_words, seed, attempt)
        try:
            text = request_fixture_text(api_key, model, prompt, request_timeout)
            validate_text(text, label, bucket, seen_texts)
            return text
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
    raise RuntimeError(f"Could not generate {label}/{bucket}: {'; '.join(errors)}")


def render_prompt(
    model: str, label: str, bucket: str, min_words: int, max_words: int, seed: str, attempt: int
) -> str:
    """Fill the reviewer-facing prompt template for one fixture row."""

    template = PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{model}}": model,
        "{{prompt_version}}": PROMPT_VERSION,
        "{{known_labels}}": ", ".join(KNOWN_LABELS),
        "{{expected_label}}": label,
        "{{length_bucket}}": bucket,
        "{{min_words}}": str(min_words),
        "{{max_words}}": str(max_words),
        "{{generation_seed}}": seed,
        "{{attempt}}": str(attempt),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def request_fixture_text(api_key: str, model: str, prompt: str, request_timeout: int) -> str:
    """Call OpenAI Responses API and parse the returned JSON text field."""

    payload = {
        "model": model,
        "input": [
            {"role": "developer", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": 5000,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "fixture_text",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            }
        },
    }
    request = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=request_timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed with HTTP {exc.code}: {detail}") from exc
    return parse_response_text(response_payload)


def parse_response_text(response_payload: dict) -> str:
    """Extract the fixture text from a Responses API JSON payload."""

    output_text = response_payload.get("output_text")
    if isinstance(output_text, str):
        return parse_json_object(output_text)["text"]
    for output in response_payload.get("output", []):
        for content in output.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                return parse_json_object(text)["text"]
    raise ValueError("OpenAI response did not include output text.")


def parse_json_object(raw_text: str) -> dict:
    """Parse a JSON object, tolerating accidental markdown fences."""

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    if not cleaned.startswith("{"):
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("text"), str):
        raise ValueError("Generated output must be a JSON object with a string text field.")
    return parsed


def validate_text(text: str, expected_label: str, expected_bucket: str, seen_texts: set[str]) -> None:
    """Reject malformed text, wrong buckets, duplicates, and label leakage."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty text")
    if normalized_text(stripped) in seen_texts:
        raise ValueError("duplicate normalized text")
    actual_bucket = TEXT_PROCESSOR.length_bucket(stripped)
    if actual_bucket != expected_bucket:
        raise ValueError(f"bucket {actual_bucket!r} does not match {expected_bucket!r}")
    tokens = {token.lower() for token in TEXT_PROCESSOR.tokenize(stripped)}
    if expected_label.lower() in tokens:
        raise ValueError(f"literal label {expected_label!r} appears as a standalone token")


def validate_row(row: dict, seen_texts: set[str]) -> None:
    """Validate one JSONL fixture row and its derived fields."""

    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    expected_id = f"i1a:{row['expected_label']}:{row['length_bucket']}"
    if row["record_id"] != expected_id:
        raise ValueError(f"record_id {row['record_id']!r} does not match {expected_id!r}")
    if row["expected_label"] not in KNOWN_LABELS:
        raise ValueError(f"unsupported label {row['expected_label']!r}")
    if row["length_bucket"] not in bucket_names():
        raise ValueError(f"unsupported bucket {row['length_bucket']!r}")
    if row["word_count"] != TEXT_PROCESSOR.word_count(row["text"]):
        raise ValueError("stored word_count does not match text")
    validate_text(row["text"], row["expected_label"], row["length_bucket"], seen_texts)


def normalized_text(text: str) -> str:
    """Normalize text for stable duplicate detection."""

    return " ".join(token.lower() for token in TEXT_PROCESSOR.tokenize(text))


def build_row(label: str, bucket: str, text: str, model: str, seed: str) -> dict:
    """Build one JSONL fixture row with traceability metadata."""

    stripped = text.strip()
    return {
        "record_id": f"i1a:{label}:{bucket}",
        "expected_label": label,
        "length_bucket": bucket,
        "word_count": TEXT_PROCESSOR.word_count(stripped),
        "generation_model": model,
        "prompt_version": PROMPT_VERSION,
        "generation_seed": seed,
        "text": stripped,
    }


def task_id(task: FixtureTask) -> str:
    """Return the stable record ID for one task."""

    return f"i1a:{task.label}:{task.bucket}"


def bucket_names() -> set[str]:
    """Return supported fixture bucket names."""

    return {bucket for bucket, _, _ in BUCKETS}


def sorted_rows(rows: Iterable[dict]) -> list[dict]:
    """Sort rows by configured label and bucket order."""

    label_order = {label: index for index, label in enumerate(KNOWN_LABELS)}
    bucket_order = {bucket: index for index, (bucket, _, _) in enumerate(BUCKETS)}
    return sorted(
        list(rows),
        key=lambda row: (label_order[row["expected_label"]], bucket_order[row["length_bucket"]]),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write fixture rows as UTF-8 JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

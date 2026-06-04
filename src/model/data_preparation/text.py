"""Text helpers for dataset loading, splitting, and leakage checks."""

from __future__ import annotations

import hashlib

from src.model.train.constants import LENGTH_BUCKETS, TOKEN_RE


def tokenize(text: str) -> list[str]:
    """Tokenize text with the shared training/runtime token rule."""

    return TOKEN_RE.findall(text)


def length_bucket_for_count(word_count: int) -> str:
    """Map a word count to the configured length bucket."""

    for name, lower, upper in LENGTH_BUCKETS:
        if word_count >= lower and (upper is None or word_count <= upper):
            return name
    raise ValueError(f"Unsupported word count: {word_count}")


def normalized_text(text: str) -> str:
    """Normalize text for duplicate checks and stable hashing."""

    return " ".join(token.lower() for token in tokenize(text))


def hash_text(text: str) -> str:
    """Return a stable SHA-256 hash for normalized text content."""

    normalized = normalized_text(text).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def shingle_set(text: str, size: int = 5) -> set[str]:
    """Return token shingles for near-duplicate checks."""

    tokens = tokenize(text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity between two sets."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

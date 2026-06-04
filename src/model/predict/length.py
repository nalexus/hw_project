"""Length-bucket helpers shared by runtime prediction."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
LENGTH_BUCKETS = (
    ("ultra_short", 0, 19),
    ("short", 20, 120),
    ("medium", 121, 435),
    ("long", 436, 738),
    ("extra_long", 739, None),
)


def word_count(text: str) -> int:
    """Count tokens with the training/runtime token rule."""

    return len(TOKEN_RE.findall(text))


def length_bucket(text: str) -> str:
    """Map document text to the configured runtime length bucket."""

    count = word_count(text)
    for name, lower, upper in LENGTH_BUCKETS:
        if count >= lower and (upper is None or count <= upper):
            return name
    raise ValueError(f"Unsupported word count: {count}")

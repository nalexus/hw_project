"""Configured tokenization and quantile-derived document-length buckets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class LengthBucketRule:
    """Configured inclusive word-count range for one length bucket."""

    name: str
    min_words: int
    max_words: int | None

    def contains(self, word_count: int) -> bool:
        """Return whether the count belongs to this bucket."""

        return word_count >= self.min_words and (
            self.max_words is None or word_count <= self.max_words
        )


@dataclass(frozen=True)
class TextRules:
    """Configured tokenization and length-bucket rules."""

    token_pattern: str
    length_buckets: tuple[LengthBucketRule, ...]


class TextProcessor:
    """Tokenize text and assign its configured length bucket."""

    def __init__(self, rules: TextRules) -> None:
        """Compile the configured token pattern once for repeated text processing."""

        self.rules = rules
        self.token_re = re.compile(rules.token_pattern)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "TextProcessor":
        """Build a processor from the configured tokenizer and bucket rules."""

        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        rules = TextRules(
            token_pattern=str(config["token_pattern"]),
            length_buckets=tuple(
                LengthBucketRule(
                    name=str(item["name"]),
                    min_words=int(item["min_words"]),
                    max_words=None
                    if item.get("max_words") is None
                    else int(item["max_words"]),
                )
                for item in config["length_buckets"]
            )
        )
        cls._validate_rules(rules)
        return cls(rules)

    @staticmethod
    def _validate_rules(rules: TextRules) -> None:
        """Validate that configured length buckets are ordered and non-overlapping."""

        expected_min = 0
        for bucket in rules.length_buckets:
            if bucket.min_words != expected_min:
                raise ValueError(f"Length bucket gap before {bucket.name}")
            if bucket.max_words is None:
                return
            if bucket.max_words < bucket.min_words:
                raise ValueError(f"Invalid length bucket range: {bucket.name}")
            expected_min = bucket.max_words + 1
        raise ValueError("Last length bucket must have max_words: null")

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with the configured shared token pattern."""

        return self.token_re.findall(text)

    def length_bucket_for_count(self, word_count: int) -> str:
        """Map a word count to its configured length bucket."""

        for bucket in self.rules.length_buckets:
            if bucket.contains(word_count):
                return bucket.name
        raise ValueError(f"Unsupported word count: {word_count}")

    def word_count(self, text: str) -> int:
        """Count tokens using the configured tokenization rule."""

        return len(self.tokenize(text))

    def length_bucket(self, text: str) -> str:
        """Map document text to its configured length bucket."""

        return self.length_bucket_for_count(self.word_count(text))

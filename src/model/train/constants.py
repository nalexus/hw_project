"""Shared constants for the clean training package."""

import re


OTHER_LABEL = "other"
PROVIDED_KNOWN = "provided_known"
PROVIDED_OTHER = "provided_other"
SYNTHETIC_KNOWN = "synthetic_known"
SYNTHETIC_OOD = "synthetic_ood"
SYNTHETIC_DATA_VERSION = "part1_synthetic_v3_ood_length_balance_file_tree"
SYNTHETIC_PROMPT_VERSION = "manual_curated_no_llm_v3"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
LENGTH_BUCKETS = (
    ("ultra_short", 0, 19),
    ("short", 20, 120),
    ("medium", 121, 435),
    ("long", 436, 738),
    ("extra_long", 739, None),
)

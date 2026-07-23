"""Audit invalid documents and review duplicate-exclusion candidates."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import re

import pandas as pd
import yaml


class InvalidDocumentAuditor:
    """Load text documents and apply vectorized validity checks."""

    FLAG_COLUMNS = ["is_empty", "is_too_short", "has_non_printable"]
    OUTPUT_COLUMNS = [
        "class", "file", "characters", "word_count", "non_printable_ratio",
        "reasons", "preview",
    ]

    def __init__(
        self,
        min_word_count: int = 5,
        max_non_printable_ratio: float = 0.05,
        preview_length: int = 160,
    ) -> None:
        """Store thresholds used to identify invalid documents."""

        if min_word_count < 0:
            raise ValueError("min_word_count must be non-negative")
        if not 0 <= max_non_printable_ratio <= 1:
            raise ValueError("max_non_printable_ratio must be between 0 and 1")
        if preview_length < 0:
            raise ValueError("preview_length must be non-negative")
        self.min_word_count = min_word_count
        self.max_non_printable_ratio = max_non_printable_ratio
        self.preview_length = preview_length

    def audit(self, dataset_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return all flagged documents and the invalid subset."""

        documents = self.load_documents(dataset_root)
        return self.flag_invalid_documents(self._measure_documents(documents))

    def load_documents(self, dataset_root: str | Path) -> pd.DataFrame:
        """Read each class text file once into a document-level DataFrame."""

        paths = sorted(Path(dataset_root).glob("*/*.txt"))
        return pd.DataFrame(
            [
                {
                    "class": path.parent.name,
                    "file": path.name,
                    "text": path.read_text(encoding="utf-8", errors="replace"),
                }
                for path in paths
            ],
            columns=["class", "file", "text"],
        )

    def _measure_documents(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Calculate document statistics without deciding validity."""

        result = documents.copy()
        text = result["text"].astype("string")
        result["characters"] = text.str.len()
        result["word_count"] = text.str.count(r"\b[A-Za-z]+\b")
        non_printable_count = text.str.count(r"[^\x20-\x7E\r\n\t]")
        denominator = result["characters"].replace(0, pd.NA)
        result["non_printable_ratio"] = non_printable_count.div(denominator).fillna(0.0)
        result["preview"] = (
            text.str.replace(r"\s+", " ", regex=True).str.strip().str[: self.preview_length]
        )
        return result

    def flag_invalid_documents(
        self, measured: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply validity rules, add reasons, and return invalid rows."""

        result = measured.copy()
        result["is_empty"] = result["text"].astype("string").str.strip().eq("")
        result["is_too_short"] = result["word_count"].lt(self.min_word_count)
        result["has_non_printable"] = result["non_printable_ratio"].gt(
            self.max_non_printable_ratio
        )
        result["reasons"] = self._build_reasons(result)
        invalid_mask = result[self.FLAG_COLUMNS].any(axis=1)
        invalid = result.loc[invalid_mask, self.OUTPUT_COLUMNS].reset_index(drop=True)
        return result, invalid

    def _build_reasons(self, documents: pd.DataFrame) -> pd.Series:
        """Create readable reason labels from the current validity flags."""

        ratio_percent = self.max_non_printable_ratio * 100
        reasons = (
            documents["is_empty"].map({True: "empty; ", False: ""})
            + documents["is_too_short"].map(
                {True: f"fewer_than_{self.min_word_count}_words; ", False: ""}
            )
            + documents["has_non_printable"].map(
                {True: f"more_than_{ratio_percent:g}_percent_non_printable; ", False: ""}
            )
        )
        return reasons.str.removesuffix("; ")


class DuplicateDocumentAuditor:
    """Find exact and near duplicates, then produce notebook review tables."""

    TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
    GROUP_COLUMNS = [
        "member_count",
        "classes",
        "files",
        "previews",
        "cross_class",
        "overlap_coefficients",
    ]

    def __init__(
        self,
        shingle_size: int = 5,
        near_duplicate_jaccard: float = 0.92,
        overlap_threshold: float = 0.95,
        min_smaller_shingles: int = 20,
        preview_length: int = 160,
    ) -> None:
        """Store near-duplicate matching thresholds."""

        self.shingle_size = shingle_size
        self.near_duplicate_jaccard = near_duplicate_jaccard
        self.overlap_threshold = overlap_threshold
        self.min_smaller_shingles = min_smaller_shingles
        self.preview_length = preview_length

    def audit(
        self,
        dataset_root: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return documents, exact-duplicate groups, and near-duplicate groups."""

        documents = self._load_documents(dataset_root)
        text = documents["text"].astype("string")
        documents["normalized_text"] = text.map(self._normalize)
        documents["token_count"] = documents["normalized_text"].str.count(r"\S+")
        documents["preview"] = (
            text.str.replace(r"\s+", " ", regex=True).str.strip().str[: self.preview_length]
        )
        return documents, self._exact_groups(documents), self._near_groups(documents)

    def names_excluding_largest(
        self,
        documents: pd.DataFrame,
        duplicate_groups: pd.DataFrame,
    ) -> list[str]:
        """Return duplicate filenames while retaining the largest document per group."""

        token_counts = documents.set_index(["class", "file"])["token_count"].to_dict()
        excluded = []
        for group in duplicate_groups.itertuples(index=False):
            members = list(zip(group.classes, group.files))
            retained = min(
                members,
                key=lambda member: (-token_counts[member], member[0], member[1]),
            )
            excluded.extend(
                file_name
                for member_class, file_name in members
                if (member_class, file_name) != retained
            )
        return sorted(set(excluded))

    def _exact_groups(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Summarize groups with identical normalized text."""

        rows = [
            self._group_row(group, [1.0] * (len(group) * (len(group) - 1) // 2))
            for _, group in documents.groupby("normalized_text", dropna=False)
            if len(group) > 1
        ]
        return pd.DataFrame(rows, columns=self.GROUP_COLUMNS)

    def _near_groups(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Find connected components of non-exact near-duplicate pairs."""

        rows = list(documents.itertuples())
        shingles = [self._shingles(row.text) for row in rows]
        edges = []
        for left_index, right_index in combinations(range(len(rows)), 2):
            left, right = rows[left_index], rows[right_index]
            if left.normalized_text == right.normalized_text:
                continue
            shared = len(shingles[left_index] & shingles[right_index])
            smaller_count = min(len(shingles[left_index]), len(shingles[right_index]))
            jaccard = shared / len(shingles[left_index] | shingles[right_index])
            overlap = shared / smaller_count if smaller_count else 0.0
            if jaccard >= self.near_duplicate_jaccard or (
                overlap >= self.overlap_threshold
                and smaller_count >= self.min_smaller_shingles
            ):
                edges.append((left_index, right_index, overlap))

        parent: dict[int, int] = {}

        def find(index: int) -> int:
            parent.setdefault(index, index)
            if parent[index] != index:
                parent[index] = find(parent[index])
            return parent[index]

        # Union matching pairs so chains such as A-B-C become one review group.
        for left_index, right_index, _ in edges:
            left_root, right_root = find(left_index), find(right_index)
            if left_root != right_root:
                parent[right_root] = left_root

        components: dict[int, set[int]] = {}
        for index in parent:
            components.setdefault(find(index), set()).add(index)

        summaries = []
        for indices in components.values():
            group = documents.iloc[sorted(indices)]
            overlaps = [
                overlap
                for left_index, right_index, overlap in edges
                if {left_index, right_index} <= indices
            ]
            summaries.append(self._group_row(group, overlaps))
        return pd.DataFrame(summaries, columns=self.GROUP_COLUMNS)

    def _shingles(self, text: str) -> set[str]:
        """Create overlapping token shingles for one document."""

        tokens = self.TOKEN_RE.findall(text.lower())
        if len(tokens) < self.shingle_size:
            return {" ".join(tokens)} if tokens else set()
        return {
            " ".join(tokens[index : index + self.shingle_size])
            for index in range(len(tokens) - self.shingle_size + 1)
        }

    def _group_row(self, group: pd.DataFrame, overlaps: list[float]) -> dict[str, object]:
        """Build the compact group shape used by the notebook."""

        group = group.sort_values(["class", "file"])
        classes = group["class"].tolist()
        return {
            "member_count": len(group),
            "classes": classes,
            "files": group["file"].tolist(),
            "previews": group["preview"].tolist(),
            "cross_class": len(set(classes)) > 1,
            "overlap_coefficients": [round(value, 4) for value in overlaps],
        }

    @staticmethod
    def _load_documents(dataset_root: str | Path) -> pd.DataFrame:
        """Read every class text file into one DataFrame."""

        paths = sorted(Path(dataset_root).glob("*/*.txt"))
        return pd.DataFrame(
            [
                {
                    "class": path.parent.name,
                    "file": path.name,
                    "text": path.read_text(encoding="utf-8", errors="replace"),
                }
                for path in paths
            ]
        )

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Normalize word-like tokens for exact comparison."""

        return " ".join(cls.TOKEN_RE.findall(text.lower()))


def exclusion_paths(
    documents: pd.DataFrame,
    *duplicate_groups: pd.DataFrame,
) -> list[str]:
    """Return deterministic class/file exclusions while retaining one group member."""

    token_counts = documents.set_index(["class", "file"])["token_count"].to_dict()
    excluded = set()
    for groups in duplicate_groups:
        for group in groups.itertuples(index=False):
            members = list(zip(group.classes, group.files))
            retained = min(
                members,
                key=lambda member: (-token_counts[member], member[0], member[1]),
            )
            excluded.update(
                f"{label}/{file_name}"
                for label, file_name in members
                if (label, file_name) != retained
            )
    return sorted(excluded)


def write_exclusions(config_path: str | Path, paths: list[str]) -> None:
    """Write reviewed duplicate candidates in DatasetLoader's YAML format."""

    content = (
        "# Generated duplicate-exclusion candidates. Review before training.\n"
        "# Paths are relative to data/dataset.\n"
        + yaml.safe_dump({"excluded_files": paths}, sort_keys=False)
    )
    Path(config_path).write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the standalone document audit."""

    parser = argparse.ArgumentParser(description="Audit invalid and duplicate dataset files.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--near-duplicate-jaccard",
        type=float,
        default=0.80,
        help="Notebook review threshold for near-duplicate Jaccard similarity.",
    )
    parser.add_argument(
        "--min-smaller-shingles",
        type=int,
        default=3,
        help="Notebook review minimum for the shorter document's shingles.",
    )
    parser.add_argument(
        "--exclusions-config",
        type=Path,
        default=Path("config/model/excluded_files.yaml"),
    )
    parser.add_argument(
        "--write-exclusions",
        action="store_true",
        help="Overwrite the exclusions YAML with current duplicate candidates.",
    )
    return parser


def main() -> None:
    """Run both audits and optionally regenerate the exclusion candidates."""

    args = build_parser().parse_args()
    _, invalid = InvalidDocumentAuditor().audit(args.dataset_dir)
    duplicate_auditor = DuplicateDocumentAuditor(
        near_duplicate_jaccard=args.near_duplicate_jaccard,
        min_smaller_shingles=args.min_smaller_shingles,
    )
    documents, exact, near = duplicate_auditor.audit(args.dataset_dir)
    print(f"Scanned documents: {len(documents)}")
    print(f"Invalid documents: {len(invalid)}")
    print(f"Exact duplicate groups: {len(exact)}")
    print(f"Near-duplicate groups: {len(near)}")
    if args.write_exclusions:
        paths = exclusion_paths(documents, exact, near)
        write_exclusions(args.exclusions_config, paths)
        print(f"Wrote exclusion candidates: {len(paths)}")


if __name__ == "__main__":
    main()

"""Tests for the production duplicate-document audit helper."""

from src.model.data_preparation.audit_document import (
    DuplicateDocumentAuditor,
    exclusion_paths,
)


def write_document(dataset_root, class_name: str, file_name: str, text: str) -> None:
    """Write one temporary class document for duplicate-auditor tests."""

    class_dir = dataset_root / class_name
    class_dir.mkdir(exist_ok=True)
    (class_dir / file_name).write_text(text, encoding="utf-8")


def test_audit_separates_exact_near_and_unrelated_documents(tmp_path):
    """Verify normalized exact matches do not repeat as near-duplicate pairs."""

    write_document(tmp_path, "space", "exact_a.txt", "Hello, SPACE world!")
    write_document(tmp_path, "medical", "exact_b.txt", "hello space world")
    base_tokens = [f"token{index}" for index in range(200)]
    edited_tokens = [*base_tokens]
    edited_tokens[100] = "replacement"
    write_document(tmp_path, "space", "near_a.txt", " ".join(base_tokens))
    write_document(tmp_path, "space", "near_b.txt", " ".join(edited_tokens))
    write_document(tmp_path, "medical", "unrelated.txt", "completely unrelated note")

    auditor = DuplicateDocumentAuditor()
    results, exact, near = auditor.audit(tmp_path)
    assert len(results) == 5
    assert list(exact.columns) == list(near.columns)
    assert len(exact) == 1
    assert exact.iloc[0]["files"] == ["exact_b.txt", "exact_a.txt"]
    assert exact.iloc[0]["member_count"] == 2
    assert exact.iloc[0]["overlap_coefficients"] == [1.0]
    assert exact.iloc[0]["cross_class"]
    assert near.iloc[0]["files"] == ["near_a.txt", "near_b.txt"]
    assert near.iloc[0]["overlap_coefficients"][0] > 0.95
    assert not near.iloc[0]["cross_class"]
    assert auditor.names_excluding_largest(results, exact) == ["exact_a.txt"]
    assert auditor.names_excluding_largest(results, near) == ["near_b.txt"]


def test_audit_detects_longer_document_containing_shorter_document(tmp_path):
    """Verify high containment is detected when symmetric Jaccard is lower."""

    smaller_tokens = [f"token{index}" for index in range(80)]
    longer_tokens = smaller_tokens + [f"extra{index}" for index in range(40)]
    write_document(tmp_path, "politics", "shorter.txt", " ".join(smaller_tokens))
    write_document(tmp_path, "politics", "longer.txt", " ".join(longer_tokens))

    auditor = DuplicateDocumentAuditor()
    results, _, near = auditor.audit(tmp_path)

    assert len(near) == 1
    assert near.iloc[0]["overlap_coefficients"] == [1.0]
    assert auditor.names_excluding_largest(results, near) == ["shorter.txt"]


def test_audit_ignores_short_contained_phrases(tmp_path):
    """Verify containment needs enough shingles to avoid generic short matches."""

    phrase = "alpha beta gamma delta epsilon zeta eta theta"
    write_document(tmp_path, "space", "short.txt", phrase)
    write_document(tmp_path, "medical", "long.txt", phrase + " extra words continue here")

    _, _, near = DuplicateDocumentAuditor().audit(tmp_path)

    assert near.empty


def test_connected_near_pairs_create_one_three_member_group(tmp_path):
    """Verify transitive near relationships collapse into one group row."""

    first = [f"a{index}" for index in range(100)]
    middle = first + [f"b{index}" for index in range(100)]
    last = [f"b{index}" for index in range(100)]
    write_document(tmp_path, "space", "a.txt", " ".join(first))
    write_document(tmp_path, "space", "middle.txt", " ".join(middle))
    write_document(tmp_path, "space", "b.txt", " ".join(last))

    auditor = DuplicateDocumentAuditor()
    results, _, near = auditor.audit(tmp_path)

    assert len(near) == 1
    assert near.iloc[0]["member_count"] == 3
    assert near.iloc[0]["overlap_coefficients"] == [1.0, 1.0]
    assert near.iloc[0]["files"] == ["a.txt", "b.txt", "middle.txt"]
    assert auditor.names_excluding_largest(results, near) == ["a.txt", "b.txt"]


def test_exclusion_paths_include_dataset_relative_labels(tmp_path):
    """Verify duplicate candidates match DatasetLoader's class/file contract."""
    tokens = [f"token{index}" for index in range(30)]
    write_document(tmp_path, "business", "duplicate.txt", " ".join(tokens))
    write_document(
        tmp_path,
        "space",
        "original.txt",
        " ".join(tokens + ["extra", "context"]),
    )
    documents, exact, near = DuplicateDocumentAuditor().audit(tmp_path)
    assert exclusion_paths(documents, exact, near) == ["business/duplicate.txt"]

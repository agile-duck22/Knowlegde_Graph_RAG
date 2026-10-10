"""Unit tests for src/ingest/loader.py.

All tests are fully offline — no file system corpus is loaded unless needed.
Tests use temporary JSONL files constructed in-memory via tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.ingest.loader import DocumentLoader, RawDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of dicts as a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _minimal_record(
    document_id: str = "doc-001",
    title: str = "Test Title",
    source_uri: str = "tests/doc.md",
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "title": title,
        "source_uri": source_uri,
    }


# ---------------------------------------------------------------------------
# RawDocument schema validation
# ---------------------------------------------------------------------------


class TestRawDocumentSchema:
    def test_valid_minimal_record(self) -> None:
        doc = RawDocument.model_validate(_minimal_record())
        assert doc.document_id == "doc-001"
        assert doc.doc_type == "technical_doc"  # default
        assert doc.content == ""  # default
        assert doc.metadata == {}  # default

    def test_full_record_parsed(self) -> None:
        data = {
            "document_id": "doc-full",
            "title": "Full Document",
            "source_uri": "path/to/doc.md",
            "doc_type": "incident_report",
            "content": "Service X failed at 03:00 UTC.",
            "metadata": {"severity": "P1", "team": "platform"},
        }
        doc = RawDocument.model_validate(data)
        assert doc.doc_type == "incident_report"
        assert doc.metadata["severity"] == "P1"

    def test_whitespace_only_document_id_raises(self) -> None:
        data = _minimal_record()
        data["document_id"] = "   "
        with pytest.raises(ValidationError):
            RawDocument.model_validate(data)

    def test_whitespace_only_title_raises(self) -> None:
        data = _minimal_record()
        data["title"] = "\t\n"
        with pytest.raises(ValidationError):
            RawDocument.model_validate(data)

    def test_missing_document_id_raises(self) -> None:
        data = {"title": "No ID", "source_uri": "x.md"}
        with pytest.raises(ValidationError):
            RawDocument.model_validate(data)

    def test_extra_fields_ignored(self) -> None:
        data = _minimal_record()
        data["unexpected_key"] = "should_be_ignored"
        doc = RawDocument.model_validate(data)
        assert not hasattr(doc, "unexpected_key")

    def test_content_hash_is_sha256(self) -> None:
        import hashlib

        doc = RawDocument.model_validate({**_minimal_record(), "content": "hello"})
        expected = hashlib.sha256(b"hello").hexdigest()
        assert doc.content_hash == expected

    def test_content_hash_empty_content(self) -> None:
        import hashlib

        doc = RawDocument.model_validate(_minimal_record())
        expected = hashlib.sha256(b"").hexdigest()
        assert doc.content_hash == expected

    def test_immutable_frozen_model(self) -> None:
        doc = RawDocument.model_validate(_minimal_record())
        with pytest.raises((AttributeError, TypeError, ValidationError)):
            doc.title = "changed"


# ---------------------------------------------------------------------------
# DocumentLoader — file not found
# ---------------------------------------------------------------------------


class TestDocumentLoaderFileNotFound:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        loader = DocumentLoader(tmp_path / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError, match="Corpus file not found"):
            loader.load()


# ---------------------------------------------------------------------------
# DocumentLoader — valid JSONL files
# ---------------------------------------------------------------------------


class TestDocumentLoaderValid:
    def test_single_record_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, [_minimal_record()])
        docs = DocumentLoader(path).load()
        assert len(docs) == 1
        assert docs[0].document_id == "doc-001"

    def test_multiple_records_loaded(self, tmp_path: Path) -> None:
        records = [_minimal_record(f"doc-{i:03d}") for i in range(10)]
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, records)
        docs = DocumentLoader(path).load()
        assert len(docs) == 10

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        with path.open("w") as f:
            f.write(json.dumps(_minimal_record("doc-A")) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps(_minimal_record("doc-B")) + "\n")
        docs = DocumentLoader(path).load()
        assert len(docs) == 2

    def test_iter_load_yields_same_records(self, tmp_path: Path) -> None:
        records = [_minimal_record(f"doc-{i}") for i in range(5)]
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, records)
        loader = DocumentLoader(path)
        streamed = list(loader.iter_load())
        batch = loader.load()
        assert [d.document_id for d in streamed] == [d.document_id for d in batch]

    def test_path_accepted_as_string(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, [_minimal_record()])
        docs = DocumentLoader(str(path)).load()
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# DocumentLoader — error conditions
# ---------------------------------------------------------------------------


class TestDocumentLoaderErrors:
    def test_malformed_json_raises_value_error_with_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        with path.open("w") as f:
            f.write(json.dumps(_minimal_record("doc-good")) + "\n")
            f.write("{bad json here\n")
        with pytest.raises(ValueError, match="line 2"):
            DocumentLoader(path).load()

    def test_non_object_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        with path.open("w") as f:
            f.write("[1, 2, 3]\n")
        with pytest.raises(ValueError, match="Expected JSON object"):
            DocumentLoader(path).load()

    def test_schema_validation_error_reports_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, [{"document_id": "x", "title": "   ", "source_uri": "y"}])
        with pytest.raises(ValueError, match="line 1"):
            DocumentLoader(path).load()

    def test_duplicate_document_id_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, [_minimal_record("dup-id"), _minimal_record("dup-id")])
        with pytest.raises(ValueError, match="Duplicate document_id"):
            DocumentLoader(path).load()

    def test_duplicate_id_error_reports_duplicate_id(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        _write_jsonl(path, [_minimal_record("shared"), _minimal_record("shared")])
        with pytest.raises(ValueError, match="shared"):
            DocumentLoader(path).load()


# ---------------------------------------------------------------------------
# DocumentLoader — real corpus sanity check (offline, deterministic)
# ---------------------------------------------------------------------------


class TestDocumentLoaderCorpus:
    """Load the project corpus and verify basic invariants without live services."""

    CORPUS_PATH = Path("data/raw/corpus.jsonl")

    @pytest.mark.skipif(
        not Path("data/raw/corpus.jsonl").exists(),
        reason="Corpus file not generated yet",
    )
    def test_corpus_loads_without_errors(self) -> None:
        docs = DocumentLoader(self.CORPUS_PATH).load()
        assert len(docs) >= 100, f"Expected >= 100 docs, got {len(docs)}"

    @pytest.mark.skipif(
        not Path("data/raw/corpus.jsonl").exists(),
        reason="Corpus file not generated yet",
    )
    def test_corpus_document_ids_are_unique(self) -> None:
        docs = DocumentLoader(self.CORPUS_PATH).load()
        ids = [d.document_id for d in docs]
        assert len(ids) == len(set(ids))

    @pytest.mark.skipif(
        not Path("data/raw/corpus.jsonl").exists(),
        reason="Corpus file not generated yet",
    )
    def test_corpus_all_titles_non_empty(self) -> None:
        docs = DocumentLoader(self.CORPUS_PATH).load()
        for doc in docs:
            assert doc.title.strip(), f"Empty title on doc {doc.document_id}"

"""Unit tests for scientific paper experiment offline validation tools and schemas (Phase A)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.scipaper.schemas.models import (
    ContentType,
    PaperManifestEntry,
    ScientificPaperDocument,
)
from experiments.scipaper.scripts.validate_experiment import (
    validate_benchmark,
    validate_corpus,
    validate_manifest,
)
from src.eval.schema import BenchmarkCategory, BenchmarkDifficulty, BenchmarkQuestion
from src.ingest.loader import RawDocument


class TestScientificPaperSchemas:
    """Tests for Phase A data contract models."""

    def test_paper_manifest_entry_valid(self) -> None:
        entry = PaperManifestEntry(
            paper_id="paper-arxiv-2401.00001",
            title="Efficient Retrieval via Graph Structures",
            authors=["Alice Smith", "Bob Jones"],
            year=2024,
            venue="ACL 2024",
            source_url="https://arxiv.org/abs/2401.00001",
            license="CC-BY-4.0",
            content_type=ContentType.FULL_TEXT,
            sha256="a" * 64,
            arxiv_id="2401.00001",
        )
        assert entry.paper_id == "paper-arxiv-2401.00001"
        assert entry.content_type == ContentType.FULL_TEXT

    def test_paper_manifest_invalid_sha256(self) -> None:
        with pytest.raises(ValidationError):
            PaperManifestEntry(
                paper_id="paper-arxiv-2401.00001",
                title="Invalid Checksum Paper",
                authors=["Alice Smith"],
                year=2024,
                venue="arXiv",
                source_url="https://arxiv.org/abs/2401.00001",
                license="CC-BY-4.0",
                content_type=ContentType.ABSTRACT,
                sha256="invalid-hash",
            )

    def test_scientific_paper_document_inherits_raw_document(self) -> None:
        doc = ScientificPaperDocument(
            document_id="paper-1",
            source_uri="https://example.com/paper.pdf",
            title="Scientific Paper Test",
            content="This is sample content for testing.",
            metadata={"authors": ["Author One"]},
        )
        assert isinstance(doc, RawDocument)
        assert doc.doc_type == "scientific_paper"
        assert len(doc.content_hash) == 64


class TestValidationTooling:
    """Tests for offline validation script."""

    def test_validate_manifest_success(self, tmp_path: Path) -> None:
        manifest_file = tmp_path / "manifest.jsonl"
        manifest_file.write_text(
            '{"paper_id": "paper-arxiv-2005.11401", "title": "Retrieval Augmented Generation", "authors": ["Patrick Lewis"], "year": 2020, "venue": "NeurIPS", "source_url": "https://arxiv.org/abs/2005.11401", "license": "CC-BY", "content_type": "abstract", "sha256": "'
            + "f" * 64
            + '"}\n',
            encoding="utf-8",
        )
        report, entries = validate_manifest(manifest_file)
        assert report.is_valid
        assert len(entries) == 1

    def test_validate_manifest_detects_duplicate_ids(self, tmp_path: Path) -> None:
        manifest_file = tmp_path / "manifest.jsonl"
        entry_str = (
            '{"paper_id": "paper-1", "title": "Test Paper", "authors": ["Author"], "year": 2024, "venue": "arXiv", "source_url": "https://arxiv.org/abs/1", "license": "CC-BY", "content_type": "abstract", "sha256": "'
            + "e" * 64
            + '"}\n'
        )
        manifest_file.write_text(entry_str + entry_str, encoding="utf-8")
        report, entries = validate_manifest(manifest_file)
        assert not report.is_valid
        assert any("Duplicate paper_id" in err for err in report.errors)

    def test_validate_benchmark_detects_category_imbalance(self, tmp_path: Path) -> None:
        benchmark_file = tmp_path / "benchmark.json"
        # Create dataset with wrong count and category distribution
        q = BenchmarkQuestion(
            id="EVAL-001",
            category=BenchmarkCategory.SINGLE_HOP_DIRECT_FACT,
            difficulty=BenchmarkDifficulty.EASY,
            question="What is RAG?",
            expected_facts=["RAG combines parametric and non-parametric memory"],
            reference_answer="RAG combines retrieval with generative models.",
        )
        data = {
            "version": "1.0.0",
            "domain": "scientific_papers_rag",
            "total_questions": 1,
            "questions": [q.model_dump()],
        }
        benchmark_file.write_text(pytest.importorskip("json").dumps(data), encoding="utf-8")

        report, dataset = validate_benchmark(benchmark_file, enforce_exact_count=50)
        assert not report.is_valid
        assert any("strictly requires exactly 50" in err for err in report.errors)

    def test_live_experiment_fixtures_pass_validation(self) -> None:
        """Verify that the authored experiment files in experiments/scipaper pass offline validation."""
        base_dir = Path("experiments/scipaper")
        manifest_path = base_dir / "data" / "manifest.jsonl"
        corpus_path = base_dir / "data" / "corpus_scipaper.jsonl"
        benchmark_path = base_dir / "eval" / "benchmark_scipaper_50.json"

        m_rep, m_entries = validate_manifest(manifest_path)
        assert m_rep.is_valid, f"Manifest errors: {m_rep.errors}"

        c_rep, c_docs = validate_corpus(corpus_path, manifest_entries=m_entries)
        assert c_rep.is_valid, f"Corpus errors: {c_rep.errors}"

        b_rep, b_data = validate_benchmark(
            benchmark_path, manifest_entries=m_entries, enforce_exact_count=50
        )
        assert b_rep.is_valid, f"Benchmark errors: {b_rep.errors}"
        assert b_data is not None
        assert len(b_data.questions) == 50

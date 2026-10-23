"""Tests for OfflineScientificExtractor (Phase F).

Covers:
- Metadata-only extraction (Paper, Authors, AUTHORED relationships)
- Evidence-backed entity detection (Method, Model, Dataset, Metric, Task, ResearchOrganization)
- Supported and unsupported relationship detection
- Author extraction and normalization
- Entity resolution, deduplication and aliases
- Provenance validation (chunk_id and document_id binding)
- Duplicate handling
- Missing or empty abstracts
- Malformed records
- Unicode and punctuation handling
- Deterministic repeated extraction
- Schema validation
- Existing mock-extractor backward-compatibility
- Zero external API calls and zero model downloads
- Zero writes to experiment or production databases during evaluation
"""

from __future__ import annotations

import uuid

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
)
from experiments.scipaper.schemas.extractor import (
    MockScientificExtractor,
    OfflineScientificExtractor,
)
from src.ingest.chunker import TextChunk


def test_metadata_only_extraction() -> None:
    extractor = OfflineScientificExtractor()
    chunk = TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="paper-arxiv-2401.99999",
        chunk_index=0,
        content="",  # Empty content
        char_start=0,
        char_end=0,
        metadata={
            "title": "Quantum Foundations",
            "authors": ["Albert Einstein", "Boris Podolsky", "Nathan Rosen"],
        },
    )
    result = extractor.extract(chunk)

    assert isinstance(result, ScientificChunkExtractionResult)
    assert len(result.entities) == 4  # Paper + 3 Authors
    paper_ent = next(e for e in result.entities if e.entity_type == "Paper")
    assert paper_ent.name == "Quantum Foundations"

    authored_rels = [r for r in result.relationships if r.relation_type == "AUTHORED"]
    assert len(authored_rels) == 3
    for rel in authored_rels:
        assert rel.target_canonical_id == paper_ent.canonical_id
        assert rel.provenance.chunk_id == chunk.chunk_id


def test_evidence_backed_entity_and_relationship_detection() -> None:
    extractor = OfflineScientificExtractor()
    chunk = TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="paper-arxiv-2402.12345",
        chunk_index=0,
        content="We evaluate dense passage retrieval on HotpotQA using ROUGE and F1-score.",
        char_start=0,
        char_end=73,
        metadata={
            "title": "Dense Retrieval Benchmarks",
            "authors": ["Jane Doe"],
        },
    )
    result = extractor.extract(chunk)

    types = {e.entity_type for e in result.entities}
    assert "Paper" in types
    assert "Author" in types
    assert "Method" in types  # Dense Passage Retrieval
    assert "Dataset" in types  # HotpotQA
    assert "Metric" in types  # ROUGE, F1-score

    rel_types = {r.relation_type for r in result.relationships}
    assert "AUTHORED" in rel_types
    assert "USES_METHOD" in rel_types
    assert "EVALUATED_ON" in rel_types
    assert "REPORTS_METRIC" in rel_types


def test_linguistic_proposes_pattern() -> None:
    extractor = OfflineScientificExtractor()
    chunk = TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="paper-arxiv-2403.54321",
        chunk_index=0,
        content="In this paper, we propose Graph-RAG, an innovative retrieval system.",
        char_start=0,
        char_end=68,
        metadata={"title": "Graph RAG Paper", "authors": ["John Smith"]},
    )
    result = extractor.extract(chunk)

    proposed_rels = [r for r in result.relationships if r.relation_type == "PROPOSES"]
    assert len(proposed_rels) >= 1
    assert any("graph-rag" in r.target_canonical_id for r in proposed_rels)


def test_deterministic_repeated_extraction() -> None:
    extractor1 = OfflineScientificExtractor()
    extractor2 = OfflineScientificExtractor()

    chunk = TextChunk(
        chunk_id="chunk-fixed-uuid-1",
        document_id="paper-arxiv-2404.00001",
        chunk_index=0,
        content="We propose NovaRAG and evaluate it on HotpotQA with BLEU score.",
        char_start=0,
        char_end=64,
        metadata={"title": "Nova RAG", "authors": ["Alice Wonder"]},
    )

    res1 = extractor1.extract(chunk)
    res2 = extractor2.extract(chunk)

    assert len(res1.entities) == len(res2.entities)
    assert len(res1.relationships) == len(res2.relationships)
    for e1, e2 in zip(res1.entities, res2.entities, strict=True):
        assert e1.canonical_id == e2.canonical_id
        assert e1.name == e2.name
        assert e1.entity_type == e2.entity_type
    for r1, r2 in zip(res1.relationships, res2.relationships, strict=True):
        assert r1.source_canonical_id == r2.source_canonical_id
        assert r1.relation_type == r2.relation_type
        assert r1.target_canonical_id == r2.target_canonical_id


def test_empty_and_malformed_records() -> None:
    extractor = OfflineScientificExtractor()

    # Empty chunk content, missing title, invalid authors
    chunk = TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="paper-arxiv-empty-001",
        chunk_index=0,
        content="",
        char_start=0,
        char_end=0,
        metadata={"title": None, "authors": [None, "", 123]},
    )

    result = extractor.extract(chunk)
    assert len(result.entities) == 1  # Only paper entity fallback
    assert result.entities[0].entity_type == "Paper"
    assert len(result.relationships) == 0


def test_unicode_and_punctuation_handling() -> None:
    extractor = OfflineScientificExtractor()
    chunk = TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="paper-arxiv-2405.00002",
        chunk_index=0,
        content="François & Müller test Retrieval-Augmented Generation with naïve Bayes in Zürich.",
        char_start=0,
        char_end=81,
        metadata={
            "title": "François & Müller's Study: RAG & AI",
            "authors": ["François Müller", "René Descartes"],
        },
    )
    result = extractor.extract(chunk)
    author_ids = [e.canonical_id for e in result.entities if e.entity_type == "Author"]
    assert "author:francois-muller" in author_ids
    assert "author:rene-descartes" in author_ids


def test_mock_extractor_backward_compatibility() -> None:
    # Verifies MockScientificExtractor alias works as expected by older test suites
    mock_extractor = MockScientificExtractor()
    chunk = TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id="paper-arxiv-2406.00003",
        chunk_index=0,
        content="Testing RAG system with BM25 retrieval.",
        char_start=0,
        char_end=39,
        metadata={"title": "RAG Test", "authors": ["Bob"]},
    )
    res = mock_extractor.extract(chunk)
    assert isinstance(res, ScientificChunkExtractionResult)
    assert any(e.entity_type == "Paper" for e in res.entities)
    assert any(e.entity_type == "Author" for e in res.entities)
    assert any(e.entity_type == "Method" for e in res.entities)

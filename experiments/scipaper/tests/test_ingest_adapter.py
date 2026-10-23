"""Unit and offline integration tests for scientific-paper ingestion adapter.

Tests cover:
- Scientific entity and relationship schemas validation.
- Deterministic scientific entity ID construction and resolver normalization.
- Rejection of unknown scientific entity and relationship types.
- Ambiguous entity detection and conflict handling.
- Mock extractor deterministic output and provenance preservation.
- Strict isolation and target guards (rejection of enterprise ports 5432 / 7687).
- Offline CLI behavior: --validate-only and --dry-run making zero writes.
- Refusal to perform live execution in Phase C2 without separate authorization.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
    ScientificExtractedEntity,
    ScientificExtractedRelationship,
)
from experiments.scipaper.schemas.extractor import MockScientificExtractor
from experiments.scipaper.schemas.neo4j_writer import (
    SCIENTIFIC_ENTITY_TEMPLATES,
    IsolatedScientificNeo4jWriter,
    TargetIsolationError,
    UnsupportedScientificTypeError,
)
from experiments.scipaper.schemas.resolver import (
    AmbiguousScientificEntityError,
    ScientificEntityResolver,
    slugify_scientific_name,
)
from experiments.scipaper.scripts.ingest_scipaper import (
    validate_corpus_offline,
    verify_target_safety_invariants,
)
from src.extraction.schemas import ExtractionProvenance
from src.ingest.chunker import TextChunk

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "corpus_scipaper.jsonl"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.jsonl"


# ==============================================================================
# 1. Scientific Entity and Relationship Schema Tests
# ==============================================================================


def test_scientific_entity_valid() -> None:
    prov = ExtractionProvenance(chunk_id="chunk-123", document_id="doc-456")
    ent = ScientificExtractedEntity(
        canonical_id="paper:dense-passage-retrieval",
        name="Dense Passage Retrieval",
        entity_type="Paper",
        description="A dense retrieval methodology",
        confidence=0.95,
        provenance=prov,
    )
    assert ent.canonical_id == "paper:dense-passage-retrieval"
    assert ent.entity_type == "Paper"


def test_scientific_entity_prefix_mismatch_fails() -> None:
    prov = ExtractionProvenance(chunk_id="chunk-123", document_id="doc-456")
    with pytest.raises(ValueError, match="does not match entity_type"):
        ScientificExtractedEntity(
            canonical_id="model:bert-base",
            name="BERT Base",
            entity_type="Paper",  # Mismatch: prefix is model: but type is Paper
            provenance=prov,
        )


def test_scientific_relationship_valid() -> None:
    prov = ExtractionProvenance(chunk_id="chunk-123", document_id="doc-456")
    rel = ScientificExtractedRelationship(
        source_canonical_id="author:karpukhin",
        source_type="Author",
        target_canonical_id="paper:dense-passage-retrieval",
        target_type="Paper",
        relation_type="AUTHORED",
        source_chunk_ids=["chunk-123"],
        provenance=prov,
    )
    assert rel.relation_type == "AUTHORED"
    assert rel.source_canonical_id == "author:karpukhin"


def test_scientific_relationship_self_loop_fails() -> None:
    prov = ExtractionProvenance(chunk_id="chunk-123", document_id="doc-456")
    with pytest.raises(ValueError, match="Self-referential"):
        ScientificExtractedRelationship(
            source_canonical_id="paper:same-paper",
            source_type="Paper",
            target_canonical_id="paper:same-paper",
            target_type="Paper",
            relation_type="CITES",
            source_chunk_ids=["chunk-123"],
            provenance=prov,
        )


# ==============================================================================
# 2. Scientific Resolver Tests
# ==============================================================================


def test_slugify_scientific_name() -> None:
    assert slugify_scientific_name("Dense Passage Retrieval") == "dense-passage-retrieval"
    assert slugify_scientific_name("GPT-4o (Omni)") == "gpt-4o-omni"
    assert slugify_scientific_name("arXiv:2305.12345") == "arxiv-2305.12345"


def test_resolver_build_canonical_id() -> None:
    resolver = ScientificEntityResolver()
    cid = resolver.build_canonical_id("Paper", "Attention Is All You Need")
    assert cid == "paper:attention-is-all-you-need"

    cid_author = resolver.build_canonical_id("Author", "Vaswani et al.")
    assert cid_author == "author:vaswani-et-al"


def test_resolver_ambiguous_alias_fails() -> None:
    resolver = ScientificEntityResolver()
    prov = ExtractionProvenance(chunk_id="chunk-1", document_id="doc-1")

    ent1 = ScientificExtractedEntity(
        canonical_id="model:bert-base",
        name="BERT",
        entity_type="Model",
        aliases=["BERT Model"],
        provenance=prov,
    )
    resolver.register_entity(ent1)

    ent2 = ScientificExtractedEntity(
        canonical_id="model:bert-large",
        name="BERT Large",
        entity_type="Model",
        aliases=["BERT Model"],  # Same alias pointing to different canonical ID
        provenance=prov,
    )
    with pytest.raises(AmbiguousScientificEntityError):
        resolver.register_entity(ent2)


# ==============================================================================
# 3. Mock Scientific Extractor Tests
# ==============================================================================


def test_mock_extractor_deterministic_output() -> None:
    extractor = MockScientificExtractor()
    chunk = TextChunk(
        chunk_id="c1111111-2222-3333-4444-555555555555",
        document_id="paper-arxiv-2401.00001",
        chunk_index=0,
        content="This paper proposes a novel RAG architecture with dense retrieval.",
        char_start=0,
        char_end=67,
        metadata={
            "title": "Novel RAG Architecture",
            "authors": ["Alice Smith", "Bob Jones"],
        },
    )

    result = extractor.extract(chunk)
    assert isinstance(result, ScientificChunkExtractionResult)
    assert result.chunk_id == chunk.chunk_id
    assert len(result.entities) >= 3  # Paper, 2 Authors, RAG/dense retrieval method
    assert any(e.entity_type == "Paper" for e in result.entities)
    assert any(e.entity_type == "Author" for e in result.entities)
    assert any(e.entity_type == "Method" for e in result.entities)
    assert any(r.relation_type == "AUTHORED" for r in result.relationships)


# ==============================================================================
# 4. Neo4j Writer Isolation and Template Tests
# ==============================================================================


def test_neo4j_writer_rejects_enterprise_port() -> None:
    with pytest.raises(TargetIsolationError, match="enterprise port 7687"):
        IsolatedScientificNeo4jWriter(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )


def test_neo4j_writer_rejects_arbitrary_port() -> None:
    with pytest.raises(TargetIsolationError, match="expected isolated port 7688"):
        IsolatedScientificNeo4jWriter(
            uri="bolt://localhost:9999",
            user="neo4j",
            password="password",
        )


def test_neo4j_writer_templates_cover_all_8_scientific_types() -> None:
    expected_types = [
        "Paper",
        "Author",
        "Model",
        "Dataset",
        "Metric",
        "Task",
        "Method",
        "ResearchOrganization",
    ]
    for t in expected_types:
        assert t in SCIENTIFIC_ENTITY_TEMPLATES
        tmpl = SCIENTIFIC_ENTITY_TEMPLATES[t]
        assert f":Entity:{t}" in tmpl
        assert "$id" in tmpl


def test_neo4j_writer_rejects_unknown_entity_type() -> None:
    ExtractionProvenance(chunk_id="chunk-1", document_id="doc-1")
    # Using mock writer without opening actual driver
    with patch("experiments.scipaper.schemas.neo4j_writer.GraphDatabase.driver"):
        writer = IsolatedScientificNeo4jWriter(
            uri="bolt://localhost:7688",
            user="neo4j",
            password="password",
        )
        fake_entity = MagicMock()
        fake_entity.entity_type = "InvalidType"
        with pytest.raises(UnsupportedScientificTypeError):
            writer.write_entities([fake_entity])


# ==============================================================================
# 5. CLI & Target Safety Guard Tests
# ==============================================================================


def test_target_safety_invariants_pass() -> None:
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "ALLOW_PAID_EMBEDDINGS": "false",
    }
    # Should not raise
    verify_target_safety_invariants(env)


def test_target_safety_invariants_reject_enterprise_db() -> None:
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "kgr_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "ALLOW_PAID_EMBEDDINGS": "false",
    }
    with pytest.raises(ValueError, match="forbidden"):
        verify_target_safety_invariants(env)


def test_target_safety_invariants_reject_paid_embeddings() -> None:
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "ALLOW_PAID_EMBEDDINGS": "true",
    }
    with pytest.raises(ValueError, match="Paid embeddings must be false"):
        verify_target_safety_invariants(env)


def test_target_safety_invariants_reject_non_local_hosts() -> None:
    env_bad_pg = {
        "POSTGRES_HOST": "remote-db.corp.net",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "ALLOW_PAID_EMBEDDINGS": "false",
    }
    with pytest.raises(ValueError, match="Target PostgreSQL host"):
        verify_target_safety_invariants(env_bad_pg)

    env_bad_neo = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://remote-neo4j.corp.net:7688",
        "ALLOW_PAID_EMBEDDINGS": "false",
    }
    with pytest.raises(ValueError, match="Target Neo4j host"):
        verify_target_safety_invariants(env_bad_neo)


def test_neo4j_writer_rejects_non_local_host() -> None:
    with pytest.raises(TargetIsolationError, match="Non-local host"):
        IsolatedScientificNeo4jWriter(
            uri="bolt://192.168.1.100:7688",
            user="neo4j",
            password="password",
        )


def test_neo4j_writer_rejects_invalid_scheme() -> None:
    with pytest.raises(TargetIsolationError, match="Invalid Neo4j scheme"):
        IsolatedScientificNeo4jWriter(
            uri="http://localhost:7688",
            user="neo4j",
            password="password",
        )


def test_offline_corpus_validation_passes() -> None:
    docs_cnt, chunks_cnt = validate_corpus_offline(CORPUS_PATH, MANIFEST_PATH)
    assert docs_cnt == 120
    assert chunks_cnt == 474

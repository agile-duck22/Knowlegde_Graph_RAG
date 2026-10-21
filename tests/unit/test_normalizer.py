"""Unit tests for Context Normalizer and Token Budget Manager (TASK-401).

Verifies:
- Normalization of graph query results and vector chunks.
- Deduplication and priority promotion for chunks appearing in both sources.
- Preservation of canonical chunk_id, canonical entity_ids, and relationship provenance.
- Token budget enforcement, truncation, and deterministic prioritization.
- Handling of empty inputs, oversized items, malformed records, and missing metadata.
- Candidate chunk ID aggregation for downstream citation verification.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.retrieval.graph_retriever import GraphQueryResult
from src.retrieval.vector_retriever import RetrievedChunk
from src.synthesis.normalizer import (
    ContextNormalizer,
    NormalizationInput,
    default_estimate_tokens,
)


@pytest.fixture
def default_settings() -> Settings:
    return Settings(
        max_context_tokens=1000,
    )


@pytest.fixture
def normalizer(default_settings: Settings) -> ContextNormalizer:
    return ContextNormalizer(settings=default_settings)


# ---------------------------------------------------------------------------
# Token Estimator Tests
# ---------------------------------------------------------------------------


def test_default_estimate_tokens() -> None:
    assert default_estimate_tokens("") == 0
    assert default_estimate_tokens("   ") == 0
    assert default_estimate_tokens("hello") >= 1
    long_text = "The quick brown fox jumps over the lazy dog. " * 10
    tokens = default_estimate_tokens(long_text)
    assert tokens > 50


# ---------------------------------------------------------------------------
# Graph Triple Normalization Tests
# ---------------------------------------------------------------------------


def test_normalize_graph_results_exact_lookup(normalizer: ContextNormalizer) -> None:
    graph_res = GraphQueryResult(
        template_id="T-01",
        template_name="Exact Entity Lookup",
        parameters={"entity_id": "service:payment-gateway"},
        records=[
            {
                "entity_id": "service:payment-gateway",
                "name": "Payment Gateway",
                "type": "Service",
                "description": "Handles payments",
                "source_chunk_ids": ["c1111111-1111-1111-1111-111111111111"],
            }
        ],
        record_count=1,
        source_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        entity_ids=["service:payment-gateway"],
    )

    inp = NormalizationInput(graph_results=[graph_res])
    res = normalizer.normalize(inp)

    assert not res.is_empty
    assert "service:payment-gateway" in res.all_entity_ids
    assert "c1111111-1111-1111-1111-111111111111" in res.candidate_chunk_ids


def test_normalize_graph_results_neighborhood_and_triples(normalizer: ContextNormalizer) -> None:
    graph_res = GraphQueryResult(
        template_id="T-02",
        template_name="Direct Neighborhood",
        parameters={"entity_id": "service:order-api"},
        records=[
            {
                "source": "service:order-api",
                "relation": "DEPENDS_ON",
                "target": "database:order-db",
                "source_chunk_ids": ["c2222222-2222-2222-2222-222222222222"],
            },
            {
                "source": {"id": "service:order-api", "name": "Order API"},
                "relation": "OWNED_BY",
                "target": {"id": "team:order-team", "name": "Order Management Team"},
                "source_chunk_ids": ["c3333333-3333-3333-3333-333333333333"],
            },
        ],
        record_count=2,
        source_chunk_ids=[
            "c2222222-2222-2222-2222-222222222222",
            "c3333333-3333-3333-3333-333333333333",
        ],
        entity_ids=["service:order-api", "database:order-db", "team:order-team"],
    )

    inp = NormalizationInput(graph_results=[graph_res])
    res = normalizer.normalize(inp)

    assert len(res.graph_triples) == 2
    assert res.graph_triples[0].source_id == "service:order-api"
    assert res.graph_triples[0].relationship_type == "DEPENDS_ON"
    assert res.graph_triples[0].target_id == "database:order-db"
    assert res.graph_triples[1].target_name == "Order Management Team"
    assert "### Knowledge Graph Context" in res.formatted_context
    assert "| `service:order-api` | `DEPENDS_ON` | `database:order-db` |" in res.formatted_context


def test_normalize_graph_results_t04_repositories(normalizer: ContextNormalizer) -> None:
    """Verify that repositories returned in T-04 are normalized into STORES_CODE triples."""
    graph_res = GraphQueryResult(
        template_id="T-04",
        template_name="Vulnerability Blast Radius",
        parameters={"vuln_id": "vuln:cve-2024-3094"},
        records=[
            {
                "vuln_id": "vuln:cve-2024-3094",
                "vuln_name": "CVE-2024-3094",
                "vulnerable_service_id": "service:auth-service",
                "vulnerable_service_name": "auth-service",
                "vuln_chunk_ids": ["c4444444-4444-4444-4444-444444444444"],
                "affected_dependents": [],
                "repositories": [
                    {
                        "id": "repo:auth-service-repo",
                        "name": "auth-service-repo",
                        "chunk_ids": ["c5555555-5555-5555-5555-555555555555"],
                    }
                ],
            }
        ],
        record_count=1,
        source_chunk_ids=[
            "c4444444-4444-4444-4444-444444444444",
            "c5555555-5555-5555-5555-555555555555",
        ],
        entity_ids=["vuln:cve-2024-3094", "service:auth-service", "repo:auth-service-repo"],
    )

    inp = NormalizationInput(graph_results=[graph_res])
    res = normalizer.normalize(inp)

    assert len(res.graph_triples) == 2
    # Affected by triple
    assert res.graph_triples[0].source_id == "service:auth-service"
    assert res.graph_triples[0].relationship_type == "AFFECTED_BY"
    assert res.graph_triples[0].target_id == "vuln:cve-2024-3094"
    # Repo stores code triple
    assert res.graph_triples[1].source_id == "repo:auth-service-repo"
    assert res.graph_triples[1].relationship_type == "STORES_CODE"
    assert res.graph_triples[1].target_id == "service:auth-service"
    assert "c5555555-5555-5555-5555-555555555555" in res.candidate_chunk_ids


def test_normalize_graph_results_t05_datastores_and_infra(normalizer: ContextNormalizer) -> None:
    """Verify that datastores and infrastructures in T-05 are normalized into graph triples."""
    graph_res = GraphQueryResult(
        template_id="T-05",
        template_name="Service Dependency Chain",
        parameters={"service_id": "service:cart-service"},
        records=[
            {
                "root_service_id": "service:cart-service",
                "downstream_service_id": "service:inventory-service",
                "downstream_service_name": "Inventory Service",
                "rel_type": "DEPENDS_ON",
                "source_chunk_ids": ["c6666666-6666-6666-6666-666666666666"],
                "datastores": [
                    {
                        "id": "database:inventory-pg",
                        "name": "inventory-pg",
                        "rel": "READS_FROM",
                        "chunk_ids": ["c7777777-7777-7777-7777-777777777777"],
                    }
                ],
                "infrastructures": [
                    {
                        "id": "infra:k8s-prod-cluster-us-east",
                        "name": "K8s Prod Cluster US East",
                        "rel": "DEPLOYED_ON",
                        "chunk_ids": ["c8888888-8888-8888-8888-888888888888"],
                    }
                ],
            }
        ],
        record_count=1,
        source_chunk_ids=["c6666666-6666-6666-6666-666666666666"],
        entity_ids=["service:cart-service", "service:inventory-service", "database:inventory-pg"],
    )

    inp = NormalizationInput(graph_results=[graph_res])
    res = normalizer.normalize(inp)

    assert len(res.graph_triples) == 3
    assert res.graph_triples[0].source_id == "service:cart-service"
    assert res.graph_triples[0].target_id == "service:inventory-service"
    assert res.graph_triples[1].source_id == "service:inventory-service"
    assert res.graph_triples[1].relationship_type == "READS_FROM"
    assert res.graph_triples[1].target_id == "database:inventory-pg"
    assert res.graph_triples[2].relationship_type == "DEPLOYED_ON"
    assert res.graph_triples[2].target_id == "infra:k8s-prod-cluster-us-east"


def test_normalize_graph_results_t09_incoming_dependents(normalizer: ContextNormalizer) -> None:
    """Verify that T-09 incoming dependent records are normalized into DEPENDS_ON and WRITES_TO triples."""
    graph_res = GraphQueryResult(
        template_id="T-09",
        template_name="Incoming Service Dependents",
        parameters={"service_id": "service:user-directory-service"},
        records=[
            {
                "service_id": "service:user-directory-service",
                "caller_service_id": "service:auth-service",
                "caller_service_name": "auth-service",
                "caller_chunk_ids": [["c9999999-9999-9999-9999-999999999999"]],
                "datastores": [
                    {
                        "id": "database:auth-session-redis",
                        "name": "auth-session-redis",
                        "rel": "WRITES_TO",
                        "chunk_ids": ["caaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
                    }
                ],
            }
        ],
        record_count=1,
        source_chunk_ids=["c9999999-9999-9999-9999-999999999999"],
        entity_ids=[
            "service:user-directory-service",
            "service:auth-service",
            "database:auth-session-redis",
        ],
    )

    inp = NormalizationInput(graph_results=[graph_res])
    res = normalizer.normalize(inp)

    assert len(res.graph_triples) == 2
    assert res.graph_triples[0].source_id == "service:auth-service"
    assert res.graph_triples[0].relationship_type == "DEPENDS_ON"
    assert res.graph_triples[0].target_id == "service:user-directory-service"
    assert res.graph_triples[1].source_id == "service:auth-service"
    assert res.graph_triples[1].relationship_type == "WRITES_TO"
    assert res.graph_triples[1].target_id == "database:auth-session-redis"


# ---------------------------------------------------------------------------
# Vector Chunks & Deduplication Tests
# ---------------------------------------------------------------------------


def test_normalize_vector_chunks(normalizer: ContextNormalizer) -> None:
    chunk1 = RetrievedChunk(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="doc-100",
        content="Payment Gateway handles credit cards and refunds.",
        similarity_score=0.88,
        token_count=10,
        char_start=0,
        char_end=50,
        metadata={"doc_type": "architecture"},
    )
    chunk2 = RetrievedChunk(
        chunk_id="c2222222-2222-2222-2222-222222222222",
        document_id="doc-200",
        content="Order Service connects to PostgreSQL on port 5432.",
        similarity_score=0.75,
        token_count=10,
        char_start=0,
        char_end=51,
        metadata={"doc_type": "guide"},
    )

    inp = NormalizationInput(vector_chunks=[chunk1, chunk2])
    res = normalizer.normalize(inp)

    assert len(res.chunks) == 2
    assert res.chunks[0].chunk_id == "c1111111-1111-1111-1111-111111111111"
    assert res.chunks[0].provenance_source == "vector"
    assert '<chunk id="c1111111-1111-1111-1111-111111111111"' in res.formatted_context
    assert "Payment Gateway handles credit cards" in res.formatted_context


def test_deduplication_and_hybrid_promotion(normalizer: ContextNormalizer) -> None:
    shared_chunk_id = "c-shared-001"

    # Graph retrieval has this chunk ID in provenance
    graph_res = GraphQueryResult(
        template_id="T-03",
        template_name="Multi-Hop",
        parameters={"entity_id": "service:checkout"},
        records=[
            {
                "source": "service:checkout",
                "relation": "CALLS",
                "target": "service:payment",
                "source_chunk_ids": [shared_chunk_id],
            }
        ],
        record_count=1,
        source_chunk_ids=[shared_chunk_id],
        entity_ids=["service:checkout", "service:payment"],
    )

    # Vector search also returned this chunk
    vec_chunk = RetrievedChunk(
        chunk_id=shared_chunk_id,
        document_id="doc-shared",
        content="Checkout service communicates directly with Payment API.",
        similarity_score=0.91,
        token_count=12,
        char_start=0,
        char_end=56,
    )

    # Another vector chunk that is vector-only
    vec_only_chunk = RetrievedChunk(
        chunk_id="c-vector-only",
        document_id="doc-vec",
        content="General documentation on checkout timeouts.",
        similarity_score=0.95,
        token_count=10,
        char_start=0,
        char_end=43,
    )

    inp = NormalizationInput(
        graph_results=[graph_res],
        vector_chunks=[vec_chunk, vec_only_chunk],
    )
    res = normalizer.normalize(inp)

    assert len(res.chunks) == 2
    # The shared chunk should be promoted to hybrid provenance
    shared_item = next(c for c in res.chunks if c.chunk_id == shared_chunk_id)
    assert shared_item.provenance_source == "hybrid"

    # In prioritization, hybrid tier comes before vector tier even if vector has slightly higher score
    assert res.chunks[0].chunk_id == shared_chunk_id


def test_hydrated_graph_chunks_merging(normalizer: ContextNormalizer) -> None:
    shared_id = "c-hydrated-01"

    vec_chunk = RetrievedChunk(
        chunk_id=shared_id,
        document_id="doc-1",
        content="Primary text content.",
        similarity_score=0.85,
        token_count=5,
        char_start=0,
        char_end=20,
    )

    hydrated_chunk = RetrievedChunk(
        chunk_id=shared_id,
        document_id="doc-1",
        content="Primary text content.",
        similarity_score=1.0,
        token_count=5,
        char_start=0,
        char_end=20,
        metadata={"hydrated": True},
    )

    graph_only_hydrated = RetrievedChunk(
        chunk_id="c-graph-only",
        document_id="doc-2",
        content="Graph only background information.",
        similarity_score=1.0,
        token_count=6,
        char_start=0,
        char_end=35,
    )

    inp = NormalizationInput(
        vector_chunks=[vec_chunk],
        hydrated_graph_chunks=[hydrated_chunk, graph_only_hydrated],
    )
    res = normalizer.normalize(inp)

    assert len(res.chunks) == 2
    chunk_ids = [c.chunk_id for c in res.chunks]
    assert shared_id in chunk_ids
    assert "c-graph-only" in chunk_ids

    # Merged chunk retains hybrid source and metadata
    merged_item = next(c for c in res.chunks if c.chunk_id == shared_id)
    assert merged_item.provenance_source == "hybrid"
    assert merged_item.metadata.get("hydrated") is True


# ---------------------------------------------------------------------------
# Token Budget Enforcement & Truncation Tests
# ---------------------------------------------------------------------------


def test_token_budget_truncates_lower_priority_chunks() -> None:
    # Set tight budget
    def _simple_word_estimator(text: str) -> int:
        return len(text.split())

    norm = ContextNormalizer(
        token_estimator=_simple_word_estimator,
    )

    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content="This is a distinct text sentence for chunk testing. " * 5,  # ~50 words
            similarity_score=1.0 - (i * 0.1),
            token_count=50,
            char_start=0,
            char_end=200,
        )
        for i in range(10)
    ]

    inp = NormalizationInput(
        vector_chunks=chunks,
        max_context_tokens=120,  # Room for header + ~2 chunks
    )
    res = norm.normalize(inp)

    assert res.is_truncated is True
    assert res.truncated_chunks_count > 0
    assert len(res.chunks) < 10
    # Highest similarity chunks preserved first
    assert res.chunks[0].chunk_id == "chunk-0"
    assert res.chunks[1].chunk_id == "chunk-1"


def test_token_budget_preserves_graph_triples_over_large_chunks() -> None:
    def _word_estimator(text: str) -> int:
        return len(text.split())

    norm = ContextNormalizer(
        token_estimator=_word_estimator,
    )

    graph_res = GraphQueryResult(
        template_id="T-02",
        template_name="Neighborhood",
        parameters={"entity_id": "service:auth"},
        records=[
            {
                "source": "service:auth",
                "relation": "USES",
                "target": "database:auth-db",
                "source_chunk_ids": ["c-auth-01"],
            }
        ],
        record_count=1,
        source_chunk_ids=["c-auth-01"],
        entity_ids=["service:auth", "database:auth-db"],
    )

    massive_chunk = RetrievedChunk(
        chunk_id="chunk-massive",
        document_id="doc-huge",
        content="Word " * 200,
        similarity_score=0.8,
        token_count=200,
        char_start=0,
        char_end=1000,
    )

    inp = NormalizationInput(
        graph_results=[graph_res],
        vector_chunks=[massive_chunk],
        max_context_tokens=50,  # Enough for graph table, but not massive chunk
    )
    res = norm.normalize(inp)

    assert len(res.graph_triples) == 1
    assert len(res.chunks) == 0
    assert res.is_truncated is True
    assert "### Knowledge Graph Context" in res.formatted_context


# ---------------------------------------------------------------------------
# Edge Cases & Resiliency Tests
# ---------------------------------------------------------------------------


def test_empty_input_handling(normalizer: ContextNormalizer) -> None:
    inp = NormalizationInput()
    res = normalizer.normalize(inp)

    assert res.is_empty
    assert res.formatted_context == ""
    assert res.chunks == []
    assert res.graph_triples == []
    assert res.candidate_chunk_ids == []
    assert res.is_truncated is False


def test_malformed_graph_record_resilience(normalizer: ContextNormalizer) -> None:
    graph_res = GraphQueryResult(
        template_id="T-01",
        template_name="Entity Lookup",
        parameters={},
        records=[
            {},  # Empty dictionary
            {"source": None, "target": None},
            {"other_key": 12345},
            {"source": "service:valid-a", "relation": "LINKS", "target": "service:valid-b"},
        ],
        record_count=4,
        source_chunk_ids=[],
        entity_ids=[],
    )

    inp = NormalizationInput(graph_results=[graph_res])
    res = normalizer.normalize(inp)

    # Malformed records skipped gracefully, valid record extracted
    assert len(res.graph_triples) == 1
    assert res.graph_triples[0].source_id == "service:valid-a"
    assert res.graph_triples[0].target_id == "service:valid-b"


def test_oversized_item_rejection(normalizer: ContextNormalizer) -> None:
    oversized = RetrievedChunk(
        chunk_id="c-giant",
        document_id="doc-giant",
        content="Huge content block " * 500,
        similarity_score=0.9,
        token_count=1500,
        char_start=0,
        char_end=9000,
    )

    inp = NormalizationInput(
        vector_chunks=[oversized],
        max_context_tokens=50,  # Far smaller than oversized chunk
    )
    res = normalizer.normalize(inp)

    assert len(res.chunks) == 0
    assert res.is_truncated is True
    assert res.truncated_chunks_count == 1

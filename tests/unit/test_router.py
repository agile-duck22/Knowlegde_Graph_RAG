"""Unit tests for Router Engine (TASK-305).

Verifies:
- Delegation for all 4 execution modes (GRAPH_ONLY, VECTOR_ONLY, HYBRID, OUT_OF_SCOPE)
- Zero database calls for OUT_OF_SCOPE queries
- Entity linking integration and template selection for graph queries
- Handling of queries with empty/unresolved entity links (fallback to HYBRID)
- Forwarding of parameters, limits, and filters to vector/graph retrievers
- Pre-computed query vector passing
- Graceful handling of retrieval failures and error encapsulation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
from src.embeddings.client import EmbeddingGenerator
from src.retrieval.cypher_templates import (
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    TEMPLATE_MULTI_HOP_DEPENDENCY,
    TEMPLATE_TEAM_OWNERSHIP_SERVICES,
)
from src.retrieval.graph_retriever import GraphQueryResult, SafeGraphRetriever
from src.retrieval.vector_retriever import RetrievedChunk, VectorRetriever
from src.routing.entity_linker import (
    EntityLink,
    EntityLinker,
    EntityLinkingResult,
    LinkResolutionStatus,
)
from src.routing.intent_classifier import (
    ClassificationMethod,
    IntentClassificationResult,
    QueryIntent,
    QueryIntentClassifier,
)
from src.routing.router import RouterEngine, RoutingRequest, RoutingResult


@pytest.fixture
def mock_classifier() -> MagicMock:
    return MagicMock(spec=QueryIntentClassifier)


@pytest.fixture
def mock_linker() -> MagicMock:
    return MagicMock(spec=EntityLinker)


@pytest.fixture
def mock_graph_retriever() -> MagicMock:
    return MagicMock(spec=SafeGraphRetriever)


@pytest.fixture
def mock_vector_retriever() -> AsyncMock:
    return AsyncMock(spec=VectorRetriever)


@pytest.fixture
def mock_embedding_generator() -> MagicMock:
    gen = MagicMock(spec=EmbeddingGenerator)
    gen.generate_single.return_value = [0.1] * 1536
    return gen


@pytest.fixture
def router(
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
    mock_vector_retriever: AsyncMock,
    mock_embedding_generator: MagicMock,
) -> RouterEngine:
    settings = Settings(
        vector_top_k=5,
        hybrid_vector_top_k=3,
        vector_similarity_threshold=0.60,
        graph_node_limit=25,
    )
    return RouterEngine(
        classifier=mock_classifier,
        entity_linker=mock_linker,
        graph_retriever=mock_graph_retriever,
        vector_retriever=mock_vector_retriever,
        embedding_generator=mock_embedding_generator,
        settings=settings,
    )


# ===========================================================================
# 1. OUT_OF_SCOPE Query Routing
# ===========================================================================


@pytest.mark.asyncio
async def test_route_out_of_scope_query(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
    mock_vector_retriever: AsyncMock,
) -> None:
    mock_classifier.classify.return_value = IntentClassificationResult(
        query="What is the weather in Tokyo?",
        mode=QueryIntent.OUT_OF_SCOPE,
        confidence=0.99,
        reasoning="Non-enterprise weather question.",
        method=ClassificationMethod.HEURISTIC,
    )

    result = await router.route_and_retrieve("What is the weather in Tokyo?")

    assert isinstance(result, RoutingResult)
    assert result.mode == QueryIntent.OUT_OF_SCOPE
    assert result.is_empty is True
    assert len(result.graph_results) == 0
    assert len(result.vector_chunks) == 0

    # Ensure zero database retrievals were performed
    mock_linker.link_query.assert_not_called()
    mock_graph_retriever.execute_template.assert_not_called()
    mock_vector_retriever.search.assert_not_called()


# ===========================================================================
# 2. GRAPH_ONLY Query Routing
# ===========================================================================


@pytest.mark.asyncio
async def test_route_graph_only_query_success(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
    mock_vector_retriever: AsyncMock,
) -> None:
    query = "Which services are owned by team:checkout-engineering?"

    mock_classifier.classify.return_value = IntentClassificationResult(
        query=query,
        mode=QueryIntent.GRAPH_ONLY,
        confidence=0.95,
        reasoning="Team ownership query.",
        suggested_template_id=TEMPLATE_TEAM_OWNERSHIP_SERVICES,
        extracted_entities=["team:checkout-engineering"],
        method=ClassificationMethod.LLM,
    )

    mock_linker.link_query.return_value = EntityLinkingResult(
        query=query,
        links=[
            EntityLink(
                mention="team:checkout-engineering",
                canonical_id="team:checkout-engineering",
                entity_type="Team",
                status=LinkResolutionStatus.RESOLVED,
            )
        ],
    )

    mock_graph_retriever.execute_template.return_value = GraphQueryResult(
        template_id=TEMPLATE_TEAM_OWNERSHIP_SERVICES,
        template_name="Team Ownership & Service Impact",
        parameters={"team_id": "team:checkout-engineering", "limit": 25},
        records=[{"team_name": "Checkout Team", "service_id": "service:checkout-service"}],
        record_count=1,
        source_chunk_ids=["chunk-prov-001"],
        entity_ids=["team:checkout-engineering", "service:checkout-service"],
    )

    result = await router.route_and_retrieve(query)

    assert result.mode == QueryIntent.GRAPH_ONLY
    assert result.has_graph_data is True
    assert result.has_vector_data is False
    assert len(result.graph_results) == 1
    assert result.all_provenance_chunk_ids == ["chunk-prov-001"]

    # Graph retriever executed with correct template and team_id parameter
    mock_graph_retriever.execute_template.assert_called_once()
    call_args = mock_graph_retriever.execute_template.call_args[1]
    assert call_args["template_id"] == TEMPLATE_TEAM_OWNERSHIP_SERVICES
    assert call_args["parameters"]["team_id"] == "team:checkout-engineering"

    # Vector retriever was NOT called
    mock_vector_retriever.search.assert_not_called()


@pytest.mark.asyncio
async def test_route_graph_only_without_entities_falls_back_to_hybrid(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
    mock_vector_retriever: AsyncMock,
) -> None:
    query = "Show dependencies of that component"

    mock_classifier.classify.return_value = IntentClassificationResult(
        query=query,
        mode=QueryIntent.GRAPH_ONLY,
        confidence=0.85,
        reasoning="Dependency question but ambiguous entity.",
        suggested_template_id=TEMPLATE_MULTI_HOP_DEPENDENCY,
        extracted_entities=[],
    )

    # Linker cannot resolve any entities
    mock_linker.link_query.return_value = EntityLinkingResult(
        query=query,
        links=[],
    )

    mock_vector_retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="chunk-vec-100",
            document_id="doc-100",
            content="Component architecture text...",
            similarity_score=0.88,
        )
    ]

    result = await router.route_and_retrieve(query)

    # Automatically demoted from GRAPH_ONLY to HYBRID to guarantee search coverage
    assert result.mode == QueryIntent.HYBRID
    assert result.has_vector_data is True
    assert len(result.vector_chunks) == 1
    assert result.all_provenance_chunk_ids == ["chunk-vec-100"]

    # Vector search was executed
    mock_vector_retriever.search.assert_called_once()


# ===========================================================================
# 3. VECTOR_ONLY Query Routing
# ===========================================================================


@pytest.mark.asyncio
async def test_route_vector_only_query(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
    mock_vector_retriever: AsyncMock,
    mock_embedding_generator: MagicMock,
) -> None:
    query = "What is the retry policy and SLA for payments?"

    mock_classifier.classify.return_value = IntentClassificationResult(
        query=query,
        mode=QueryIntent.VECTOR_ONLY,
        confidence=0.92,
        reasoning="Asks for policy prose.",
        suggested_template_id=None,
    )

    mock_linker.link_query.return_value = EntityLinkingResult(query=query, links=[])

    mock_vector_retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="chunk-policy-01",
            document_id="doc-spec-01",
            content="Retry policy is exponential backoff...",
            similarity_score=0.91,
        )
    ]

    result = await router.route_and_retrieve(query)

    assert result.mode == QueryIntent.VECTOR_ONLY
    assert result.has_vector_data is True
    assert result.has_graph_data is False
    assert len(result.vector_chunks) == 1
    assert result.all_provenance_chunk_ids == ["chunk-policy-01"]

    # Embedding generated and vector search called with top_k=5 (vector mode default)
    mock_embedding_generator.generate_single.assert_called_once_with(query)
    mock_vector_retriever.search.assert_called_once()
    search_kwargs = mock_vector_retriever.search.call_args[1]
    assert search_kwargs["top_k"] == 5

    # Zero graph queries executed
    mock_graph_retriever.execute_template.assert_not_called()


# ===========================================================================
# 4. HYBRID Query Routing
# ===========================================================================


@pytest.mark.asyncio
async def test_route_hybrid_query_both_stores(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
    mock_vector_retriever: AsyncMock,
) -> None:
    query = "Describe the payment architecture and which team owns database:customer-orders-pg"

    mock_classifier.classify.return_value = IntentClassificationResult(
        query=query,
        mode=QueryIntent.HYBRID,
        confidence=0.88,
        reasoning="Broad architectural question touching both graph topology and text spec.",
        suggested_template_id=TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
        extracted_entities=["database:customer-orders-pg"],
    )

    mock_linker.link_query.return_value = EntityLinkingResult(
        query=query,
        links=[
            EntityLink(
                mention="database:customer-orders-pg",
                canonical_id="database:customer-orders-pg",
                entity_type="Database",
                status=LinkResolutionStatus.RESOLVED,
            )
        ],
    )

    mock_graph_retriever.execute_template.return_value = GraphQueryResult(
        template_id=TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
        template_name="Database Producers & Consumers",
        parameters={"database_id": "database:customer-orders-pg", "limit": 25},
        records=[{"database_id": "database:customer-orders-pg", "readers": []}],
        record_count=1,
        source_chunk_ids=["chunk-graph-01"],
    )

    mock_vector_retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="chunk-vec-02",
            document_id="doc-arch-02",
            content="Payment architecture overview...",
            similarity_score=0.85,
        )
    ]

    req = RoutingRequest(
        query=query,
        top_k=3,
        document_id="doc-arch-02",
    )
    result = await router.route_and_retrieve(req)

    assert result.mode == QueryIntent.HYBRID
    assert result.has_graph_data is True
    assert result.has_vector_data is True
    assert len(result.graph_results) == 1
    assert len(result.vector_chunks) == 1

    # Provenance contains chunk IDs from both graph and vector
    assert "chunk-graph-01" in result.all_provenance_chunk_ids
    assert "chunk-vec-02" in result.all_provenance_chunk_ids

    # Vector search called with custom top_k=3 and document_id filter
    mock_vector_retriever.search.assert_called_once()
    search_kwargs = mock_vector_retriever.search.call_args[1]
    assert search_kwargs["top_k"] == 3
    assert search_kwargs["document_id"] == "doc-arch-02"


# ===========================================================================
# 5. Pre-computed Embedding Vector & Error Handling
# ===========================================================================


@pytest.mark.asyncio
async def test_route_with_precomputed_query_vector(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_vector_retriever: AsyncMock,
    mock_embedding_generator: MagicMock,
) -> None:
    mock_classifier.classify.return_value = IntentClassificationResult(
        query="Explain authentication",
        mode=QueryIntent.VECTOR_ONLY,
        confidence=0.9,
        reasoning="Prose lookup.",
    )
    mock_linker.link_query.return_value = EntityLinkingResult(
        query="Explain authentication", links=[]
    )
    mock_vector_retriever.search.return_value = []

    custom_vec = [0.5] * 1536
    await router.route_and_retrieve("Explain authentication", query_vector=custom_vec)

    # Embedding generator was bypassed because query_vector was provided directly
    mock_embedding_generator.generate_single.assert_not_called()
    mock_vector_retriever.search.assert_called_once()
    assert mock_vector_retriever.search.call_args[1]["query_vector"] == custom_vec


@pytest.mark.asyncio
async def test_route_graph_retrieval_failure_encapsulation(
    router: RouterEngine,
    mock_classifier: MagicMock,
    mock_linker: MagicMock,
    mock_graph_retriever: MagicMock,
) -> None:
    mock_classifier.classify.return_value = IntentClassificationResult(
        query="Show blast radius for vuln:cve-2024-21626",
        mode=QueryIntent.GRAPH_ONLY,
        confidence=0.95,
        reasoning="Vulnerability blast radius.",
    )
    mock_linker.link_query.return_value = EntityLinkingResult(
        query="Show blast radius for vuln:cve-2024-21626",
        links=[
            EntityLink(
                mention="vuln:cve-2024-21626",
                canonical_id="vuln:cve-2024-21626",
                entity_type="Vulnerability",
                status=LinkResolutionStatus.RESOLVED,
            )
        ],
    )
    mock_graph_retriever.execute_template.side_effect = RuntimeError("Neo4j node timeout")

    result = await router.route_and_retrieve("Show blast radius for vuln:cve-2024-21626")

    assert result.mode == QueryIntent.GRAPH_ONLY
    assert result.has_graph_data is False
    assert "Graph retrieval failed" in (result.error_message or "")

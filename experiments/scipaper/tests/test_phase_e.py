"""Unit and integration tests for Phase E Scientific Graph, Hybrid RAG, and Abstention."""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from experiments.scipaper.eval.abstention import evaluate_local_abstention
from experiments.scipaper.retrieval.cypher_templates import (
    TEMPLATE_FIND_AUTHORS_OF_PAPER,
    ScientificCypherTemplateCatalog,
)
from experiments.scipaper.retrieval.graph_retriever import (
    ScientificGraphQueryResult,
    ScientificGraphRetriever,
    ScientificInvalidTemplateError,
)
from experiments.scipaper.retrieval.hybrid_retriever import (
    ScientificHybridRetriever,
)
from experiments.scipaper.schemas.neo4j_writer import TargetIsolationError
from src.retrieval.vector_retriever import RetrievedChunk


def test_scientific_cypher_templates_catalog() -> None:
    """Verify all approved scientific templates are registered and parameterized."""
    templates = ScientificCypherTemplateCatalog.list_templates()
    assert len(templates) == 7

    spec = ScientificCypherTemplateCatalog.get(TEMPLATE_FIND_AUTHORS_OF_PAPER)
    assert "MATCH (a:Entity:Author)-[r:AUTHORED]->(p:Entity:Paper)" in spec.cypher
    assert "LIMIT $limit" in spec.cypher
    assert "paper_id" in spec.required_params

    # Check unknown template raises KeyError
    with pytest.raises(KeyError):
        ScientificCypherTemplateCatalog.get("UNKNOWN_TEMPLATE")


def test_scientific_graph_retriever_target_guards() -> None:
    """Verify ScientificGraphRetriever rejects non-local hosts and enterprise ports."""
    # Enterprise port 7687
    with pytest.raises(TargetIsolationError):
        ScientificGraphRetriever("bolt://localhost:7687", "user", "pass")

    # Enterprise port 5432
    with pytest.raises(TargetIsolationError):
        ScientificGraphRetriever("bolt://localhost:5432", "user", "pass")

    # Non-local host
    with pytest.raises(TargetIsolationError):
        ScientificGraphRetriever("bolt://remote-neo4j.org:7688", "user", "pass")

    # Invalid scheme
    with pytest.raises(TargetIsolationError):
        ScientificGraphRetriever("http://localhost:7688", "user", "pass")


def test_scientific_graph_retriever_mocked(mocker: MockerFixture) -> None:
    """Verify ScientificGraphRetriever parses records and extracts provenance."""
    mock_driver = mocker.MagicMock()
    mock_session = mocker.MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mocker.patch("neo4j.GraphDatabase.driver", return_value=mock_driver)

    retriever = ScientificGraphRetriever("bolt://localhost:7688", "neo4j", "testpass")

    mock_record = mocker.MagicMock()
    mock_record.data.return_value = {
        "paper_id": "paper:arxiv-2411.18583",
        "author_id": "author:nurshat-fateh-ali",
        "source_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2"],
    }
    mock_session.run.return_value = [mock_record]

    res = retriever.execute_template(
        TEMPLATE_FIND_AUTHORS_OF_PAPER,
        {"paper_id": "paper:arxiv-2411.18583", "paper_name": "paper-arxiv-2411.18583"},
    )

    assert not res.is_empty
    assert res.record_count == 1
    assert "chunk-uuid-1" in res.provenance_chunk_ids
    assert "chunk-uuid-2" in res.provenance_chunk_ids
    assert "paper:arxiv-2411.18583" in res.entity_ids

    # Unapproved template raises
    with pytest.raises(ScientificInvalidTemplateError):
        retriever.execute_template("UNAPPROVED")


def test_entity_detection_in_hybrid_retriever(mocker: MockerFixture) -> None:
    """Verify paper and method entities are extracted from query text."""
    mock_vec = mocker.MagicMock()
    mock_graph = mocker.MagicMock()

    hybrid = ScientificHybridRetriever(mock_vec, mock_graph)

    detected = hybrid.detect_entities(
        "What does paper-arxiv-2411.18583 say about retrieval-augmented generation and knowledge graph?"
    )
    assert "paper:arxiv-2411.18583" in detected["papers"]
    assert "method:retrieval-augmented-generation" in detected["methods"]
    assert "method:knowledge-graph" in detected["methods"]


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion_logic(mocker: MockerFixture) -> None:
    """Verify RRF correctly boosts chunks appearing in both vector and graph lists."""
    mock_vec = mocker.MagicMock()
    mock_graph = mocker.MagicMock()

    # Vector returns chunk A (rank 1) and chunk B (rank 2)
    chunk_a = RetrievedChunk(
        chunk_id="chunk-A",
        document_id="doc-1",
        content="Content A",
        similarity_score=0.75,
        token_count=10,
        char_start=0,
        char_end=50,
        metadata={},
    )
    chunk_b = RetrievedChunk(
        chunk_id="chunk-B",
        document_id="doc-1",
        content="Content B",
        similarity_score=0.60,
        token_count=10,
        char_start=50,
        char_end=100,
        metadata={},
    )
    mock_vec.search = mocker.AsyncMock(return_value=[chunk_a, chunk_b])

    # Graph returns chunk B (rank 1) and chunk C (rank 2)
    chunk_c = RetrievedChunk(
        chunk_id="chunk-C",
        document_id="doc-2",
        content="Content C",
        similarity_score=0.0,
        token_count=10,
        char_start=0,
        char_end=50,
        metadata={},
    )
    mock_graph_res = ScientificGraphQueryResult(
        template_id=TEMPLATE_FIND_AUTHORS_OF_PAPER,
        template_name="Find Authors",
        parameters={},
        records=[{"dummy": 1}],
        record_count=1,
        provenance_chunk_ids=["chunk-B", "chunk-C"],
    )
    mock_graph.execute_template.return_value = mock_graph_res
    mock_vec.get_chunks_by_ids = mocker.AsyncMock(return_value=[chunk_b, chunk_c])

    hybrid = ScientificHybridRetriever(mock_vec, mock_graph, rrf_k=60)
    out = await hybrid.retrieve(
        query="paper-arxiv-2411.18583 authors",
        query_embedding=[0.1] * 384,
        top_k=5,
    )

    # Chunk B should be boosted to Rank 1 because it has rank 2 in vector + rank 1 in graph:
    # Score(B) = 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
    # Score(A) = 1/(60+1) = 0.016393
    # Score(C) = 1/(60+2) = 0.016129
    assert len(out.hybrid_results) == 3
    assert out.hybrid_results[0].chunk_id == "chunk-B"
    assert out.hybrid_results[0].final_rank == 1
    assert out.hybrid_results[0].vector_rank == 2
    assert out.hybrid_results[0].graph_rank == 1
    assert out.hybrid_results[1].chunk_id == "chunk-A"
    assert out.hybrid_results[2].chunk_id == "chunk-C"


def test_local_abstention_evaluation() -> None:
    """Verify local abstention calculation on synthetic evaluations."""
    evals = [
        # Answerable queries with top similarities
        {"category": "single_hop_direct_fact", "top_similarity": 0.65},
        {"category": "2_hop_relationship", "top_similarity": 0.52},
        {"category": "3_hop_dependency", "top_similarity": 0.42},
        {"category": "relationship_aggregation", "top_similarity": 0.38},
        # Out-of-scope queries
        {"category": "out_of_scope", "top_similarity": 0.30},
        {"category": "out_of_scope", "top_similarity": 0.44},
    ]

    results = evaluate_local_abstention(evals, thresholds=[0.40, 0.50])
    assert len(results) == 2

    # At threshold 0.40:
    # Answerable: [0.65, 0.52, 0.42] pass; 0.38 refused (1 false refusal)
    # Out of scope: 0.30 refused (<0.40); 0.44 passed (1 correct refusal, 1 missed refusal)
    r40 = results[0]
    assert r40.threshold == 0.40
    assert r40.answerable_refusals == 1
    assert r40.answerable_pass == 3
    assert r40.out_of_scope_refusals == 1
    assert r40.out_of_scope_pass == 1
    assert r40.refusal_recall == 0.50
    assert r40.answerable_availability == 0.75

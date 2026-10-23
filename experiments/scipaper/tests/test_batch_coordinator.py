"""Offline failure-injection, recovery, and idempotency tests for the Batch Coordinator.

Tests:
1. Pure mock offline execution of batch coordinator across 5 documents.
2. PostgreSQL failure injection: ensures state is marked FAILED, affected_store='postgresql', zero writes to Neo4j.
3. Neo4j failure injection: ensures state is marked NEEDS_RECONCILIATION, affected_store='neo4j'.
4. Re-reconciliation / retry of partially written document: ensures successful idempotency and transition to COMPLETE.
5. Dimension validation failure: non-384 or NaN vectors are rejected before any database write.
6. Batch size limit enforcement: batches > 20 documents are rejected fail-closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from experiments.scipaper.schemas.batch_coordinator import (
    DocumentState,
    ScientificBatchCoordinator,
)
from src.ingest.loader import RawDocument


class FakeEmbeddingModel:
    def __init__(self, dim: int = 384, return_nan: bool = False) -> None:
        self.dim = dim
        self.return_nan = return_nan

    def encode(self, texts: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray:
        if self.return_nan:
            arr = np.zeros((len(texts), self.dim), dtype=np.float32)
            arr[0, 0] = float("nan")
            return arr
        # Deterministic float vector
        return np.ones((len(texts), self.dim), dtype=np.float32) * 0.1


def create_sample_doc(doc_id: str) -> RawDocument:
    return RawDocument(
        document_id=doc_id,
        source_uri=f"https://arxiv.org/abs/{doc_id}",
        title=f"Sample Paper {doc_id}",
        doc_type="scientific_paper",
        content="We evaluate retrieval-augmented generation on knowledge graphs and vector search.",
        metadata={"authors": ["Alice Smith", "Bob Jones"]},
    )


def setup_mock_pg_client(execute_side_effect: Any = None) -> tuple[MagicMock, AsyncMock]:
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=None)
    mock_tx.__aexit__ = AsyncMock(return_value=None)

    mock_conn = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_tx)
    mock_conn.executemany = AsyncMock()
    if execute_side_effect:
        mock_conn.fetchval.side_effect = execute_side_effect
        mock_conn.execute.side_effect = execute_side_effect
    else:
        mock_conn.fetchval.return_value = "doc-ok"
        mock_conn.execute.return_value = "INSERT 0 1"

    mock_pg_client = MagicMock()
    mock_pg_client.connection.return_value.__aenter__.return_value = mock_conn
    mock_pg_client.connection.return_value.__aexit__.return_value = None
    return mock_pg_client, mock_conn


@pytest.mark.asyncio
async def test_coordinator_success_flow() -> None:
    mock_pg_client, _ = setup_mock_pg_client()

    mock_neo_writer = MagicMock()
    mock_neo_writer.write_chunk_result.return_value = (5, 4)

    model = FakeEmbeddingModel(dim=384)
    coordinator = ScientificBatchCoordinator(
        pg_client=mock_pg_client,
        neo_writer=mock_neo_writer,
        embedding_model=model,
    )

    docs = [create_sample_doc(f"doc-{i}") for i in range(3)]
    report = await coordinator.ingest_batch(docs)

    assert report.total_documents == 3
    assert report.completed == 3
    assert report.failed == 0
    assert report.needs_reconciliation == 0
    for doc in docs:
        d_rep = report.document_reports[doc.document_id]
        assert d_rep.state == DocumentState.COMPLETE
        assert d_rep.chunks_count >= 1
        assert d_rep.pg_chunks_written >= 1
        assert d_rep.neo_entities_written == 5
        assert d_rep.neo_relationships_written == 4


@pytest.mark.asyncio
async def test_coordinator_pg_failure_injection() -> None:
    """If PostgreSQL write fails, document is marked FAILED and Neo4j is never called."""
    mock_pg_client, _ = setup_mock_pg_client(
        execute_side_effect=RuntimeError("PostgreSQL disk full")
    )

    mock_neo_writer = MagicMock()
    model = FakeEmbeddingModel(dim=384)

    coordinator = ScientificBatchCoordinator(
        pg_client=mock_pg_client,
        neo_writer=mock_neo_writer,
        embedding_model=model,
    )

    doc = create_sample_doc("doc-pg-fail")
    report = await coordinator.ingest_batch([doc])

    assert report.total_documents == 1
    assert report.completed == 0
    assert report.failed == 1
    d_rep = report.document_reports["doc-pg-fail"]
    assert d_rep.state == DocumentState.FAILED
    assert d_rep.affected_store == "postgresql"
    assert "PostgreSQL disk full" in (d_rep.error_message or "")
    # Neo4j must never have been called
    mock_neo_writer.write_chunk_result.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_neo4j_failure_injection_and_reconciliation() -> None:
    """If Neo4j write fails after PG succeeded, state is NEEDS_RECONCILIATION and retry works."""
    mock_pg_client, _ = setup_mock_pg_client()

    mock_neo_writer = MagicMock()
    # First attempt raises error
    mock_neo_writer.write_chunk_result.side_effect = RuntimeError("Neo4j connection reset")
    model = FakeEmbeddingModel(dim=384)

    coordinator = ScientificBatchCoordinator(
        pg_client=mock_pg_client,
        neo_writer=mock_neo_writer,
        embedding_model=model,
    )

    doc = create_sample_doc("doc-neo-fail")
    report = await coordinator.ingest_batch([doc])

    assert report.total_documents == 1
    assert report.completed == 0
    assert report.needs_reconciliation == 1
    d_rep = report.document_reports["doc-neo-fail"]
    assert d_rep.state == DocumentState.NEEDS_RECONCILIATION
    assert d_rep.affected_store == "neo4j"
    assert d_rep.pg_chunks_written >= 1  # PG wrote successfully

    # Now simulate recovery / retry with working Neo4j
    mock_neo_writer.write_chunk_result.side_effect = None
    mock_neo_writer.write_chunk_result.return_value = (3, 2)

    reconcile_rep = await coordinator.reconcile_document(doc)
    assert reconcile_rep.state == DocumentState.COMPLETE
    assert reconcile_rep.neo_entities_written == 3
    assert reconcile_rep.neo_relationships_written == 2


@pytest.mark.asyncio
async def test_coordinator_invalid_embedding_dimension_rejected() -> None:
    """Embedding model producing 768d instead of 384d is rejected before any DB write."""
    mock_pg_client = MagicMock()
    mock_neo_writer = MagicMock()
    bad_model = FakeEmbeddingModel(dim=768)

    coordinator = ScientificBatchCoordinator(
        pg_client=mock_pg_client,
        neo_writer=mock_neo_writer,
        embedding_model=bad_model,
    )

    doc = create_sample_doc("doc-dim-fail")
    report = await coordinator.ingest_batch([doc])

    assert report.failed == 1
    d_rep = report.document_reports["doc-dim-fail"]
    assert d_rep.state == DocumentState.FAILED
    assert d_rep.failed_stage == "pre_write_processing"
    assert "expected 384" in (d_rep.error_message or "")
    # Zero DB calls
    mock_pg_client.connection.assert_not_called()
    mock_neo_writer.write_chunk_result.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_nan_embedding_rejected() -> None:
    """Embedding vector with NaN is rejected before any DB write."""
    mock_pg_client = MagicMock()
    mock_neo_writer = MagicMock()
    nan_model = FakeEmbeddingModel(dim=384, return_nan=True)

    coordinator = ScientificBatchCoordinator(
        pg_client=mock_pg_client,
        neo_writer=mock_neo_writer,
        embedding_model=nan_model,
    )

    doc = create_sample_doc("doc-nan-fail")
    report = await coordinator.ingest_batch([doc])

    assert report.failed == 1
    d_rep = report.document_reports["doc-nan-fail"]
    assert d_rep.state == DocumentState.FAILED
    assert d_rep.failed_stage == "pre_write_processing"
    assert "non-finite value" in (d_rep.error_message or "")
    mock_pg_client.connection.assert_not_called()
    mock_neo_writer.write_chunk_result.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_batch_limit_enforced() -> None:
    """Batch size > 20 is rejected fail-closed."""
    coordinator = ScientificBatchCoordinator(
        pg_client=MagicMock(),
        neo_writer=MagicMock(),
        embedding_model=FakeEmbeddingModel(dim=384),
        batch_doc_limit=20,
    )

    docs = [create_sample_doc(f"doc-{i}") for i in range(21)]
    with pytest.raises(ValueError, match="exceeds limit 20"):
        await coordinator.ingest_batch(docs)

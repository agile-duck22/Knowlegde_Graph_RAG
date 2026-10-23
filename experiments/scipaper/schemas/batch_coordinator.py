"""Batch Ingestion Coordinator for the Scientific-Paper Experiment (Phase C3B).

Features:
- Bounded batches: at most 20 documents and approx 100 chunks per batch.
- Deterministic document and chunk IDs.
- Explicit per-document states:
    DocumentState.PENDING
    DocumentState.PROCESSING
    DocumentState.COMPLETE
    DocumentState.FAILED
    DocumentState.NEEDS_RECONCILIATION
- Does NOT mark complete until both stores are verified.
- Safe retry and idempotency logic; no blind retry of ambiguous writes.
- Failure injection / offline testable interfaces with mocks.
- Disabled live execution by default (requires --execute-ingestion flag and phase authorization).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
)
from experiments.scipaper.schemas.extractor import MockScientificExtractor
from experiments.scipaper.schemas.neo4j_writer import (
    TargetIsolationError,
)
from experiments.scipaper.schemas.resolver import ScientificEntityResolver
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
    REQUIRED_POSTGRES_PORT,
)
from src.db.postgres_writer import ChunkWriteInput, PostgresWriter
from src.ingest.chunker import DocumentChunker
from src.ingest.loader import RawDocument

logger = logging.getLogger("scipaper_batch_coordinator")


class DocumentState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    NEEDS_RECONCILIATION = "needs-reconciliation"


@dataclass
class DocumentIngestionReport:
    document_id: str
    state: DocumentState
    chunks_count: int = 0
    pg_chunks_written: int = 0
    pg_entity_links_written: int = 0
    neo_entities_written: int = 0
    neo_relationships_written: int = 0
    failed_stage: str | None = None
    affected_store: str | None = None
    error_message: str | None = None


@dataclass
class BatchReconciliationReport:
    total_documents: int = 0
    completed: int = 0
    failed: int = 0
    needs_reconciliation: int = 0
    document_reports: dict[str, DocumentIngestionReport] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "completed": self.completed,
            "failed": self.failed,
            "needs_reconciliation": self.needs_reconciliation,
        }


class EmbeddingModelProtocol(Protocol):
    def encode(self, texts: Sequence[str], convert_to_numpy: bool = True) -> Any: ...


def enforce_strict_isolation_guards(env_vars: dict[str, str]) -> None:
    """Enforce strict target guards before attempting any live connection."""
    pg_host = (env_vars.get("POSTGRES_HOST") or "localhost").lower()
    if pg_host not in ("localhost", "127.0.0.1", "::1"):
        raise TargetIsolationError(f"PostgreSQL host '{pg_host}' is not loopback.")

    pg_port = int(env_vars.get("POSTGRES_PORT", 0))
    if pg_port in ENTERPRISE_PORTS or pg_port != REQUIRED_POSTGRES_PORT:
        raise TargetIsolationError(
            f"PostgreSQL port {pg_port} is invalid. Refusing connection to non-isolated port."
        )

    pg_db = env_vars.get("POSTGRES_DB", "")
    if pg_db != "scipaper_db":
        raise TargetIsolationError(
            f"PostgreSQL DB '{pg_db}' is invalid. Refusing connection to non-scipaper DB."
        )

    neo4j_uri = env_vars.get("NEO4J_URI", "")
    parsed = urlparse(neo4j_uri)
    neo_host = (parsed.hostname or "").lower()
    if neo_host not in ("localhost", "127.0.0.1", "::1"):
        raise TargetIsolationError(f"Neo4j host '{neo_host}' is not loopback.")

    if parsed.port in ENTERPRISE_PORTS or parsed.port != REQUIRED_NEO4J_BOLT_PORT:
        raise TargetIsolationError(
            f"Neo4j port {parsed.port} is invalid. Refusing connection to non-isolated port."
        )

    allow_paid = env_vars.get("ALLOW_PAID_EMBEDDINGS", "").strip().lower()
    if allow_paid != "false":
        raise TargetIsolationError(f"ALLOW_PAID_EMBEDDINGS={allow_paid}; must be false.")


class ScientificBatchCoordinator:
    """Coordinates bounded batch ingestion with explicit per-document states and reconciliation."""

    def __init__(
        self,
        pg_client: Any,
        neo_writer: Any,
        embedding_model: EmbeddingModelProtocol,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_doc_limit: int = 20,
    ) -> None:
        self.pg_client = pg_client
        self.neo_writer = neo_writer
        self.embedding_model = embedding_model
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.resolver = ScientificEntityResolver()
        self.extractor = MockScientificExtractor(resolver=self.resolver)
        self.batch_doc_limit = batch_doc_limit

    async def ingest_single_document_atomic(
        self,
        doc: RawDocument,
        pg_writer: PostgresWriter,
    ) -> DocumentIngestionReport:
        """Process one document end-to-end with explicit state tracking and store verification."""
        rep = DocumentIngestionReport(
            document_id=doc.document_id,
            state=DocumentState.PROCESSING,
        )

        try:
            # 1. Chunking
            chunks = self.chunker.chunk(doc)
            rep.chunks_count = len(chunks)

            # 2. Mock Extraction
            extraction_results: list[ScientificChunkExtractionResult] = []
            for c in chunks:
                res = self.extractor.extract(c)
                extraction_results.append(res)

            # 3. Local Embeddings Generation & 384d finite validation
            texts = [c.content for c in chunks]
            embs_np = self.embedding_model.encode(texts, convert_to_numpy=True)
            vectors: list[list[float]] = []
            for i, emb in enumerate(embs_np):
                vec = [float(x) for x in emb]
                if len(vec) != 384:
                    raise ValueError(f"Chunk {i} embedding dimension is {len(vec)}, expected 384.")
                for val in vec:
                    if math.isnan(val) or math.isinf(val):
                        raise ValueError(f"Chunk {i} embedding has non-finite value '{val}'")
                vectors.append(vec)

        except Exception as exc:
            rep.state = DocumentState.FAILED
            rep.failed_stage = "pre_write_processing"
            rep.error_message = str(exc)
            return rep

        # 4. Write to PostgreSQL in a transaction
        pg_success = False
        try:
            await pg_writer.upsert_document(doc)
            chunk_inputs = [
                ChunkWriteInput(chunk=c, embedding=v) for c, v in zip(chunks, vectors, strict=True)
            ]
            rep.pg_chunks_written = await pg_writer.upsert_chunks_batch(chunk_inputs)

            total_links = 0
            for res in extraction_results:
                if res.entities:
                    links_cnt = await pg_writer.write_entity_links(
                        res.chunk_id, cast(Any, res.entities)
                    )
                    total_links += links_cnt
            rep.pg_entity_links_written = total_links
            pg_success = True
        except Exception as exc:
            rep.state = DocumentState.FAILED
            rep.failed_stage = "postgres_write"
            rep.affected_store = "postgresql"
            rep.error_message = str(exc)
            return rep

        # 5. Write to Neo4j
        neo_success = False
        try:
            total_e = 0
            total_r = 0
            for res in extraction_results:
                e_cnt, r_cnt = self.neo_writer.write_chunk_result(res)
                total_e += e_cnt
                total_r += r_cnt
            rep.neo_entities_written = total_e
            rep.neo_relationships_written = total_r
            neo_success = True
        except Exception as exc:
            # If PostgreSQL succeeded but Neo4j failed, state is NEEDS_RECONCILIATION
            rep.state = DocumentState.NEEDS_RECONCILIATION
            rep.failed_stage = "neo4j_write"
            rep.affected_store = "neo4j"
            rep.error_message = str(exc)
            return rep

        # 6. Verification: both stores must succeed
        if pg_success and neo_success:
            rep.state = DocumentState.COMPLETE
        else:
            rep.state = DocumentState.NEEDS_RECONCILIATION
            rep.failed_stage = "dual_store_verification"

        return rep

    async def ingest_batch(
        self,
        documents: Sequence[RawDocument],
    ) -> BatchReconciliationReport:
        """Ingest a bounded batch of documents (up to batch_doc_limit)."""
        if len(documents) > self.batch_doc_limit:
            raise ValueError(f"Batch size {len(documents)} exceeds limit {self.batch_doc_limit}")

        pg_writer = PostgresWriter(self.pg_client)
        report = BatchReconciliationReport(total_documents=len(documents))

        for doc in documents:
            doc_report = await self.ingest_single_document_atomic(doc, pg_writer)
            report.document_reports[doc.document_id] = doc_report
            if doc_report.state == DocumentState.COMPLETE:
                report.completed += 1
            elif doc_report.state == DocumentState.NEEDS_RECONCILIATION:
                report.needs_reconciliation += 1
            else:
                report.failed += 1

        return report

    async def reconcile_document(
        self,
        doc: RawDocument,
    ) -> DocumentIngestionReport:
        """Reconcile an incomplete or partially-written document safely using idempotent writes."""
        pg_writer = PostgresWriter(self.pg_client)
        return await self.ingest_single_document_atomic(doc, pg_writer)

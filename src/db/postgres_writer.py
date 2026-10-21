"""PostgreSQL chunk and vector embedding writer with idempotent upserts.

Design & Invariants:
- Implements idempotent persistence for documents, chunks (with pgvector embeddings), and entity_links.
- Uses parameterized SQL exclusively to prevent SQL injection.
- Preserves canonical UUID chunk IDs (UUID5 matching Neo4j provenance).
- Bounded batch operations within explicit database transactions.
- Provides atomic document-level writing: upsert document -> upsert chunks with embeddings -> upsert entity links.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from src.db.postgres import PostgresClient, PostgresClientError
from src.extraction.schemas import ExtractedEntity
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument

logger = logging.getLogger(__name__)


class PostgresWriterError(PostgresClientError):
    """Base exception for PostgreSQL write operations."""


class DocumentNotFoundError(PostgresWriterError):
    """Raised when attempting to write chunks for a non-existent parent document."""


class BatchUpsertError(PostgresWriterError):
    """Raised when a batch upsert transaction fails and is rolled back."""


@dataclass(frozen=True)
class ChunkWriteInput:
    """Input payload bundling a TextChunk with its optional vector embedding and extracted entities."""

    chunk: TextChunk
    embedding: list[float] | None = None
    entities: Sequence[ExtractedEntity] = ()


class PostgresWriter:
    """Manages transactional, idempotent writes of documents, chunks, and entity links to PostgreSQL."""

    def __init__(self, client: PostgresClient) -> None:
        self.client = client

    # ------------------------------------------------------------------
    # Document upsert
    # ------------------------------------------------------------------

    async def upsert_document(self, document: RawDocument) -> str:
        """Idempotently insert or update a document record in PostgreSQL.

        Uses ``ON CONFLICT (document_id) DO UPDATE``.

        Parameters
        ----------
        document:
            Validated RawDocument instance.

        Returns
        -------
        str
            The document_id of the upserted document.
        """
        sql = """
        INSERT INTO documents (
            document_id, source_uri, title, doc_type, content_hash, metadata, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6::jsonb, CURRENT_TIMESTAMP
        )
        ON CONFLICT (document_id) DO UPDATE SET
            source_uri = EXCLUDED.source_uri,
            title = EXCLUDED.title,
            doc_type = EXCLUDED.doc_type,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP
        RETURNING document_id;
        """
        metadata_json = json.dumps(document.metadata)
        try:
            async with self.client.connection() as conn:
                res = await conn.fetchval(
                    sql,
                    document.document_id,
                    document.source_uri,
                    document.title,
                    document.doc_type,
                    document.content_hash,
                    metadata_json,
                )
                logger.debug("Upserted document id='%s'", document.document_id)
                return str(res)
        except Exception as exc:
            logger.error(
                "Failed to upsert document id='%s': %s", document.document_id, exc, exc_info=True
            )
            raise PostgresWriterError(
                f"Failed to upsert document '{document.document_id}': {exc}"
            ) from exc

    async def upsert_documents_batch(
        self,
        documents: Sequence[RawDocument],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upsert a batch of RawDocument records in transactions.

        Parameters
        ----------
        documents:
            Sequence of RawDocument records.
        batch_size:
            Batch size per transaction.

        Returns
        -------
        int
            Number of documents upserted.
        """
        if not documents:
            return 0

        sql = """
        INSERT INTO documents (
            document_id, source_uri, title, doc_type, content_hash, metadata, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6::jsonb, CURRENT_TIMESTAMP
        )
        ON CONFLICT (document_id) DO UPDATE SET
            source_uri = EXCLUDED.source_uri,
            title = EXCLUDED.title,
            doc_type = EXCLUDED.doc_type,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP;
        """

        total_upserted = 0
        try:
            async with self.client.connection() as conn:
                for i in range(0, len(documents), batch_size):
                    batch = documents[i : i + batch_size]
                    records = [
                        (
                            doc.document_id,
                            doc.source_uri,
                            doc.title,
                            doc.doc_type,
                            doc.content_hash,
                            json.dumps(doc.metadata),
                        )
                        for doc in batch
                    ]
                    async with conn.transaction():
                        await conn.executemany(sql, records)
                    total_upserted += len(batch)
            logger.info("Upserted %d documents successfully", total_upserted)
            return total_upserted
        except Exception as exc:
            logger.error("Batch document upsert failed: %s", exc, exc_info=True)
            raise BatchUpsertError(f"Batch document upsert failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Chunks upsert (with vector embeddings)
    # ------------------------------------------------------------------

    async def upsert_chunks_batch(
        self,
        items: Sequence[ChunkWriteInput],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upsert a batch of text chunks and their embeddings into PostgreSQL.

        Parameters
        ----------
        items:
            Sequence of ChunkWriteInput objects.
        batch_size:
            Batch size per transaction.

        Returns
        -------
        int
            Number of chunks upserted.
        """
        # Support dual-dimension vector columns:
        # 1536-dim -> embedding, 384-dim -> embedding_384.
        # On conflict, preserve existing value in the other column using COALESCE.
        sql = """
        INSERT INTO chunks (
            id, document_id, chunk_index, content, token_count, char_start, char_end, metadata, embedding, embedding_384
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::vector, $10::vector
        )
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            id = EXCLUDED.id,
            content = EXCLUDED.content,
            token_count = EXCLUDED.token_count,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            metadata = EXCLUDED.metadata,
            embedding = COALESCE(EXCLUDED.embedding, chunks.embedding),
            embedding_384 = COALESCE(EXCLUDED.embedding_384, chunks.embedding_384);
        """

        total_upserted = 0
        try:
            async with self.client.connection() as conn:
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    records = []
                    for item in batch:
                        chunk = item.chunk
                        chunk_uuid = uuid.UUID(chunk.chunk_id)
                        emb = item.embedding
                        emb_1536 = str(emb) if emb is not None and len(emb) == 1536 else None
                        emb_384 = str(emb) if emb is not None and len(emb) == 384 else None
                        records.append(
                            (
                                chunk_uuid,
                                chunk.document_id,
                                chunk.chunk_index,
                                chunk.content,
                                chunk.char_length,  # approx token count/char len
                                chunk.char_start,
                                chunk.char_end,
                                json.dumps(chunk.metadata),
                                emb_1536,
                                emb_384,
                            )
                        )

                    async with conn.transaction():
                        await conn.executemany(sql, records)
                    total_upserted += len(batch)
            logger.info("Upserted %d chunks successfully", total_upserted)
            return total_upserted
        except Exception as exc:
            logger.error("Batch chunk upsert failed: %s", exc, exc_info=True)
            raise BatchUpsertError(f"Batch chunk upsert failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Entity links upsert
    # ------------------------------------------------------------------

    async def write_entity_links(
        self,
        chunk_id: str | uuid.UUID,
        entities: Sequence[ExtractedEntity],
    ) -> int:
        """Write relational entity links for a chunk, purging existing links first.

        Parameters
        ----------
        chunk_id:
            Canonical chunk UUID.
        entities:
            Extracted entities associated with this chunk.

        Returns
        -------
        int
            Number of entity links written.
        """
        c_uuid = uuid.UUID(str(chunk_id))
        if not entities:
            return 0

        delete_sql = "DELETE FROM entity_links WHERE chunk_id = $1;"
        insert_sql = """
        INSERT INTO entity_links (
            chunk_id, canonical_entity_id, entity_type, mention_text, confidence
        ) VALUES (
            $1, $2, $3, $4, $5
        );
        """

        records = [
            (
                c_uuid,
                ent.canonical_id,
                ent.entity_type,
                ent.name,
                float(ent.confidence),
            )
            for ent in entities
        ]

        try:
            async with self.client.connection() as conn:
                async with conn.transaction():
                    await conn.execute(delete_sql, c_uuid)
                    await conn.executemany(insert_sql, records)
            return len(records)
        except Exception as exc:
            logger.error(
                "Failed to write entity links for chunk '%s': %s", chunk_id, exc, exc_info=True
            )
            raise PostgresWriterError(
                f"Failed to write entity links for chunk '{chunk_id}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Atomic document & chunks write bundle
    # ------------------------------------------------------------------

    async def write_document_bundle(
        self,
        document: RawDocument,
        chunk_items: Sequence[ChunkWriteInput],
    ) -> tuple[int, int]:
        """Atomically upsert a parent document and all its chunks (with embeddings and entity links).

        Parameters
        ----------
        document:
            The parent RawDocument.
        chunk_items:
            The sequence of ChunkWriteInput items for this document.

        Returns
        -------
        tuple[int, int]
            (chunks_upserted, entity_links_written)
        """
        doc_sql = """
        INSERT INTO documents (
            document_id, source_uri, title, doc_type, content_hash, metadata, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6::jsonb, CURRENT_TIMESTAMP
        )
        ON CONFLICT (document_id) DO UPDATE SET
            source_uri = EXCLUDED.source_uri,
            title = EXCLUDED.title,
            doc_type = EXCLUDED.doc_type,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP;
        """

        chunk_sql = """
        INSERT INTO chunks (
            id, document_id, chunk_index, content, token_count, char_start, char_end, metadata, embedding, embedding_384
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::vector, $10::vector
        )
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            id = EXCLUDED.id,
            content = EXCLUDED.content,
            token_count = EXCLUDED.token_count,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            metadata = EXCLUDED.metadata,
            embedding = COALESCE(EXCLUDED.embedding, chunks.embedding),
            embedding_384 = COALESCE(EXCLUDED.embedding_384, chunks.embedding_384);
        """

        link_sql = """
        INSERT INTO entity_links (
            chunk_id, canonical_entity_id, entity_type, mention_text, confidence
        ) VALUES (
            $1, $2, $3, $4, $5
        );
        """

        total_links = 0
        try:
            async with self.client.connection() as conn:
                async with conn.transaction():
                    # 1. Upsert document
                    await conn.execute(
                        doc_sql,
                        document.document_id,
                        document.source_uri,
                        document.title,
                        document.doc_type,
                        document.content_hash,
                        json.dumps(document.metadata),
                    )

                    # 2. Upsert chunks
                    if chunk_items:
                        chunk_records = []
                        all_link_records = []
                        for item in chunk_items:
                            chunk = item.chunk
                            c_uuid = uuid.UUID(chunk.chunk_id)
                            emb = item.embedding
                            emb_1536 = str(emb) if emb is not None and len(emb) == 1536 else None
                            emb_384 = str(emb) if emb is not None and len(emb) == 384 else None
                            chunk_records.append(
                                (
                                    c_uuid,
                                    chunk.document_id,
                                    chunk.chunk_index,
                                    chunk.content,
                                    chunk.char_length,
                                    chunk.char_start,
                                    chunk.char_end,
                                    json.dumps(chunk.metadata),
                                    emb_1536,
                                    emb_384,
                                )
                            )

                            for ent in item.entities:
                                all_link_records.append(
                                    (
                                        c_uuid,
                                        ent.canonical_id,
                                        ent.entity_type,
                                        ent.name,
                                        float(ent.confidence),
                                    )
                                )

                        await conn.executemany(chunk_sql, chunk_records)

                        # 3. Clean and write entity links
                        if all_link_records:
                            chunk_uuids = [r[0] for r in chunk_records]
                            await conn.execute(
                                "DELETE FROM entity_links WHERE chunk_id = ANY($1::uuid[]);",
                                chunk_uuids,
                            )
                            await conn.executemany(link_sql, all_link_records)
                            total_links = len(all_link_records)

            logger.info(
                "Successfully wrote document bundle '%s' (%d chunks, %d entity links)",
                document.document_id,
                len(chunk_items),
                total_links,
            )
            return len(chunk_items), total_links

        except Exception as exc:
            logger.error(
                "Failed to write document bundle '%s': %s", document.document_id, exc, exc_info=True
            )
            raise PostgresWriterError(
                f"Failed to write document bundle '{document.document_id}': {exc}"
            ) from exc

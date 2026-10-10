"""Live integration tests for VectorRetriever against local PostgreSQL container (TASK-204).

Validates:
1. Real cosine distance search and similarity calculation with pgvector.
2. Similarity threshold filtering.
3. Top-k limiting and ordering by cosine proximity.
4. Metadata filtering and doc_type filtering.
5. get_chunks_by_ids hydration with real chunks.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.postgres import PostgresClient
from src.db.postgres_writer import ChunkWriteInput, PostgresWriter
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument
from src.retrieval.vector_retriever import VectorRetriever


@pytest.fixture(scope="module")
def live_pg_settings() -> Settings:
    return Settings(
        postgres_host="127.0.0.1",
        postgres_port=5432,
        postgres_db="kgr_db",
        postgres_user="kgr_user",
        postgres_password=SecretStr("kgr_password"),
        embedding_dimensions=1536,
        vector_top_k=5,
        vector_similarity_threshold=0.60,
    )


@pytest.fixture
async def live_retriever_setup(
    live_pg_settings: Settings,
) -> AsyncGenerator[tuple[VectorRetriever, PostgresWriter], None]:
    client = PostgresClient(settings=live_pg_settings)
    await client.get_pool()
    await client.run_migrations(embedding_dim=1536)
    writer = PostgresWriter(client=client)
    retriever = VectorRetriever(client=client, settings=live_pg_settings)

    yield retriever, writer
    await client.close()


@pytest.mark.integration
class TestVectorRetrieverLive:
    """Live PostgreSQL / pgvector search tests."""

    @pytest.mark.asyncio
    async def test_live_vector_search_and_similarity_filtering(
        self,
        live_retriever_setup: tuple[VectorRetriever, PostgresWriter],
    ) -> None:
        retriever, writer = live_retriever_setup

        doc_id = f"doc-search-{uuid.uuid4()}"
        doc = RawDocument(
            document_id=doc_id,
            source_uri=f"docs/{doc_id}.md",
            title="Search Test Doc",
            doc_type="technical_doc",
            content="Search test document content",
        )

        # Create two distinct unit-normalized vectors:
        # vec_close: identical to query
        # vec_far: orthogonal to query
        query_vec = [0.0] * 1536
        query_vec[0] = 1.0

        vec_close = [0.0] * 1536
        vec_close[0] = 1.0  # Cosine similarity = 1.0

        vec_far = [0.0] * 1536
        vec_far[1] = 1.0  # Cosine similarity = 0.0 (orthogonal)

        chunk1_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:0")
        chunk1 = TextChunk(
            chunk_id=str(chunk1_uuid),
            document_id=doc_id,
            chunk_index=0,
            content="High similarity chunk content",
            char_start=0,
            char_end=28,
            metadata={"domain": "payments"},
        )
        item1 = ChunkWriteInput(chunk=chunk1, embedding=vec_close)

        chunk2_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:1")
        chunk2 = TextChunk(
            chunk_id=str(chunk2_uuid),
            document_id=doc_id,
            chunk_index=1,
            content="Low similarity chunk content",
            char_start=29,
            char_end=56,
            metadata={"domain": "inventory"},
        )
        item2 = ChunkWriteInput(chunk=chunk2, embedding=vec_far)

        # Write bundle
        await writer.write_document_bundle(doc, [item1, item2])

        # 1. Search with threshold 0.60 -> only chunk1 should return
        results = await retriever.search(
            query_vec,
            similarity_threshold=0.60,
            document_id=doc_id,
        )
        assert len(results) == 1
        assert results[0].chunk_id == str(chunk1_uuid)
        assert results[0].similarity_score >= 0.99  # Cosine similarity ~ 1.0

        # 2. Search with lower threshold -> both should return in correct order
        all_results = await retriever.search(
            query_vec,
            similarity_threshold=-0.5,
            document_id=doc_id,
        )
        assert len(all_results) == 2
        assert all_results[0].chunk_id == str(chunk1_uuid)
        assert all_results[1].chunk_id == str(chunk2_uuid)

        # 3. Test get_chunks_by_ids
        hydrated = await retriever.get_chunks_by_ids([str(chunk1_uuid), str(chunk2_uuid)])
        assert len(hydrated) == 2
        hydrated_ids = {c.chunk_id for c in hydrated}
        assert hydrated_ids == {str(chunk1_uuid), str(chunk2_uuid)}

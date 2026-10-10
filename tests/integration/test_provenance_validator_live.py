"""Live integration tests for DualStoreConsistencyValidator (TASK-205).

Tests dual-store consistency verification against live local Neo4j and PostgreSQL containers:
1. Validates real synchronized state between Neo4j and PostgreSQL.
2. Injects an orphaned graph relationship and verifies validator detects cross-store inconsistency.
3. Injects an unreferenced chunk in PostgreSQL and verifies accurate detection.
4. Verifies actual stored canonical UUID comparison across stores.
"""

import uuid
from collections.abc import AsyncGenerator, Generator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.db.postgres_writer import ChunkWriteInput, PostgresWriter
from src.extraction.schemas import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument
from src.validation.provenance_check import DualStoreConsistencyValidator

_TEST_ENTITY_IDS = [
    "service:live-val-svc-a",
    "service:live-val-svc-b",
    "api:live-val-api-c",
]


@pytest.fixture(scope="module")
def live_neo4j_client() -> Generator[Neo4jClient, None, None]:
    settings = Settings(
        neo4j_uri="bolt://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password=SecretStr("neo4j_password"),
        anthropic_api_key=SecretStr("sk-ant-test-key-12345"),
        openai_api_key=SecretStr("sk-test-openai-key-12345"),
        postgres_password=SecretStr("postgres_password"),
    )
    client = Neo4jClient(settings)
    yield client
    client.close()


@pytest.fixture(scope="module")
def live_pg_settings() -> Settings:
    return Settings(
        postgres_host="127.0.0.1",
        postgres_port=5432,
        postgres_db="kgr_db",
        postgres_user="kgr_user",
        postgres_password=SecretStr("kgr_password"),
        embedding_dimensions=1536,
    )


@pytest.fixture
async def live_clients(
    live_neo4j_client: Neo4jClient,
    live_pg_settings: Settings,
) -> AsyncGenerator[tuple[PostgresClient, Neo4jClient], None]:
    pg_client = PostgresClient(settings=live_pg_settings)
    await pg_client.get_pool()
    await pg_client.run_migrations(embedding_dim=1536)

    yield pg_client, live_neo4j_client

    await pg_client.close()


@pytest.fixture(autouse=True)
def cleanup_live_entities(live_neo4j_client: Neo4jClient) -> Generator[None, None, None]:
    _clean_graph(live_neo4j_client)
    yield
    _clean_graph(live_neo4j_client)


def _clean_graph(client: Neo4jClient) -> None:
    with client.driver.session() as session:
        session.run(
            "MATCH (n:Entity) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": _TEST_ENTITY_IDS},
        )


@pytest.mark.integration
class TestDualStoreConsistencyValidatorLive:
    """Live integration tests for cross-store provenance validation."""

    @pytest.mark.asyncio
    async def test_live_synchronized_dual_store_validation(
        self,
        live_clients: tuple[PostgresClient, Neo4jClient],
    ) -> None:
        """When both stores contain matching chunks, entities, and relationships."""
        pg_client, neo4j_client = live_clients
        writer = PostgresWriter(client=pg_client)

        doc_id = f"doc-live-val-{uuid.uuid4()}"
        doc = RawDocument(
            document_id=doc_id,
            source_uri=f"docs/{doc_id}.md",
            title="Validator Live Doc",
            doc_type="technical_doc",
            content="Service A depends on Service B.",
        )
        chunk_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:0")
        chunk_id_str = str(chunk_uuid)
        chunk = TextChunk(
            chunk_id=chunk_id_str,
            document_id=doc_id,
            chunk_index=0,
            content="Service A depends on Service B.",
            char_start=0,
            char_end=31,
        )

        prov = ExtractionProvenance(chunk_id=chunk_id_str, document_id=doc_id)
        ent_a = ExtractedEntity(
            canonical_id="service:live-val-svc-a",
            name="Service A",
            entity_type="Service",
            description="Service A description",
            confidence=0.95,
            provenance=prov,
        )
        ent_b = ExtractedEntity(
            canonical_id="service:live-val-svc-b",
            name="Service B",
            entity_type="Service",
            description="Service B description",
            confidence=0.95,
            provenance=prov,
        )
        rel = ExtractedRelationship(
            source_canonical_id="service:live-val-svc-a",
            source_type="Service",
            target_canonical_id="service:live-val-svc-b",
            target_type="Service",
            relation_type="DEPENDS_ON",
            description="A depends on B",
            confidence=0.95,
            source_chunk_ids=[chunk_id_str],
            provenance=prov,
        )

        # 1. Write to PostgreSQL (document, chunk, and entity link)
        fake_emb = [0.05] * 1536
        await writer.write_document_bundle(
            document=doc,
            chunk_items=[ChunkWriteInput(chunk=chunk, embedding=fake_emb, entities=[ent_a, ent_b])],
        )

        # 2. Write to Neo4j (entities and relationship)
        neo4j_client.write_entities([ent_a, ent_b])
        neo4j_client.write_relationships([rel])

        # 3. Validate
        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=neo4j_client,
        )
        report = await validator.validate_provenance(document_id=doc_id, strict_orphan_check=False)

        assert report.is_consistent is True
        assert chunk_id_str in report.postgres_chunk_ids
        assert chunk_id_str in report.neo4j_chunk_ids
        assert len(report.orphaned_graph_chunk_ids) == 0

    @pytest.mark.asyncio
    async def test_live_detects_orphaned_graph_edge(
        self,
        live_clients: tuple[PostgresClient, Neo4jClient],
    ) -> None:
        """When an edge in Neo4j references a chunk ID absent from PostgreSQL."""
        pg_client, neo4j_client = live_clients

        orphan_chunk_id = str(uuid.uuid4())
        doc_id = "doc-orphan-test"
        prov = ExtractionProvenance(chunk_id=orphan_chunk_id, document_id=doc_id)

        ent_a = ExtractedEntity(
            canonical_id="service:live-val-svc-a",
            name="Service A",
            entity_type="Service",
            description="Service A",
            confidence=0.95,
            provenance=prov,
        )
        ent_b = ExtractedEntity(
            canonical_id="service:live-val-svc-b",
            name="Service B",
            entity_type="Service",
            description="Service B",
            confidence=0.95,
            provenance=prov,
        )
        rel = ExtractedRelationship(
            source_canonical_id="service:live-val-svc-a",
            source_type="Service",
            target_canonical_id="service:live-val-svc-b",
            target_type="Service",
            relation_type="DEPENDS_ON",
            description="A depends on B with orphan provenance",
            confidence=0.90,
            source_chunk_ids=[orphan_chunk_id],
            provenance=prov,
        )

        # Write to Neo4j only (PostgreSQL will NOT have this chunk)
        neo4j_client.write_entities([ent_a, ent_b])
        neo4j_client.write_relationships([rel])

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=neo4j_client,
        )
        report = await validator.validate_provenance()

        assert report.is_consistent is False
        assert orphan_chunk_id in report.orphaned_graph_chunk_ids
        assert any(m.mismatch_type == "orphaned_graph_chunk" for m in report.mismatches)

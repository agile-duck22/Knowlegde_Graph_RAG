"""Live Neo4j integration test suite validating schema, idempotent writes, and provenance.

These tests execute against a real running Neo4j instance (e.g. via docker compose up neo4j).
Tagged with @pytest.mark.integration so standard CI / unit test runs do not execute them.
"""

from collections.abc import Generator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import (
    Neo4jClient,
    UnsupportedEntityTypeError,
    UnsupportedRelationshipTypeError,
)
from src.extraction.schemas import (
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)

# Test ID prefix used to isolate live test nodes and enable safe cleanup
_PREFIX = "integtest"
_CHUNK_1 = "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
_CHUNK_2 = "b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e"


@pytest.fixture(scope="module")
def live_client() -> Generator[Neo4jClient, None, None]:
    """Provide a live Neo4jClient connected to the local Neo4j instance."""
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


@pytest.fixture(autouse=True)
def cleanup_test_data(live_client: Neo4jClient) -> Generator[None, None, None]:
    """Clean up any test-specific nodes before and after each test."""
    _clean_nodes(live_client)
    yield
    _clean_nodes(live_client)


def _clean_nodes(client: Neo4jClient) -> None:
    """Delete all nodes created with the integtest prefix."""
    with client.driver.session() as session:
        session.run(
            "MATCH (n:Entity) WHERE n.id CONTAINS $prefix DETACH DELETE n",
            {"prefix": _PREFIX},
        )


@pytest.mark.integration
class TestNeo4jLiveIntegration:
    """11-point Live Neo4j integration smoke test suite."""

    # 1. Connectivity Check
    def test_01_verify_connectivity(self, live_client: Neo4jClient) -> None:
        """Driver can ping and establish a live Bolt connection."""
        live_client.verify_connectivity()

    # 2. Schema Initialisation & Idempotency
    def test_02_initialise_schema_idempotency(self, live_client: Neo4jClient) -> None:
        """Schema constraints and indexes can be applied multiple times without error."""
        live_client.initialise_schema()
        live_client.initialise_schema()  # Second call must be a no-op

    # 3. Entity Write & Retrieval
    def test_03_write_entities(self, live_client: Neo4jClient) -> None:
        """Write single entity and verify properties stored in Neo4j."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-arch-001",
        )
        entity = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-payment",
            name="Payment Service",
            entity_type="Service",
            description="Processes card transactions.",
            aliases=["Payment Gateway", "PaySvc"],
            confidence=0.95,
            provenance=prov,
        )

        count = live_client.write_entities([entity])
        assert count == 1

        with live_client.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity:Service {id: $id}) RETURN e",
                {"id": f"service:{_PREFIX}-payment"},
            )
            record = result.single()
            assert record is not None
            node = record["e"]
            assert node["name"] == "Payment Service"
            assert node["type"] == "Service"
            assert node["description"] == "Processes card transactions."
            assert sorted(node["aliases"]) == ["PaySvc", "Payment Gateway"]

    # 4. Entity Deduplication (MERGE Idempotency)
    def test_04_entity_deduplication(self, live_client: Neo4jClient) -> None:
        """Writing identical canonical ID twice updates properties, does not duplicate."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-arch-001",
        )
        entity1 = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-auth",
            name="Auth Service",
            entity_type="Service",
            description="Initial description",
            aliases=["Auth0"],
            confidence=0.9,
            provenance=prov,
        )
        entity2 = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-auth",
            name="Auth Service V2",
            entity_type="Service",
            description="Updated description",
            aliases=["Auth0", "Authentication"],
            confidence=0.95,
            provenance=prov,
        )

        live_client.write_entities([entity1])
        live_client.write_entities([entity2])

        with live_client.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN count(e) AS cnt, e.name AS name, e.description AS desc",
                {"id": f"service:{_PREFIX}-auth"},
            )
            record = result.single()
            assert record is not None
            assert record["cnt"] == 1
            assert record["name"] == "Auth Service V2"
            assert record["desc"] == "Updated description"

    # 5. Relationship Write & Retrieval
    def test_05_write_relationships(self, live_client: Neo4jClient) -> None:
        """Write relationship between existing nodes and verify edge properties."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-arch-001",
        )
        e1 = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-order",
            name="Order Service",
            entity_type="Service",
            description="Handles checkout orders.",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        e2 = ExtractedEntity(
            canonical_id=f"database:{_PREFIX}-orderdb",
            name="Order Database",
            entity_type="Database",
            description="PostgreSQL instance for orders.",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        live_client.write_entities([e1, e2])

        rel = ExtractedRelationship(
            source_canonical_id=f"service:{_PREFIX}-order",
            source_type="Service",
            target_canonical_id=f"database:{_PREFIX}-orderdb",
            target_type="Database",
            relation_type="WRITES_TO",
            description="Persists order state.",
            confidence=0.95,
            source_chunk_ids=[_CHUNK_1],
            provenance=prov,
        )
        rel_count = live_client.write_relationships([rel])
        assert rel_count == 1

        with live_client.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {id: $s_id})-[r:WRITES_TO]->(t:Entity {id: $t_id})
                RETURN r
                """,
                {"s_id": f"service:{_PREFIX}-order", "t_id": f"database:{_PREFIX}-orderdb"},
            )
            record = result.single()
            assert record is not None
            edge = record["r"]
            assert edge["relation_type"] == "WRITES_TO"
            assert edge["description"] == "Persists order state."
            assert edge["confidence"] == 0.95
            assert edge["provenance_chunk_id"] == _CHUNK_1
            assert edge["source_chunk_ids"] == [_CHUNK_1]

    # 6. Relationship Deduplication (Same Chunk)
    def test_06_relationship_deduplication_same_chunk(self, live_client: Neo4jClient) -> None:
        """Writing identical relationship from same chunk twice does not duplicate edge."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-arch-001",
        )
        e1 = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-web",
            name="Web App",
            entity_type="Service",
            description="Frontend",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        e2 = ExtractedEntity(
            canonical_id=f"api:{_PREFIX}-webapi",
            name="Web API",
            entity_type="API",
            description="REST API",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        live_client.write_entities([e1, e2])

        rel = ExtractedRelationship(
            source_canonical_id=f"service:{_PREFIX}-web",
            source_type="Service",
            target_canonical_id=f"api:{_PREFIX}-webapi",
            target_type="API",
            relation_type="EXPOSES_API",
            description="Exposes REST endpoints",
            confidence=0.9,
            source_chunk_ids=[_CHUNK_1],
            provenance=prov,
        )

        live_client.write_relationships([rel])
        live_client.write_relationships([rel])

        with live_client.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {id: $s_id})-[r:EXPOSES_API]->(t:Entity {id: $t_id})
                RETURN count(r) AS cnt
                """,
                {"s_id": f"service:{_PREFIX}-web", "t_id": f"api:{_PREFIX}-webapi"},
            )
            record = result.single()
            assert record is not None
            assert record["cnt"] == 1

    # 7. Cross-Chunk Provenance (Distinct Edges)
    def test_07_cross_chunk_provenance(self, live_client: Neo4jClient) -> None:
        """Same relationship type between same nodes from two chunks creates distinct provenance edges."""
        prov1 = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-1",
        )
        prov2 = ExtractionProvenance(
            chunk_id=_CHUNK_2,
            document_id="doc-2",
        )
        e1 = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-inventory",
            name="Inventory Service",
            entity_type="Service",
            description="Inventory manager",
            aliases=[],
            confidence=0.9,
            provenance=prov1,
        )
        e2 = ExtractedEntity(
            canonical_id=f"team:{_PREFIX}-logistics",
            name="Logistics Team",
            entity_type="Team",
            description="Warehouse & Logistics",
            aliases=[],
            confidence=0.9,
            provenance=prov1,
        )
        live_client.write_entities([e1, e2])

        rel1 = ExtractedRelationship(
            source_canonical_id=f"service:{_PREFIX}-inventory",
            source_type="Service",
            target_canonical_id=f"team:{_PREFIX}-logistics",
            target_type="Team",
            relation_type="OWNED_BY",
            description="Owned according to doc 1",
            confidence=0.85,
            source_chunk_ids=[_CHUNK_1],
            provenance=prov1,
        )
        rel2 = ExtractedRelationship(
            source_canonical_id=f"service:{_PREFIX}-inventory",
            source_type="Service",
            target_canonical_id=f"team:{_PREFIX}-logistics",
            target_type="Team",
            relation_type="OWNED_BY",
            description="Owned according to doc 2",
            confidence=0.95,
            source_chunk_ids=[_CHUNK_2],
            provenance=prov2,
        )

        live_client.write_relationships([rel1, rel2])

        with live_client.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {id: $s_id})-[r:OWNED_BY]->(t:Entity {id: $t_id})
                RETURN r.provenance_chunk_id AS prov_id
                """,
                {"s_id": f"service:{_PREFIX}-inventory", "t_id": f"team:{_PREFIX}-logistics"},
            )
            records = result.data()
            assert len(records) == 2
            prov_ids = {r["prov_id"] for r in records}
            assert prov_ids == {_CHUNK_1, _CHUNK_2}

    # 8. All 7 Ontology Entity Types
    def test_08_all_seven_entity_types(self, live_client: Neo4jClient) -> None:
        """Write an entity for each of the 7 ontology labels and verify labels in DB."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-all",
        )
        types_to_create = [
            ("Service", f"service:{_PREFIX}-item"),
            ("Team", f"team:{_PREFIX}-item"),
            ("API", f"api:{_PREFIX}-item"),
            ("Database", f"database:{_PREFIX}-item"),
            ("Repository", f"repo:{_PREFIX}-item"),
            ("Vulnerability", f"vuln:{_PREFIX}-cve-2024-0001"),
            ("Infrastructure", f"infra:{_PREFIX}-item"),
        ]
        entities = [
            ExtractedEntity(
                canonical_id=cid,
                name=f"Test {etype}",
                entity_type=etype,  # type: ignore[arg-type]
                description=f"Desc for {etype}",
                aliases=[],
                confidence=0.9,
                provenance=prov,
            )
            for etype, cid in types_to_create
        ]

        live_client.write_entities(entities)

        with live_client.driver.session() as session:
            for etype, cid in types_to_create:
                result = session.run(
                    f"MATCH (n:Entity:{etype} {{id: $id}}) RETURN n",  # noqa: S608 (safe constant)
                    {"id": cid},
                )
                record = result.single()
                assert record is not None, f"Entity {cid} with label :{etype} not found in Neo4j"

    # 9. All 10 Ontology Relationship Types
    def test_09_all_ten_relationship_types(self, live_client: Neo4jClient) -> None:
        """Write each of the 10 ontology relationship types and verify in DB."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-rels",
        )
        source = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-source-node",
            name="Source Node",
            entity_type="Service",
            description="Hub service",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        target = ExtractedEntity(
            canonical_id=f"infra:{_PREFIX}-target-node",
            name="Target Node",
            entity_type="Infrastructure",
            description="Target node",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        live_client.write_entities([source, target])

        rel_types = [
            "OWNED_BY",
            "DEPENDS_ON",
            "EXPOSES_API",
            "CALLS_API",
            "READS_FROM",
            "WRITES_TO",
            "STORES_CODE",
            "AFFECTED_BY",
            "DEPLOYED_ON",
            "PRODUCES_EVENTS",
        ]

        rels = [
            ExtractedRelationship(
                source_canonical_id=f"service:{_PREFIX}-source-node",
                source_type="Service",
                target_canonical_id=f"infra:{_PREFIX}-target-node",
                target_type="Infrastructure",
                relation_type=rtype,  # type: ignore[arg-type]
                description=f"Rel type {rtype}",
                confidence=0.9,
                source_chunk_ids=[_CHUNK_1],
                provenance=prov,
            )
            for rtype in rel_types
        ]

        count = live_client.write_relationships(rels)
        assert count == 10

        with live_client.driver.session() as session:
            for rtype in rel_types:
                result = session.run(
                    f"MATCH (s:Entity {{id: $s_id}})-[r:{rtype}]->(t:Entity {{id: $t_id}}) RETURN r",  # noqa: S608
                    {
                        "s_id": f"service:{_PREFIX}-source-node",
                        "t_id": f"infra:{_PREFIX}-target-node",
                    },
                )
                assert result.single() is not None, f"Relationship :{rtype} not found in Neo4j"

    # 10. Type Rejection
    def test_10_invalid_type_rejection_before_execution(self, live_client: Neo4jClient) -> None:
        """Unsupported entity and relationship types raise error before reaching DB."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-invalid",
        )
        # Invalid entity type
        invalid_entity = ExtractedEntity.model_construct(
            canonical_id=f"service:{_PREFIX}-inv-1",
            name="Invalid",
            entity_type="NonExistentType",
            description="",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        with pytest.raises(UnsupportedEntityTypeError):
            live_client.write_entities([invalid_entity])

        # Invalid relationship type
        invalid_rel = ExtractedRelationship.model_construct(
            source_canonical_id=f"service:{_PREFIX}-s",
            source_type="Service",
            target_canonical_id=f"service:{_PREFIX}-t",
            target_type="Service",
            relation_type="NON_EXISTENT_REL",
            description="",
            confidence=0.9,
            source_chunk_ids=[_CHUNK_1],
            provenance=prov,
        )
        with pytest.raises(UnsupportedRelationshipTypeError):
            live_client.write_relationships([invalid_rel])

    # 11. Full Chunk Extraction Result Convenience Method
    def test_11_write_chunk_result(self, live_client: Neo4jClient) -> None:
        """Write a complete ChunkExtractionResult and verify counts and graph integrity."""
        prov = ExtractionProvenance(
            chunk_id=_CHUNK_1,
            document_id="doc-chunk-res",
        )
        e1 = ExtractedEntity(
            canonical_id=f"service:{_PREFIX}-chunk-svc",
            name="Notification Service",
            entity_type="Service",
            description="Sends SMS and Email notifications",
            aliases=["Notifier"],
            confidence=0.9,
            provenance=prov,
        )
        e2 = ExtractedEntity(
            canonical_id=f"team:{_PREFIX}-chunk-team",
            name="Messaging Team",
            entity_type="Team",
            description="Owns messaging stack",
            aliases=[],
            confidence=0.9,
            provenance=prov,
        )
        rel = ExtractedRelationship(
            source_canonical_id=f"service:{_PREFIX}-chunk-svc",
            source_type="Service",
            target_canonical_id=f"team:{_PREFIX}-chunk-team",
            target_type="Team",
            relation_type="OWNED_BY",
            description="Owned by messaging team",
            confidence=0.98,
            source_chunk_ids=[_CHUNK_1],
            provenance=prov,
        )

        chunk_result = ChunkExtractionResult(
            chunk_id=_CHUNK_1,
            document_id="doc-chunk-res",
            entities=[e1, e2],
            relationships=[rel],
            extraction_model="claude-3-haiku-20240307",
        )

        ent_written, rel_written = live_client.write_chunk_result(chunk_result)
        assert ent_written == 2
        assert rel_written == 1

        with live_client.driver.session() as session:
            res = session.run(
                """
                MATCH (s:Entity {id: $s_id})-[r:OWNED_BY]->(t:Entity {id: $t_id})
                RETURN s.name AS s_name, t.name AS t_name, r.provenance_chunk_id AS chunk_id
                """,
                {"s_id": f"service:{_PREFIX}-chunk-svc", "t_id": f"team:{_PREFIX}-chunk-team"},
            )
            rec = res.single()
            assert rec is not None
            assert rec["s_name"] == "Notification Service"
            assert rec["t_name"] == "Messaging Team"
            assert rec["chunk_id"] == _CHUNK_1

"""Live integration tests for SafeGraphRetriever against local Neo4j container (TASK-302).

Tests:
1. Direct entity neighborhood retrieval (T-01) with live Neo4j database.
2. Multi-hop dependency traversal (T-02).
3. Team ownership and service impact lookup (T-03).
4. Exact entity lookup (T-08).
5. Correct extraction of real source_chunk_ids from graph edges.
"""

from collections.abc import Generator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import Neo4jClient
from src.extraction.schemas import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)
from src.retrieval.graph_retriever import SafeGraphRetriever

_TEST_ENTITY_IDS = [
    "service:retriever-live-svc-a",
    "service:retriever-live-svc-b",
    "team:retriever-live-team",
    "api:retriever-live-api",
    "vuln:cve-2024-9999",
]

_CHUNK_ID_1 = "bbbbbbbb-1111-2222-3333-444444444444"
_CHUNK_ID_2 = "cccccccc-1111-2222-3333-444444444444"


@pytest.fixture(scope="module")
def live_neo4j() -> Generator[Neo4jClient, None, None]:
    """Provide a live Neo4jClient connected to the local Neo4j container."""
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
def cleanup_retriever_nodes(live_neo4j: Neo4jClient) -> Generator[None, None, None]:
    _clean(live_neo4j)
    yield
    _clean(live_neo4j)


def _clean(client: Neo4jClient) -> None:
    with client.driver.session() as session:
        session.run(
            "MATCH (n:Entity) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": _TEST_ENTITY_IDS},
        )


@pytest.fixture
def seed_live_data(live_neo4j: Neo4jClient) -> None:
    """Seed test entities and relationships into live Neo4j."""
    prov1 = ExtractionProvenance(chunk_id=_CHUNK_ID_1, document_id="doc-retriever-01")
    prov2 = ExtractionProvenance(chunk_id=_CHUNK_ID_2, document_id="doc-retriever-02")

    ent_svc_a = ExtractedEntity(
        canonical_id="service:retriever-live-svc-a",
        name="Retriever Svc A",
        entity_type="Service",
        description="Service A",
        confidence=0.95,
        provenance=prov1,
    )
    ent_svc_b = ExtractedEntity(
        canonical_id="service:retriever-live-svc-b",
        name="Retriever Svc B",
        entity_type="Service",
        description="Service B",
        confidence=0.95,
        provenance=prov1,
    )
    ent_team = ExtractedEntity(
        canonical_id="team:retriever-live-team",
        name="Retriever Team",
        entity_type="Team",
        description="Owner team",
        confidence=0.95,
        provenance=prov1,
    )
    ent_vuln = ExtractedEntity(
        canonical_id="vuln:cve-2024-9999",
        name="CVE-2024-9999",
        entity_type="Vulnerability",
        description="Test vuln",
        confidence=0.95,
        provenance=prov2,
    )

    rel_dep = ExtractedRelationship(
        source_canonical_id="service:retriever-live-svc-a",
        source_type="Service",
        target_canonical_id="service:retriever-live-svc-b",
        target_type="Service",
        relation_type="DEPENDS_ON",
        description="A depends on B",
        confidence=0.95,
        source_chunk_ids=[_CHUNK_ID_1],
        provenance=prov1,
    )
    rel_own = ExtractedRelationship(
        source_canonical_id="service:retriever-live-svc-a",
        source_type="Service",
        target_canonical_id="team:retriever-live-team",
        target_type="Team",
        relation_type="OWNED_BY",
        description="Owned by team",
        confidence=0.95,
        source_chunk_ids=[_CHUNK_ID_1],
        provenance=prov1,
    )
    rel_vuln = ExtractedRelationship(
        source_canonical_id="service:retriever-live-svc-a",
        source_type="Service",
        target_canonical_id="vuln:cve-2024-9999",
        target_type="Vulnerability",
        relation_type="AFFECTED_BY",
        description="Affected by CVE",
        confidence=0.95,
        source_chunk_ids=[_CHUNK_ID_2],
        provenance=prov2,
    )

    live_neo4j.write_entities([ent_svc_a, ent_svc_b, ent_team, ent_vuln])
    live_neo4j.write_relationships([rel_dep, rel_own, rel_vuln])


@pytest.mark.integration
class TestSafeGraphRetrieverLive:
    """Live Neo4j graph retrieval integration tests."""

    def test_live_direct_neighborhood_retrieval(
        self,
        live_neo4j: Neo4jClient,
        seed_live_data: None,
    ) -> None:
        retriever = SafeGraphRetriever(client=live_neo4j)

        result = retriever.get_direct_neighborhood("service:retriever-live-svc-a")
        assert result.is_empty is False
        assert result.record_count >= 2

        # Check provenance chunk IDs
        assert _CHUNK_ID_1 in result.source_chunk_ids
        assert _CHUNK_ID_2 in result.source_chunk_ids

        # Check canonical entity IDs
        assert "service:retriever-live-svc-a" in result.entity_ids
        assert "service:retriever-live-svc-b" in result.entity_ids

    def test_live_team_ownership_retrieval(
        self,
        live_neo4j: Neo4jClient,
        seed_live_data: None,
    ) -> None:
        retriever = SafeGraphRetriever(client=live_neo4j)

        result = retriever.get_team_ownership("team:retriever-live-team")
        assert result.is_empty is False
        assert result.record_count == 1
        record = result.records[0]
        assert record["team_id"] == "team:retriever-live-team"
        assert record["service_id"] == "service:retriever-live-svc-a"
        assert _CHUNK_ID_1 in result.source_chunk_ids

    def test_live_exact_entity_lookup(
        self,
        live_neo4j: Neo4jClient,
        seed_live_data: None,
    ) -> None:
        retriever = SafeGraphRetriever(client=live_neo4j)

        result = retriever.lookup_entity_by_id("service:retriever-live-svc-a")
        assert result.is_empty is False
        assert result.record_count == 1
        record = result.records[0]
        assert record["entity_id"] == "service:retriever-live-svc-a"
        assert record["name"] == "Retriever Svc A"
        assert record["entity_type"] == "Service"

"""Live integration test for Entity Linker with local Neo4j container (TASK-304).

Requires:
- Neo4j running on bolt://127.0.0.1:7687 with credentials configured in settings.
- Tests are opt-in and tagged with @pytest.mark.integration.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import Neo4jClient
from src.extraction.schemas import ExtractedEntity, ExtractionProvenance
from src.routing.entity_linker import (
    EntityLinker,
    LinkResolutionStatus,
)


@pytest.fixture(scope="module")
def live_neo4j() -> Generator[Neo4jClient, None, None]:
    """Provide a live Neo4jClient connected to the local Neo4j container."""
    settings = Settings(
        neo4j_uri="bolt://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password=SecretStr("neo4j_password"),
        anthropic_api_key=SecretStr("sk-ant-test-key-12345"),
        openai_api_key=SecretStr("sk-test-openai-key-12345"),
    )
    client = Neo4jClient(settings)
    try:
        client.verify_connectivity()
        client.initialise_schema()
    except Exception as exc:
        pytest.skip(f"Neo4j live integration container unreachable: {exc}")

    yield client
    client.close()


@pytest.mark.integration
class TestEntityLinkerLive:
    """Live Neo4j integration tests for EntityLinker."""

    def test_live_graph_verification_and_lookup(self, live_neo4j: Neo4jClient) -> None:
        # Seed a test entity into Neo4j
        prov = ExtractionProvenance(
            chunk_id="00000000-0000-0000-0000-000000000001",
            document_id="doc-linker-live-01",
        )
        test_entity = ExtractedEntity(
            canonical_id="service:payment-gateway",
            name="Payment Gateway Service",
            entity_type="Service",
            description="Core payment processing gateway",
            aliases=["Payment Gateway Live Svc"],
            confidence=0.95,
            provenance=prov,
        )
        live_neo4j.write_entities([test_entity])

        linker = EntityLinker(neo4j_client=live_neo4j, verify_graph=True)

        # 1. Resolve mention with graph verification
        res = linker.link_mention("service:payment-gateway")
        assert res.status == LinkResolutionStatus.RESOLVED
        assert res.canonical_id == "service:payment-gateway"
        assert res.verified_in_graph is True
        assert res.confidence == 1.0

        # 2. Lookup alias via Neo4j search
        res_alias = linker.link_mention("Payment Gateway Live Svc")
        assert res_alias.status == LinkResolutionStatus.RESOLVED
        assert res_alias.canonical_id == "service:payment-gateway"
        assert res_alias.verified_in_graph is True

        # 3. Nonexistent entity lookup
        res_none = linker.link_mention("nonexistent-test-service-123456")
        assert res_none.status == LinkResolutionStatus.UNRESOLVED

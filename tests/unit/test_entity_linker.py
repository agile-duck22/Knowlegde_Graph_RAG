"""Unit tests for Entity Linker (TASK-304).

Verifies:
- Exact canonical ID matches
- Alias and normalized-name resolution
- Multiple entity mentions in a query
- Unknown and ambiguous mentions
- Empty and whitespace input handling
- Entity type filtering and hints
- Neo4j graph verification and lookup with mocks
- Neo4j database errors and graceful handling
- Preservation of canonical IDs, entity types, and display names
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.db.neo4j import Neo4jClient, Neo4jConnectionError
from src.extraction.resolver import EntityResolver
from src.routing.entity_linker import (
    EntityLinker,
    EntityLinkingResult,
    LinkResolutionStatus,
)


class TestEntityLinkerInMemory:
    """Unit tests for in-memory entity resolution without Neo4j."""

    @pytest.fixture
    def linker(self) -> EntityLinker:
        return EntityLinker()

    def test_link_empty_and_whitespace_mention(self, linker: EntityLinker) -> None:
        link_empty = linker.link_mention("")
        assert link_empty.status == LinkResolutionStatus.UNRESOLVED
        assert link_empty.confidence == 0.0
        assert link_empty.canonical_id is None

        link_space = linker.link_mention("   \t  ")
        assert link_space.status == LinkResolutionStatus.UNRESOLVED

    def test_link_exact_canonical_ids(self, linker: EntityLinker) -> None:
        samples = [
            ("service:payment-gateway", "service:payment-gateway", "Service"),
            ("team:checkout-engineering", "team:checkout-engineering", "Team"),
            ("database:customer-orders-pg", "database:customer-orders-pg", "Database"),
            ("vuln:cve-2024-21626", "vuln:cve-2024-21626", "Vulnerability"),
            ("api:process-charge-v2", "api:process-charge-v2", "API"),
            ("infra:k8s-prod-cluster-us-east", "infra:k8s-prod-cluster-us-east", "Infrastructure"),
        ]
        for raw_id, expected_canon, expected_type in samples:
            res = linker.link_mention(raw_id)
            assert res.status == LinkResolutionStatus.RESOLVED
            assert res.canonical_id == expected_canon
            assert res.entity_type == expected_type
            assert res.confidence >= 0.95

    def test_link_aliases_and_display_names(self, linker: EntityLinker) -> None:
        aliases = [
            ("Payment Svc", "service:payment-gateway", "Service"),
            ("checkout-service", "service:checkout-service", "Service"),
            ("Core Checkout", "team:checkout-engineering", "Team"),
            ("iam team", "team:identity-access", "Team"),
            ("customer orders pg", "database:customer-orders-pg", "Database"),
            ("CVE-2024-21626", "vuln:cve-2024-21626", "Vulnerability"),
            ("cve_2023_38545", "vuln:cve-2023-38545", "Vulnerability"),
            ("process charge v2", "api:process-charge-v2", "API"),
        ]
        for alias, expected_canon, expected_type in aliases:
            res = linker.link_mention(alias)
            assert res.status == LinkResolutionStatus.RESOLVED, f"Failed for alias: {alias}"
            assert res.canonical_id == expected_canon
            assert res.entity_type == expected_type

    def test_link_ambiguous_mention(self) -> None:
        # Create a resolver with an ambiguous alias pointing to multiple entities
        from src.extraction.schemas import EntityType

        custom_aliases: dict[str, tuple[str, EntityType, str]] = {
            "ambig core": ("service:service-a", "Service", "Service A"),
            "ambig-core": ("service:service-b", "Service", "Service B"),
        }
        resolver = EntityResolver(custom_aliases=custom_aliases)
        linker = EntityLinker(resolver=resolver)

        res = linker.link_mention("ambig core")
        assert res.status == LinkResolutionStatus.AMBIGUOUS
        assert res.canonical_id is None
        assert res.confidence == 0.5
        assert set(res.candidate_ids) == {"service:service-a", "service:service-b"}

    def test_link_ambiguous_resolved_by_hint_type(self) -> None:
        # Alias registered for both a Service and a Team
        from src.extraction.schemas import EntityType

        custom_aliases: dict[str, tuple[str, EntityType, str]] = {
            "shared name": ("service:shared-svc", "Service", "Shared Service"),
            "shared-name": ("team:shared-team", "Team", "Shared Team"),
        }
        resolver = EntityResolver(custom_aliases=custom_aliases)
        linker = EntityLinker(resolver=resolver)

        # Without hint -> ambiguous
        res_ambig = linker.link_mention("shared name")
        assert res_ambig.status == LinkResolutionStatus.AMBIGUOUS

        # With hint -> successfully resolved
        res_svc = linker.link_mention("shared name", hint_type="Service")
        assert res_svc.status == LinkResolutionStatus.RESOLVED
        assert res_svc.canonical_id == "service:shared-svc"

        res_team = linker.link_mention("shared name", hint_type="Team")
        assert res_team.status == LinkResolutionStatus.RESOLVED
        assert res_team.canonical_id == "team:shared-team"

    def test_link_unknown_entity(self, linker: EntityLinker) -> None:
        res = linker.link_mention("completely unknown nonexistent service xyz 999")
        assert res.status == LinkResolutionStatus.UNRESOLVED
        assert res.canonical_id is None
        assert res.confidence == 0.0
        assert "could not be resolved" in (res.error_message or "")

    def test_link_query_extraction_and_resolution(self, linker: EntityLinker) -> None:
        query = (
            "Which services depend on service:payment-gateway or are owned by "
            "the Checkout Team and affected by CVE-2024-21626?"
        )
        result = linker.link_query(query)

        assert isinstance(result, EntityLinkingResult)
        assert result.query == query
        assert len(result.resolved_links) >= 3

        canon_ids = result.canonical_ids
        assert "service:payment-gateway" in canon_ids
        assert "team:checkout-engineering" in canon_ids
        assert "vuln:cve-2024-21626" in canon_ids

    def test_link_query_with_explicit_extracted_entities(self, linker: EntityLinker) -> None:
        query = "Show dependencies"
        entities = ["payment svc", "iam core"]
        result = linker.link_query(query, extracted_entities=entities)

        assert len(result.resolved_links) == 2
        assert "service:payment-gateway" in result.canonical_ids
        assert "service:auth-service" in result.canonical_ids

    def test_link_query_empty(self, linker: EntityLinker) -> None:
        result = linker.link_query("")
        assert result.query == ""
        assert len(result.links) == 0
        assert len(result.canonical_ids) == 0


class TestEntityLinkerWithNeo4j:
    """Unit tests for entity linking with mocked Neo4jClient."""

    @pytest.fixture
    def mock_neo4j_client(self) -> MagicMock:
        return MagicMock(spec=Neo4jClient)

    def test_link_with_graph_verification_success(self, mock_neo4j_client: MagicMock) -> None:
        # Mock single count record returning 1
        mock_record = {"cnt": 1}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_neo4j_client.driver.session.return_value.__enter__.return_value = mock_session

        linker = EntityLinker(neo4j_client=mock_neo4j_client, verify_graph=True)
        res = linker.link_mention("service:payment-gateway")

        assert res.status == LinkResolutionStatus.RESOLVED
        assert res.canonical_id == "service:payment-gateway"
        assert res.verified_in_graph is True
        assert res.confidence == 1.0

    def test_neo4j_lookup_for_unresolved_mention(self, mock_neo4j_client: MagicMock) -> None:
        # Mock database lookup returning 1 matching node
        mock_records = [
            {
                "entity_id": "service:custom-new-service",
                "name": "Custom New Service",
                "entity_type": "Service",
                "aliases": ["custom-new"],
            }
        ]
        mock_session = MagicMock()
        mock_session.run.return_value = iter(mock_records)
        mock_neo4j_client.driver.session.return_value.__enter__.return_value = mock_session

        linker = EntityLinker(neo4j_client=mock_neo4j_client)
        res = linker.link_mention("custom-new")

        assert res.status == LinkResolutionStatus.RESOLVED
        assert res.canonical_id == "service:custom-new-service"
        assert res.entity_type == "Service"
        assert res.verified_in_graph is True

    def test_neo4j_lookup_multiple_matches_is_ambiguous(self, mock_neo4j_client: MagicMock) -> None:
        # Mock database lookup returning 2 matching nodes
        mock_records = [
            {
                "entity_id": "service:cluster-service",
                "name": "Cluster",
                "entity_type": "Service",
            },
            {
                "entity_id": "infra:cluster-infra",
                "name": "Cluster",
                "entity_type": "Infrastructure",
            },
        ]
        mock_session = MagicMock()
        mock_session.run.return_value = iter(mock_records)
        mock_neo4j_client.driver.session.return_value.__enter__.return_value = mock_session

        linker = EntityLinker(neo4j_client=mock_neo4j_client)
        res = linker.link_mention("Cluster")

        assert res.status == LinkResolutionStatus.AMBIGUOUS
        assert set(res.candidate_ids) == {"service:cluster-service", "infra:cluster-infra"}
        assert res.confidence == 0.5

    def test_neo4j_database_error_handling(self, mock_neo4j_client: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session.run.side_effect = Neo4jConnectionError("Failed to connect to Neo4j")
        mock_neo4j_client.driver.session.return_value.__enter__.return_value = mock_session

        linker = EntityLinker(neo4j_client=mock_neo4j_client)
        # Mention not in memory catalog triggers database lookup
        res = linker.link_mention("unindexed-service-triggering-db-call")

        assert res.status == LinkResolutionStatus.DATABASE_ERROR
        assert res.canonical_id is None
        assert res.confidence == 0.0
        assert "Graph database error" in (res.error_message or "")

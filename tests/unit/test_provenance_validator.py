"""Offline unit tests for DualStoreConsistencyValidator using mocks (TASK-205).

Validates:
1. Complete consistency when chunk IDs match across Neo4j and PostgreSQL.
2. Detection of orphaned graph chunk references (in Neo4j but not PostgreSQL).
3. Detection of unreferenced PostgreSQL chunks (in PostgreSQL but not Neo4j).
4. Detection of malformed/invalid UUID strings in Neo4j provenance properties.
5. Detection of broken entity links.
6. Safe handling of database errors / connection failures.
7. Verification that validator compares actual stored canonical IDs, not just counts.
"""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient, PostgresClientError
from src.validation.provenance_check import (
    ConsistencyReport,
    DualStoreConsistencyValidator,
    DualStoreValidatorError,
    ProvenanceMismatch,
)


@pytest.fixture
def mock_postgres_settings() -> Settings:
    return Settings(
        environment="test",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="kgr_test_db",
        postgres_user="kgr_test_user",
        postgres_password=SecretStr("mock_secret_pw"),
        embedding_dimensions=1536,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=SecretStr("mock_neo4j_pw"),
    )


@pytest.fixture
def mock_driver() -> MagicMock:
    driver = MagicMock(name="neo4j_driver")
    session = MagicMock(name="session")
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = session
    return driver


@pytest.fixture
def mock_neo4j_client(mock_postgres_settings: Settings, mock_driver: MagicMock) -> Neo4jClient:
    with patch("src.db.neo4j.GraphDatabase.driver", return_value=mock_driver):
        return Neo4jClient(settings=mock_postgres_settings)


class TestDualStoreConsistencyValidatorUnit:
    """Offline unit tests for DualStoreConsistencyValidator."""

    @pytest.mark.asyncio
    async def test_provenance_validation_perfect_parity(
        self,
        mock_postgres_settings: Settings,
        mock_neo4j_client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """When all Neo4j relationships reference valid PostgreSQL chunk UUIDs."""
        chunk1_id = str(uuid.uuid4())
        chunk2_id = str(uuid.uuid4())

        empty_entity_links: list[dict[str, Any]] = []

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            [{"id": uuid.UUID(chunk1_id)}, {"id": uuid.UUID(chunk2_id)}],  # chunks
            empty_entity_links,  # entity_links
        ]
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        pg_client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)

        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = [
            # relationships
            [
                {
                    "relation_type": "DEPENDS_ON",
                    "provenance_chunk_id": chunk1_id,
                    "source_chunk_ids": [chunk1_id],
                    "source_id": "service:svc-a",
                    "target_id": "service:svc-b",
                },
                {
                    "relation_type": "CALLS_API",
                    "provenance_chunk_id": chunk2_id,
                    "source_chunk_ids": [chunk2_id],
                    "source_id": "service:svc-b",
                    "target_id": "api:charge-v1",
                },
            ],
            # entities
            [
                {"entity_id": "service:svc-a", "entity_type": "Service"},
                {"entity_id": "service:svc-b", "entity_type": "Service"},
                {"entity_id": "api:charge-v1", "entity_type": "API"},
            ],
        ]

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=mock_neo4j_client,
        )

        report = await validator.validate_provenance(strict_orphan_check=True)

        assert report.is_consistent is True
        assert report.total_postgres_chunks == 2
        assert report.total_neo4j_relationships == 2
        assert len(report.orphaned_graph_chunk_ids) == 0
        assert len(report.unreferenced_postgres_chunk_ids) == 0
        assert len(report.mismatches) == 0

    @pytest.mark.asyncio
    async def test_provenance_validation_detects_orphaned_graph_chunks(
        self,
        mock_postgres_settings: Settings,
        mock_neo4j_client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """When Neo4j contains chunk references missing from PostgreSQL."""
        pg_chunk = str(uuid.uuid4())
        orphan_chunk = str(uuid.uuid4())
        empty_entity_links: list[dict[str, Any]] = []
        empty_entities: list[dict[str, Any]] = []

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            [{"id": uuid.UUID(pg_chunk)}],  # chunks
            empty_entity_links,  # entity_links
        ]
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        pg_client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)

        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = [
            [
                {
                    "relation_type": "DEPENDS_ON",
                    "provenance_chunk_id": orphan_chunk,
                    "source_chunk_ids": [orphan_chunk],
                    "source_id": "service:svc-a",
                    "target_id": "service:svc-b",
                }
            ],
            empty_entities,  # entities
        ]

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=mock_neo4j_client,
        )

        report = await validator.validate_provenance(strict_orphan_check=False)

        assert report.is_consistent is False
        assert orphan_chunk in report.orphaned_graph_chunk_ids
        assert any(m.mismatch_type == "orphaned_graph_chunk" for m in report.mismatches)

    @pytest.mark.asyncio
    async def test_provenance_validation_detects_count_equal_but_id_mismatch(
        self,
        mock_postgres_settings: Settings,
        mock_neo4j_client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Equal counts (1 and 1) but different IDs must be detected as an inconsistency."""
        chunk_in_pg = str(uuid.uuid4())
        chunk_in_neo = str(uuid.uuid4())
        empty_entity_links: list[dict[str, Any]] = []
        empty_entities: list[dict[str, Any]] = []

        assert chunk_in_pg != chunk_in_neo

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            [{"id": uuid.UUID(chunk_in_pg)}],  # chunks
            empty_entity_links,  # entity_links
        ]
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        pg_client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)

        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = [
            [
                {
                    "relation_type": "DEPENDS_ON",
                    "provenance_chunk_id": chunk_in_neo,
                    "source_chunk_ids": [chunk_in_neo],
                    "source_id": "service:svc-a",
                    "target_id": "service:svc-b",
                }
            ],
            empty_entities,  # entities
        ]

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=mock_neo4j_client,
        )

        report = await validator.validate_provenance(strict_orphan_check=True)

        assert report.is_consistent is False
        assert chunk_in_neo in report.orphaned_graph_chunk_ids
        assert chunk_in_pg in report.unreferenced_postgres_chunk_ids
        assert len(report.mismatches) >= 2

    @pytest.mark.asyncio
    async def test_provenance_validation_detects_malformed_uuid(
        self,
        mock_postgres_settings: Settings,
        mock_neo4j_client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """Invalid non-UUID strings in Neo4j properties must be flagged."""
        pg_chunk = str(uuid.uuid4())
        empty_entity_links: list[dict[str, Any]] = []
        empty_entities: list[dict[str, Any]] = []

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            [{"id": uuid.UUID(pg_chunk)}],
            empty_entity_links,
        ]
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        pg_client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)

        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = [
            [
                {
                    "relation_type": "DEPENDS_ON",
                    "provenance_chunk_id": "invalid-non-uuid-string",
                    "source_chunk_ids": ["not_a_uuid"],
                    "source_id": "service:svc-a",
                    "target_id": "service:svc-b",
                }
            ],
            empty_entities,
        ]

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=mock_neo4j_client,
        )

        report = await validator.validate_provenance()

        assert report.is_consistent is False
        assert any(m.mismatch_type == "invalid_uuid_format" for m in report.mismatches)

    @pytest.mark.asyncio
    async def test_provenance_validation_detects_broken_entity_links(
        self,
        mock_postgres_settings: Settings,
        mock_neo4j_client: Neo4jClient,
        mock_driver: MagicMock,
    ) -> None:
        """When entity_links point to non-existent Neo4j entities."""
        valid_chunk = str(uuid.uuid4())
        link_id = str(uuid.uuid4())

        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = [
            [{"id": uuid.UUID(valid_chunk)}],
            [
                {
                    "id": uuid.UUID(link_id),
                    "chunk_id": uuid.UUID(valid_chunk),
                    "canonical_entity_id": "service:missing-service",
                    "entity_type": "Service",
                    "mention_text": "Missing Svc",
                }
            ],
        ]
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        pg_client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)

        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = [
            [
                {
                    "relation_type": "DEPENDS_ON",
                    "provenance_chunk_id": valid_chunk,
                    "source_chunk_ids": [valid_chunk],
                    "source_id": "service:svc-a",
                    "target_id": "service:svc-b",
                }
            ],
            [
                {"entity_id": "service:svc-a", "entity_type": "Service"},
                {"entity_id": "service:svc-b", "entity_type": "Service"},
            ],  # "service:missing-service" is absent
        ]

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=mock_neo4j_client,
        )

        report = await validator.validate_provenance(check_entity_links=True)

        assert report.is_consistent is False
        assert any(
            m.mismatch_type == "broken_entity_link" and "service:missing-service" in m.identifier
            for m in report.mismatches
        )

    @pytest.mark.asyncio
    async def test_provenance_validation_database_error_handling(
        self,
        mock_postgres_settings: Settings,
        mock_neo4j_client: Neo4jClient,
    ) -> None:
        """Database connection or query failure raises DualStoreValidatorError."""
        mock_pool = MagicMock()
        mock_pool.acquire.side_effect = PostgresClientError("Postgres unreachable")
        pg_client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)

        validator = DualStoreConsistencyValidator(
            postgres_client=pg_client,
            neo4j_client=mock_neo4j_client,
        )

        with pytest.raises(DualStoreValidatorError, match="Postgres unreachable"):
            await validator.validate_provenance()

    def test_consistency_report_summary_formatting(self) -> None:
        """Summary string renders cleanly."""
        report = ConsistencyReport(
            is_consistent=False,
            total_postgres_chunks=10,
            total_neo4j_relationships=5,
            total_graph_chunk_references=5,
            mismatches=[
                ProvenanceMismatch(
                    mismatch_type="orphaned_graph_chunk",
                    source_store="cross_store",
                    identifier="chunk-123",
                    details="Missing in PostgreSQL",
                )
            ],
        )
        summary = report.summary()
        assert "FAILED (INCONSISTENCIES FOUND)" in summary
        assert "orphaned_graph_chunk" in summary
        assert "chunk-123" in summary

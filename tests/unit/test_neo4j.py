"""Unit tests for src/db/neo4j.py — Neo4j driver, schema, and batch writes.

All tests run offline using ``unittest.mock``.  No live Neo4j instance is
required (or accessed).  Integration tests that need a real database are tagged
``@pytest.mark.integration`` and excluded from the default CI run.

Coverage targets
----------------
* Driver initialisation using Settings
* Context-manager entry / exit calls ``close()``
* ``verify_connectivity`` — happy path and ServiceUnavailable
* ``initialise_schema`` — correct DDL strings sent in order
* ``write_entities`` — parameterisation, template dispatch, batching, rollback
* ``write_relationships`` — parameterisation, template dispatch, batching, rollback
* ``write_chunk_result`` — delegates to both writers
* Ontology-allowlist enforcement for entities and relationships
* Empty-sequence short-circuit (zero DB calls)
* Cypher template safety — no f-string / format() interpolation of user data
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import (
    _ALLOWED_ENTITY_TYPES,
    _ALLOWED_RELATIONSHIP_TYPES,
    _CONSTRAINT_UNIQUE_ENTITY_ID,
    _ENTITY_UPSERT_TEMPLATES,
    _INDEX_ENTITY_FULLTEXT,
    _PING_QUERY,
    _RELATIONSHIP_BATCH_TEMPLATES,
    BatchWriteError,
    Neo4jClient,
    Neo4jConnectionError,
    UnsupportedEntityTypeError,
    UnsupportedRelationshipTypeError,
)
from src.extraction.schemas import (
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

CHUNK_ID = "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
DOC_ID = "doc-payment-gateway-spec"

_PROVENANCE = ExtractionProvenance(chunk_id=CHUNK_ID, document_id=DOC_ID)


def _make_entity(
    canonical_id: str = "service:payment-gateway",
    name: str = "Payment Gateway",
    entity_type: str = "Service",
) -> ExtractedEntity:
    return ExtractedEntity(
        canonical_id=canonical_id,
        name=name,
        entity_type=entity_type,  # type: ignore[arg-type]
        description="Processes card payments.",
        aliases=["pay-gw"],
        confidence=0.95,
        provenance=_PROVENANCE,
    )


def _make_relationship(
    source_id: str = "service:payment-gateway",
    target_id: str = "service:checkout-service",
    relation_type: str = "DEPENDS_ON",
) -> ExtractedRelationship:
    return ExtractedRelationship(
        source_canonical_id=source_id,
        source_type="Service",
        target_canonical_id=target_id,
        target_type="Service",
        relation_type=relation_type,  # type: ignore[arg-type]
        description="Synchronous gRPC dependency.",
        confidence=0.92,
        source_chunk_ids=[CHUNK_ID],
        provenance=_PROVENANCE,
    )


def _make_test_settings() -> Settings:
    return Settings(
        environment="test",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=SecretStr("test_password"),
    )


# ---------------------------------------------------------------------------
# Fixture: a fully mocked Neo4j driver hierarchy
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_driver() -> MagicMock:
    """Return a MagicMock that behaves like a neo4j.Driver."""
    driver = MagicMock(name="neo4j_driver")
    session = MagicMock(name="session")
    tx = MagicMock(name="transaction")

    # session context manager
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    # transaction context manager
    tx.__enter__ = MagicMock(return_value=tx)
    tx.__exit__ = MagicMock(return_value=False)

    session.begin_transaction.return_value = tx
    driver.session.return_value = session
    return driver


@pytest.fixture()
def client(mock_driver: MagicMock) -> Neo4jClient:
    """Return a Neo4jClient with the real driver replaced by mock_driver."""
    settings = _make_test_settings()
    with patch("src.db.neo4j.GraphDatabase.driver", return_value=mock_driver):
        neo4j_client = Neo4jClient(settings=settings)
    return neo4j_client


# ---------------------------------------------------------------------------
# 1. Allowlist constants
# ---------------------------------------------------------------------------


class TestAllowlists:
    def test_entity_types_count(self) -> None:
        assert len(_ALLOWED_ENTITY_TYPES) == 7

    def test_entity_types_values(self) -> None:
        expected = {
            "Service",
            "Team",
            "API",
            "Database",
            "Repository",
            "Vulnerability",
            "Infrastructure",
        }
        assert _ALLOWED_ENTITY_TYPES == expected

    def test_relationship_types_count(self) -> None:
        assert len(_ALLOWED_RELATIONSHIP_TYPES) == 10

    def test_relationship_types_values(self) -> None:
        expected = {
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
        }
        assert _ALLOWED_RELATIONSHIP_TYPES == expected

    def test_entity_template_keys_match_allowlist(self) -> None:
        assert set(_ENTITY_UPSERT_TEMPLATES.keys()) == _ALLOWED_ENTITY_TYPES

    def test_relationship_template_keys_match_allowlist(self) -> None:
        assert set(_RELATIONSHIP_BATCH_TEMPLATES.keys()) == _ALLOWED_RELATIONSHIP_TYPES


# ---------------------------------------------------------------------------
# 2. Cypher template safety checks
# ---------------------------------------------------------------------------


class TestCypherTemplateSafety:
    """Verify that no template contains Python f-string / format artefacts."""

    def test_entity_templates_contain_no_format_placeholders(self) -> None:
        """Templates must use $param notation, not {} or %s."""
        for entity_type, template in _ENTITY_UPSERT_TEMPLATES.items():
            # Curly braces in valid Cypher are only used for property maps like {id: $id}
            # There should be NO standalone {} that would indicate str.format() usage.
            assert "%" not in template, f"Template for {entity_type} contains '%'"

    def test_relationship_templates_contain_no_format_placeholders(self) -> None:
        for rel_type, template in _RELATIONSHIP_BATCH_TEMPLATES.items():
            assert "%" not in template, f"Template for {rel_type} contains '%'"

    def test_entity_templates_use_dollar_params(self) -> None:
        for entity_type, template in _ENTITY_UPSERT_TEMPLATES.items():
            assert "$id" in template, f"Template for {entity_type} missing $id"
            assert "$name" in template, f"Template for {entity_type} missing $name"

    def test_relationship_templates_use_rows_param(self) -> None:
        for rel_type, template in _RELATIONSHIP_BATCH_TEMPLATES.items():
            assert "$rows" in template, f"Template for {rel_type} missing $rows"
            assert "row.source_id" in template, f"Template for {rel_type} missing row.source_id"

    def test_constraint_is_literal_string(self) -> None:
        assert "IF NOT EXISTS" in _CONSTRAINT_UNIQUE_ENTITY_ID
        assert "Entity" in _CONSTRAINT_UNIQUE_ENTITY_ID
        assert "$" not in _CONSTRAINT_UNIQUE_ENTITY_ID  # no params in DDL

    def test_fulltext_index_is_literal_string(self) -> None:
        assert "IF NOT EXISTS" in _INDEX_ENTITY_FULLTEXT
        assert "entity_text_index" in _INDEX_ENTITY_FULLTEXT

    def test_each_entity_template_contains_correct_label(self) -> None:
        label_map = {
            "Service": ":Service",
            "Team": ":Team",
            "API": ":API",
            "Database": ":Database",
            "Repository": ":Repository",
            "Vulnerability": ":Vulnerability",
            "Infrastructure": ":Infrastructure",
        }
        for entity_type, label in label_map.items():
            assert label in _ENTITY_UPSERT_TEMPLATES[entity_type], (
                f"Template for {entity_type} does not contain literal label '{label}'"
            )

    def test_each_relationship_template_contains_correct_type(self) -> None:
        for rel_type, template in _RELATIONSHIP_BATCH_TEMPLATES.items():
            assert f":{rel_type}" in template or f"[r:{rel_type}" in template, (
                f"Template for {rel_type} does not contain literal relationship type"
            )


# ---------------------------------------------------------------------------
# 3. Driver initialisation
# ---------------------------------------------------------------------------


class TestDriverInitialisation:
    def test_driver_created_with_correct_uri(self) -> None:
        settings = _make_test_settings()
        with patch("src.db.neo4j.GraphDatabase.driver") as mock_gdb:
            mock_gdb.return_value = MagicMock()
            Neo4jClient(settings=settings)
            mock_gdb.assert_called_once_with(
                "bolt://localhost:7687",
                auth=("neo4j", "test_password"),
            )

    def test_driver_created_with_default_settings(self) -> None:
        with (
            patch("src.db.neo4j.GraphDatabase.driver") as mock_gdb,
            patch("src.db.neo4j.get_settings") as mock_get,
        ):
            mock_get.return_value = _make_test_settings()
            mock_gdb.return_value = MagicMock()
            Neo4jClient()
            mock_gdb.assert_called_once()

    def test_close_delegates_to_driver(self, client: Neo4jClient, mock_driver: MagicMock) -> None:
        client.close()
        mock_driver.close.assert_called_once()

    def test_context_manager_calls_close(self, mock_driver: MagicMock) -> None:
        settings = _make_test_settings()
        with patch("src.db.neo4j.GraphDatabase.driver", return_value=mock_driver):
            with Neo4jClient(settings=settings):
                pass
        mock_driver.close.assert_called_once()

    def test_driver_property_returns_driver(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        assert client.driver is mock_driver


# ---------------------------------------------------------------------------
# 4. verify_connectivity
# ---------------------------------------------------------------------------


class TestVerifyConnectivity:
    def test_happy_path(self, client: Neo4jClient, mock_driver: MagicMock) -> None:
        session = mock_driver.session.return_value
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(side_effect=lambda key: 1 if key == "ping" else None)
        session.run.return_value.single.return_value = mock_record

        client.verify_connectivity()  # should not raise

        session.run.assert_called_once_with(_PING_QUERY)

    def test_raises_connection_error_on_service_unavailable(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        from neo4j.exceptions import ServiceUnavailable

        session = mock_driver.session.return_value
        session.run.side_effect = ServiceUnavailable("refused")

        with pytest.raises(Neo4jConnectionError, match="Cannot connect"):
            client.verify_connectivity()

    def test_raises_connection_error_on_unexpected_result(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        session.run.return_value.single.return_value = None

        with pytest.raises(Neo4jConnectionError):
            client.verify_connectivity()


# ---------------------------------------------------------------------------
# 5. initialise_schema
# ---------------------------------------------------------------------------


class TestInitialiseSchema:
    def test_constraint_ddl_is_executed(self, client: Neo4jClient, mock_driver: MagicMock) -> None:
        session = mock_driver.session.return_value
        client.initialise_schema()

        calls = [c.args[0] for c in session.run.call_args_list]
        assert _CONSTRAINT_UNIQUE_ENTITY_ID in calls

    def test_fulltext_index_ddl_is_executed(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        client.initialise_schema()

        calls = [c.args[0] for c in session.run.call_args_list]
        assert _INDEX_ENTITY_FULLTEXT in calls

    def test_constraint_is_run_before_index(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        client.initialise_schema()

        calls = [c.args[0] for c in session.run.call_args_list]
        assert calls.index(_CONSTRAINT_UNIQUE_ENTITY_ID) < calls.index(_INDEX_ENTITY_FULLTEXT)


# ---------------------------------------------------------------------------
# 6. write_entities
# ---------------------------------------------------------------------------


class TestWriteEntities:
    def test_empty_sequence_returns_zero(self, client: Neo4jClient, mock_driver: MagicMock) -> None:
        result = client.write_entities([])
        assert result == 0
        mock_driver.session.assert_not_called()

    def test_single_entity_uses_correct_template(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        entity = _make_entity()
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_entities([entity])

        tx.run.assert_called_once()
        actual_query = tx.run.call_args.args[0]
        assert actual_query == _ENTITY_UPSERT_TEMPLATES["Service"]

    def test_parameters_contain_canonical_id(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        entity = _make_entity(canonical_id="service:payment-gateway")
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_entities([entity])

        params: dict[str, Any] = tx.run.call_args.args[1]
        assert params["id"] == "service:payment-gateway"
        assert params["name"] == "Payment Gateway"
        assert params["entity_type"] == "Service"

    def test_parameters_contain_aliases_as_list(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        entity = _make_entity()
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_entities([entity])

        params: dict[str, Any] = tx.run.call_args.args[1]
        assert isinstance(params["aliases"], list)
        assert "pay-gw" in params["aliases"]

    def test_returns_count_of_written_entities(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        entities = [
            _make_entity(),
            _make_entity(canonical_id="service:checkout-service", name="Checkout"),
        ]
        result = client.write_entities(entities)
        assert result == 2

    def test_batching_creates_multiple_transactions(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        entities = [
            _make_entity(canonical_id=f"service:svc-{i}", name=f"Svc {i}") for i in range(5)
        ]
        session = mock_driver.session.return_value
        client.write_entities(entities, batch_size=2)

        # 5 entities / batch_size 2 = 3 batches -> 3 begin_transaction calls
        assert session.begin_transaction.call_count == 3

    def test_all_entity_type_templates_are_dispatched(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        type_map = {
            "Service": "service:svc-a",
            "Team": "team:svc-a",
            "API": "api:svc-a",
            "Database": "database:svc-a",
            "Repository": "repo:svc-a",
            "Vulnerability": "vuln:cve-2024-00001",
            "Infrastructure": "infra:svc-a",
        }
        name_map = {
            "Service": "Svc A",
            "Team": "Team A",
            "API": "API A",
            "Database": "DB A",
            "Repository": "Repo A",
            "Vulnerability": "CVE-2024-00001",
            "Infrastructure": "Infra A",
        }
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        for entity_type, canonical_id in type_map.items():
            tx.run.reset_mock()
            entity = _make_entity(
                canonical_id=canonical_id,
                name=name_map[entity_type],
                entity_type=entity_type,
            )
            client.write_entities([entity])
            actual_query = tx.run.call_args.args[0]
            assert actual_query == _ENTITY_UPSERT_TEMPLATES[entity_type]

    def test_unsupported_entity_type_raises_before_db_call(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        # Bypass Pydantic by building a mock with a bad type
        bad_entity = MagicMock(spec=ExtractedEntity)
        bad_entity.entity_type = "UnknownType"

        with pytest.raises(UnsupportedEntityTypeError, match="UnknownType"):
            client.write_entities([bad_entity])

        mock_driver.session.assert_not_called()

    def test_batch_failure_raises_batch_write_error(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value
        tx.run.side_effect = RuntimeError("DB write failed")

        with pytest.raises(BatchWriteError, match="rolled back"):
            client.write_entities([_make_entity()])

    def test_transaction_commit_is_called(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_entities([_make_entity()])

        tx.commit.assert_called_once()


# ---------------------------------------------------------------------------
# 7. write_relationships
# ---------------------------------------------------------------------------


class TestWriteRelationships:
    def test_empty_sequence_returns_zero(self, client: Neo4jClient, mock_driver: MagicMock) -> None:
        result = client.write_relationships([])
        assert result == 0
        mock_driver.session.assert_not_called()

    def test_single_relationship_uses_correct_template(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rel = _make_relationship(relation_type="DEPENDS_ON")
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_relationships([rel])

        tx.run.assert_called_once()
        actual_query = tx.run.call_args.args[0]
        assert actual_query == _RELATIONSHIP_BATCH_TEMPLATES["DEPENDS_ON"]

    def test_rows_parameter_contains_source_and_target(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rel = _make_relationship()
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_relationships([rel])

        params: dict[str, Any] = tx.run.call_args.args[1]
        rows = params["rows"]
        assert len(rows) == 1
        assert rows[0]["source_id"] == "service:payment-gateway"
        assert rows[0]["target_id"] == "service:checkout-service"
        assert rows[0]["relation_type"] == "DEPENDS_ON"

    def test_rows_contain_provenance_chunk_id(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rel = _make_relationship()
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_relationships([rel])

        params: dict[str, Any] = tx.run.call_args.args[1]
        rows = params["rows"]
        assert rows[0]["provenance_chunk_id"] == CHUNK_ID

    def test_rows_contain_source_chunk_ids_as_list(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rel = _make_relationship()
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_relationships([rel])

        params: dict[str, Any] = tx.run.call_args.args[1]
        rows = params["rows"]
        assert isinstance(rows[0]["source_chunk_ids"], list)
        assert CHUNK_ID in rows[0]["source_chunk_ids"]

    def test_returns_count_of_written_relationships(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rel1 = _make_relationship()
        rel2 = ExtractedRelationship(
            source_canonical_id="service:checkout-service",
            source_type="Service",
            target_canonical_id="database:orders-db",
            target_type="Database",
            relation_type="READS_FROM",
            description="",
            confidence=0.9,
            source_chunk_ids=[CHUNK_ID],
            provenance=_PROVENANCE,
        )
        result = client.write_relationships([rel1, rel2])
        assert result == 2

    def test_batching_creates_multiple_transactions(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rels = [_make_relationship() for _ in range(5)]
        session = mock_driver.session.return_value
        client.write_relationships(rels, batch_size=2)

        assert session.begin_transaction.call_count == 3

    def test_relationships_grouped_by_type_in_single_batch(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        rel1 = _make_relationship(relation_type="DEPENDS_ON")
        rel2 = ExtractedRelationship(
            source_canonical_id="service:checkout-service",
            source_type="Service",
            target_canonical_id="database:orders-db",
            target_type="Database",
            relation_type="READS_FROM",
            description="",
            confidence=0.9,
            source_chunk_ids=[CHUNK_ID],
            provenance=_PROVENANCE,
        )
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_relationships([rel1, rel2])

        # Two different types => two tx.run calls (one per template)
        assert tx.run.call_count == 2
        queries = {c.args[0] for c in tx.run.call_args_list}
        assert _RELATIONSHIP_BATCH_TEMPLATES["DEPENDS_ON"] in queries
        assert _RELATIONSHIP_BATCH_TEMPLATES["READS_FROM"] in queries

    def test_all_relationship_type_templates_are_dispatched(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        """Verify that every ontology relationship type routes to its template."""
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        type_source_target: list[tuple[str, str, str, str, str]] = [
            ("OWNED_BY", "service:svc-a", "Service", "team:team-a", "Team"),
            ("DEPENDS_ON", "service:svc-a", "Service", "service:svc-b", "Service"),
            ("EXPOSES_API", "service:svc-a", "Service", "api:api-a", "API"),
            ("CALLS_API", "service:svc-a", "Service", "api:api-b", "API"),
            ("READS_FROM", "service:svc-a", "Service", "database:db-a", "Database"),
            ("WRITES_TO", "service:svc-a", "Service", "database:db-b", "Database"),
            ("STORES_CODE", "repo:repo-a", "Repository", "service:svc-a", "Service"),
            ("AFFECTED_BY", "service:svc-a", "Service", "vuln:cve-2024-00001", "Vulnerability"),
            ("DEPLOYED_ON", "service:svc-a", "Service", "infra:infra-a", "Infrastructure"),
            ("PRODUCES_EVENTS", "service:svc-a", "Service", "infra:kafka-bus", "Infrastructure"),
        ]

        for rel_type, src_id, src_type, tgt_id, tgt_type in type_source_target:
            tx.run.reset_mock()
            rel = ExtractedRelationship(
                source_canonical_id=src_id,
                source_type=src_type,  # type: ignore[arg-type]
                target_canonical_id=tgt_id,
                target_type=tgt_type,  # type: ignore[arg-type]
                relation_type=rel_type,  # type: ignore[arg-type]
                description="test",
                confidence=0.9,
                source_chunk_ids=[CHUNK_ID],
                provenance=_PROVENANCE,
            )
            client.write_relationships([rel])
            actual_query = tx.run.call_args.args[0]
            assert actual_query == _RELATIONSHIP_BATCH_TEMPLATES[rel_type]

    def test_unsupported_relationship_type_raises_before_db_call(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        bad_rel = MagicMock(spec=ExtractedRelationship)
        bad_rel.relation_type = "INVENTED_TYPE"

        with pytest.raises(UnsupportedRelationshipTypeError, match="INVENTED_TYPE"):
            client.write_relationships([bad_rel])

        mock_driver.session.assert_not_called()

    def test_batch_failure_raises_batch_write_error(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value
        tx.run.side_effect = RuntimeError("relationship write failed")

        with pytest.raises(BatchWriteError, match="rolled back"):
            client.write_relationships([_make_relationship()])

    def test_transaction_commit_is_called(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        client.write_relationships([_make_relationship()])

        tx.commit.assert_called_once()


# ---------------------------------------------------------------------------
# 8. write_chunk_result
# ---------------------------------------------------------------------------


class TestWriteChunkResult:
    def _make_chunk_result(self) -> ChunkExtractionResult:
        entity = _make_entity()
        rel = _make_relationship()
        return ChunkExtractionResult(
            chunk_id=CHUNK_ID,
            document_id=DOC_ID,
            extraction_model="claude-3-5-sonnet-20241022",
            entities=[entity],
            relationships=[rel],
        )

    def test_returns_entity_and_relationship_counts(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        result = self._make_chunk_result()
        entities_written, rels_written = client.write_chunk_result(result)
        assert entities_written == 1
        assert rels_written == 1

    def test_entities_written_before_relationships(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        session = mock_driver.session.return_value
        tx = session.begin_transaction.return_value

        result = self._make_chunk_result()
        client.write_chunk_result(result)

        # The entity template should appear in calls before the relationship template.
        all_queries = [c.args[0] for c in tx.run.call_args_list]
        entity_template = _ENTITY_UPSERT_TEMPLATES["Service"]
        rel_template = _RELATIONSHIP_BATCH_TEMPLATES["DEPENDS_ON"]
        assert entity_template in all_queries
        assert rel_template in all_queries
        assert all_queries.index(entity_template) < all_queries.index(rel_template)

    def test_empty_chunk_result_returns_zeros(
        self, client: Neo4jClient, mock_driver: MagicMock
    ) -> None:
        empty_result = ChunkExtractionResult(
            chunk_id=CHUNK_ID,
            document_id=DOC_ID,
            extraction_model="claude-3-5-sonnet-20241022",
            entities=[],
            relationships=[],
        )
        e, r = client.write_chunk_result(empty_result)
        assert e == 0
        assert r == 0
        mock_driver.session.assert_not_called()

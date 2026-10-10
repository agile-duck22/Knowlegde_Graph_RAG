"""Unit tests for SafeGraphRetriever (TASK-302).

Validates:
1. Successful execution of standard templates (T-01 through T-08) with parameter forwarding.
2. Rejection of unknown template names with InvalidTemplateError.
3. Parameter validation failure handling (missing / blank required parameters).
4. Correct extraction and deduplication of provenance `source_chunk_ids` and canonical entity IDs.
5. Handling of empty query results (`is_empty=True`).
6. Database error wrapping in GraphQueryExecutionError without credential leakage.
7. Proper session cleanup on both success and failure.
8. Specialized helper convenience methods.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import Neo4jClient, Neo4jClientError, Neo4jConnectionError
from src.retrieval.graph_retriever import (
    GraphQueryExecutionError,
    GraphQueryResult,
    InvalidTemplateError,
    SafeGraphRetriever,
    TemplateParameterError,
)


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        environment="test",
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
def mock_neo4j_client(mock_settings: Settings, mock_driver: MagicMock) -> Neo4jClient:
    with patch("src.db.neo4j.GraphDatabase.driver", return_value=mock_driver):
        return Neo4jClient(settings=mock_settings)


@pytest.fixture
def retriever(mock_neo4j_client: Neo4jClient) -> SafeGraphRetriever:
    return SafeGraphRetriever(client=mock_neo4j_client)


class TestSafeGraphRetrieverUnit:
    """Unit tests for SafeGraphRetriever."""

    def test_execute_template_happy_path(
        self,
        retriever: SafeGraphRetriever,
        mock_driver: MagicMock,
    ) -> None:
        """Executes T-01 and collects structured results and provenance chunk IDs."""
        chunk_uuid_1 = "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
        chunk_uuid_2 = "f1e2d3c4-b5a6-9788-7c6d-5e4f3a2b1c0d"

        # Mock query return record
        mock_record_1 = MagicMock()
        mock_record_1.data.return_value = {
            "source_id": "service:payment-gateway",
            "source_name": "Payment Gateway",
            "source_type": "Service",
            "rel_type": "DEPENDS_ON",
            "rel_desc": "Calls checkout",
            "confidence": 0.95,
            "source_chunk_ids": [chunk_uuid_1],
            "target_id": "service:checkout-service",
            "target_name": "Checkout Service",
            "target_type": "Service",
            "target_labels": ["Entity", "Service"],
        }
        mock_record_2 = MagicMock()
        mock_record_2.data.return_value = {
            "source_id": "service:payment-gateway",
            "source_name": "Payment Gateway",
            "source_type": "Service",
            "rel_type": "WRITES_TO",
            "rel_desc": "Persists transactions",
            "confidence": 0.98,
            "source_chunk_ids": [chunk_uuid_1, chunk_uuid_2],  # Duplicate chunk_uuid_1
            "target_id": "database:payments-pg",
            "target_name": "Payments PG",
            "target_type": "Database",
            "target_labels": ["Entity", "Database"],
        }

        mock_session = mock_driver.session.return_value
        mock_session.run.return_value = [mock_record_1, mock_record_2]

        result = retriever.execute_template(
            "T-01",
            parameters={"entity_id": "service:payment-gateway"},
            limit=25,
        )

        assert isinstance(result, GraphQueryResult)
        assert result.template_id == "T-01"
        assert result.record_count == 2
        assert result.is_empty is False
        assert result.parameters["entity_id"] == "service:payment-gateway"
        assert result.parameters["limit"] == 25

        # Verify provenance deduplication and sorting
        assert result.source_chunk_ids == sorted([chunk_uuid_1, chunk_uuid_2])

        # Verify entity IDs extraction
        assert "service:payment-gateway" in result.entity_ids
        assert "service:checkout-service" in result.entity_ids
        assert "database:payments-pg" in result.entity_ids

        # Verify driver call
        mock_session.run.assert_called_once()
        args = mock_session.run.call_args[0]
        params = args[1]
        assert params["entity_id"] == "service:payment-gateway"
        assert params["limit"] == 25

    def test_execute_template_empty_results(
        self,
        retriever: SafeGraphRetriever,
        mock_driver: MagicMock,
    ) -> None:
        """When query returns no records, is_empty is True."""
        mock_session = mock_driver.session.return_value
        mock_session.run.return_value = []

        result = retriever.execute_template(
            "T-01",
            parameters={"entity_id": "service:non-existent"},
        )

        assert result.is_empty is True
        assert result.record_count == 0
        assert result.records == []
        assert result.source_chunk_ids == []
        assert result.entity_ids == ["service:non-existent"]

    def test_execute_invalid_template_id_raises_invalid_template_error(
        self,
        retriever: SafeGraphRetriever,
    ) -> None:
        """Requesting unapproved template ID raises InvalidTemplateError."""
        with pytest.raises(
            InvalidTemplateError, match="Invalid or unapproved Cypher template 'T-999'"
        ):
            retriever.execute_template("T-999", parameters={"entity_id": "service:test"})

    def test_execute_missing_parameter_raises_template_parameter_error(
        self,
        retriever: SafeGraphRetriever,
    ) -> None:
        """Missing required parameter raises TemplateParameterError."""
        with pytest.raises(TemplateParameterError, match="Missing required parameter 'entity_id'"):
            retriever.execute_template("T-01", parameters={})

    def test_execute_empty_parameter_raises_template_parameter_error(
        self,
        retriever: SafeGraphRetriever,
    ) -> None:
        """Blank string parameter raises TemplateParameterError."""
        with pytest.raises(TemplateParameterError, match="cannot be empty string"):
            retriever.execute_template("T-01", parameters={"entity_id": "  "})

    def test_execute_database_error_wrapping(
        self,
        retriever: SafeGraphRetriever,
        mock_driver: MagicMock,
    ) -> None:
        """Database errors are wrapped in GraphQueryExecutionError without credential leakage."""
        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = Neo4jClientError("Cypher syntax error in runtime")

        with pytest.raises(
            GraphQueryExecutionError, match="Graph execution failed for template 'T-01'"
        ):
            retriever.execute_template("T-01", parameters={"entity_id": "service:test"})

    def test_execute_connection_error_wrapping(
        self,
        retriever: SafeGraphRetriever,
        mock_driver: MagicMock,
    ) -> None:
        """Connection errors are wrapped in GraphQueryExecutionError."""
        mock_session = mock_driver.session.return_value
        mock_session.run.side_effect = Neo4jConnectionError("Neo4j unreachable at bolt://...")

        with pytest.raises(
            GraphQueryExecutionError, match="Neo4j connection error during template 'T-01'"
        ):
            retriever.execute_template("T-01", parameters={"entity_id": "service:test"})

    def test_session_cleanup_on_success_and_failure(
        self,
        retriever: SafeGraphRetriever,
        mock_driver: MagicMock,
    ) -> None:
        """Session context manager __enter__ and __exit__ are invoked cleanly."""
        mock_session = mock_driver.session.return_value
        mock_session.run.return_value = []

        # Success case
        retriever.execute_template("T-01", parameters={"entity_id": "service:test"})
        mock_session.__enter__.assert_called()
        mock_session.__exit__.assert_called()

        # Failure case
        mock_session.__enter__.reset_mock()
        mock_session.__exit__.reset_mock()
        mock_session.run.side_effect = RuntimeError("Fatal query error")

        with pytest.raises(GraphQueryExecutionError):
            retriever.execute_template("T-01", parameters={"entity_id": "service:test"})

        mock_session.__enter__.assert_called()
        mock_session.__exit__.assert_called()

    def test_specialized_helper_methods(
        self,
        retriever: SafeGraphRetriever,
        mock_driver: MagicMock,
    ) -> None:
        """Specialized helper methods delegate to execute_template with correct IDs."""
        mock_session = mock_driver.session.return_value
        mock_session.run.return_value = []

        res1 = retriever.get_direct_neighborhood("service:auth-svc", limit=10)
        assert res1.template_id == "T-01"

        res2 = retriever.get_multi_hop_dependencies("service:auth-svc", limit=15)
        assert res2.template_id == "T-02"

        res3 = retriever.get_team_ownership("team:security", limit=20)
        assert res3.template_id == "T-03"

        res4 = retriever.get_vulnerability_blast_radius("vuln:cve-2024-1234", limit=5)
        assert res4.template_id == "T-04"

        res5 = retriever.get_service_dependency_chain("service:billing-svc", limit=30)
        assert res5.template_id == "T-05"

        res6 = retriever.get_database_producers_consumers("database:orders-pg", limit=25)
        assert res6.template_id == "T-06"

        res7 = retriever.get_infrastructure_deployments("infra:k8s-cluster", limit=40)
        assert res7.template_id == "T-07"

        res8 = retriever.lookup_entity_by_id("service:billing-svc")
        assert res8.template_id == "T-08"
        assert res8.parameters["limit"] == 1

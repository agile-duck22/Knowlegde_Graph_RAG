r"""Safe Graph Retriever executing approved parameterized Cypher templates (TASK-302).

Design Constraints & Invariants (see docs/SECURITY.md, docs/RETRIEVAL_STRATEGY.md, docs/DATA_MODEL.md):
------------------------------------------------------------------------------------------------------
* **Zero dynamic Cypher**: All queries are strictly dispatched from the audited `CypherTemplateCatalog`.
  No dynamic Cypher generation, raw string interpolation, or ad-hoc queries are permitted.
* **Strict Parameter Validation**: Validates all incoming parameters against template specifications before
  touching the database driver. Missing or blank required parameters raise descriptive errors.
* **Bounded Traversals**: Automatically enforces query limits and depth constraints ($D \le 3$) defined
  by the template catalog.
* **Structured & Typed Results**: Returns `GraphQueryResult` containing structured record mappings,
  distinct canonical entity IDs, and deduplicated provenance `chunk_id`s ready for downstream
  vector hydration (Phase 4).
* **Safe Resource Lifecycle**: Manages Neo4j driver sessions with robust try/finally cleanup,
  re-raising sanitized `GraphRetrieverError` without exposing credentials or internal connection strings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.db.neo4j import Neo4jClient, Neo4jClientError, Neo4jConnectionError
from src.retrieval.cypher_templates import (
    CypherTemplateCatalog,
    CypherTemplateSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GraphRetrieverError(Exception):
    """Base exception for all Graph Retriever operations."""


class InvalidTemplateError(GraphRetrieverError):
    """Raised when an unrecognized Cypher template ID is requested."""


class TemplateParameterError(GraphRetrieverError):
    """Raised when template parameter validation fails."""


class GraphQueryExecutionError(GraphRetrieverError):
    """Raised when a Cypher query fails during execution against Neo4j."""


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphQueryResult:
    """Structured result from executing an approved Cypher query template.

    Attributes
    ----------
    template_id:
        Identifier of the executed template (e.g. 'T-01').
    template_name:
        Human-readable name of the template.
    parameters:
        Sanitized and validated parameters passed to the Cypher query.
    records:
        List of raw result records returned by Neo4j as dictionaries.
    record_count:
        Total number of records returned.
    source_chunk_ids:
        Deduplicated list of all provenance chunk IDs extracted from query results.
    entity_ids:
        Deduplicated list of all canonical entity IDs referenced in query results.
    metadata:
        Execution metadata (e.g., execution time or extra context).
    """

    template_id: str
    template_name: str
    parameters: dict[str, Any]
    records: list[dict[str, Any]]
    record_count: int
    source_chunk_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Return True if no records were returned by the graph query."""
        return self.record_count == 0


# ---------------------------------------------------------------------------
# Safe Graph Retriever Implementation
# ---------------------------------------------------------------------------


class SafeGraphRetriever:
    """Executes audited parameterized Cypher query templates against Neo4j."""

    def __init__(
        self,
        client: Neo4jClient,
        catalog: type[CypherTemplateCatalog] = CypherTemplateCatalog,
    ) -> None:
        """Initialize SafeGraphRetriever with a Neo4jClient and template catalog.

        Parameters
        ----------
        client:
            Active Neo4jClient instance managing the database driver.
        catalog:
            The template catalog registry (defaults to CypherTemplateCatalog).
        """
        self.client = client
        self.catalog = catalog

    def execute_template(
        self,
        template_id: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> GraphQueryResult:
        """Execute an approved Cypher query template with validated parameters.

        Parameters
        ----------
        template_id:
            The catalog identifier of the template to run (e.g., 'T-01', 'T-02').
        parameters:
            Dictionary of runtime parameter values required by the template.
        limit:
            Optional override for the query result limit.

        Returns
        -------
        GraphQueryResult
            Structured query results, extracted provenance chunk IDs, and entity IDs.

        Raises
        ------
        InvalidTemplateError:
            If template_id is not registered in the approved catalog.
        TemplateParameterError:
            If required parameters are missing, empty, or have invalid types.
        GraphQueryExecutionError:
            If Neo4j query execution fails.
        """
        # 1. Retrieve template specification
        try:
            spec: CypherTemplateSpec = self.catalog.get_template(template_id)
        except KeyError as exc:
            raise InvalidTemplateError(
                f"Invalid or unapproved Cypher template '{template_id}'. "
                f"Available templates: {self.catalog.list_template_ids()}"
            ) from exc

        # 2. Prepare and validate parameters
        raw_params: dict[str, Any] = dict(parameters or {})
        if limit is not None:
            raw_params["limit"] = limit

        try:
            validated_params = spec.validate_and_build_parameters(raw_params)
        except ValueError as exc:
            raise TemplateParameterError(
                f"Parameter validation failed for template '{spec.template_id}' ({spec.name}): {exc}"
            ) from exc

        logger.debug(
            "Executing Cypher template %s (%s) with params: %s",
            spec.template_id,
            spec.name,
            validated_params,
        )

        # 3. Execute Cypher template in a read transaction
        records: list[dict[str, Any]] = []
        try:
            with self.client.driver.session() as session:
                result = session.run(spec.cypher, validated_params)
                for record in result:
                    records.append(record.data())
        except Neo4jConnectionError as exc:
            logger.error("Neo4j connection error executing template %s: %s", spec.template_id, exc)
            raise GraphQueryExecutionError(
                f"Neo4j connection error during template '{spec.template_id}' execution"
            ) from exc
        except Neo4jClientError as exc:
            logger.error("Neo4j client error executing template %s: %s", spec.template_id, exc)
            raise GraphQueryExecutionError(
                f"Graph execution failed for template '{spec.template_id}': {exc}"
            ) from exc
        except Exception as exc:
            logger.error(
                "Unexpected error executing Cypher template %s: %s",
                spec.template_id,
                exc,
                exc_info=True,
            )
            raise GraphQueryExecutionError(
                f"Unexpected database failure executing template '{spec.template_id}': {exc}"
            ) from exc

        # 4. Extract distinct provenance chunk IDs and canonical entity IDs
        source_chunk_ids = self._extract_chunk_ids(records)
        entity_ids = self._extract_entity_ids(records, validated_params)

        logger.info(
            "Executed template %s (%s): %d records, %d provenance chunk IDs, %d entity IDs",
            spec.template_id,
            spec.name,
            len(records),
            len(source_chunk_ids),
            len(entity_ids),
        )

        return GraphQueryResult(
            template_id=spec.template_id,
            template_name=spec.name,
            parameters=validated_params,
            records=records,
            record_count=len(records),
            source_chunk_ids=source_chunk_ids,
            entity_ids=entity_ids,
        )

    # ------------------------------------------------------------------
    # Specialized Helper Methods for Core Templates
    # ------------------------------------------------------------------

    def get_direct_neighborhood(
        self,
        entity_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Fetch 1-hop connected entities, edge descriptions, and supporting chunk IDs (T-01)."""
        return self.execute_template(
            "T-01",
            parameters={"entity_id": entity_id},
            limit=limit,
        )

    def get_multi_hop_dependencies(
        self,
        entity_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Trace directional DEPENDS_ON and CALLS_API paths (1-3 hops) from entity_id (T-02)."""
        return self.execute_template(
            "T-02",
            parameters={"entity_id": entity_id},
            limit=limit,
        )

    def get_team_ownership(
        self,
        team_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Find all services owned by a team, their exposed APIs, and active vulnerabilities (T-03)."""
        return self.execute_template(
            "T-03",
            parameters={"team_id": team_id},
            limit=limit,
        )

    def get_vulnerability_blast_radius(
        self,
        vuln_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Identify all services directly or transitively impacted by a vulnerability (T-04)."""
        return self.execute_template(
            "T-04",
            parameters={"vuln_id": vuln_id},
            limit=limit,
        )

    def get_service_dependency_chain(
        self,
        service_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Retrieve strict downstream service dependency chain up to 3 hops (T-05)."""
        return self.execute_template(
            "T-05",
            parameters={"service_id": service_id},
            limit=limit,
        )

    def get_database_producers_consumers(
        self,
        database_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Find all services reading from or writing to a database (T-06)."""
        return self.execute_template(
            "T-06",
            parameters={"database_id": database_id},
            limit=limit,
        )

    def get_infrastructure_deployments(
        self,
        infra_id: str,
        *,
        limit: int = 50,
    ) -> GraphQueryResult:
        """Find components deployed on or producing events to an infrastructure resource (T-07)."""
        return self.execute_template(
            "T-07",
            parameters={"infra_id": infra_id},
            limit=limit,
        )

    def lookup_entity_by_id(
        self,
        entity_id: str,
    ) -> GraphQueryResult:
        """Fetch exact node properties and metadata for a canonical entity ID (T-08)."""
        return self.execute_template(
            "T-08",
            parameters={"entity_id": entity_id},
            limit=1,
        )

    # ------------------------------------------------------------------
    # Provenance and Entity Extraction Helpers
    # ------------------------------------------------------------------

    def _extract_chunk_ids(self, records: Sequence[dict[str, Any]]) -> list[str]:
        """Recursively scan result records for source_chunk_ids and chunk_ids."""
        chunk_ids_set: set[str] = set()

        def _collect(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in (
                        "source_chunk_ids",
                        "chunk_ids",
                        "ownership_chunk_ids",
                        "vuln_chunk_ids",
                    ):
                        if isinstance(v, (list, tuple, set)):
                            for item in v:
                                if item and isinstance(item, str) and item.strip():
                                    chunk_ids_set.add(item.strip())
                        elif isinstance(v, str) and v.strip():
                            chunk_ids_set.add(v.strip())
                    else:
                        _collect(v)
            elif isinstance(obj, (list, tuple, set)):
                for elem in obj:
                    _collect(elem)

        for record in records:
            _collect(record)

        return sorted(chunk_ids_set)

    def _extract_entity_ids(
        self,
        records: Sequence[dict[str, Any]],
        params: dict[str, Any],
    ) -> list[str]:
        """Collect all canonical entity identifiers found in query records and parameters."""
        entity_ids_set: set[str] = set()

        # Add target entity IDs from input parameters if they match canonical patterns
        for k, v in params.items():
            if k.endswith("_id") and isinstance(v, str) and ":" in v:
                entity_ids_set.add(v.strip())

        def _collect_ids(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in (
                        "source_id",
                        "target_id",
                        "entity_id",
                        "service_id",
                        "team_id",
                        "vuln_id",
                        "database_id",
                        "infra_id",
                        "vulnerable_service_id",
                        "downstream_service_id",
                        "rel_start_id",
                        "rel_end_id",
                        "id",
                    ):
                        if isinstance(v, str) and ":" in v:
                            entity_ids_set.add(v.strip())
                    else:
                        _collect_ids(v)
            elif isinstance(obj, (list, tuple, set)):
                for elem in obj:
                    _collect_ids(elem)

        for record in records:
            _collect_ids(record)

        return sorted(entity_ids_set)

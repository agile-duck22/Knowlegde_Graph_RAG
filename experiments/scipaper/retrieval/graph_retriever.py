"""Audited Scientific Graph Retriever executing parameterized templates (Phase E).

Strict Constraints:
1. Only connects to verified isolated Neo4j (bolt://localhost:7688).
2. Uses only approved templates from ScientificCypherTemplateCatalog.
3. Restricts labels and relationship types to explicit allowlists.
4. Returns structured ScientificGraphQueryResult with provenance chunk IDs.
5. Returns clear empty result when no match found (never infers missing relationships).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from neo4j import Driver, GraphDatabase

from experiments.scipaper.retrieval.cypher_templates import (
    ScientificCypherTemplateCatalog,
    ScientificCypherTemplateSpec,
)
from experiments.scipaper.schemas.neo4j_writer import (
    TargetIsolationError,
)
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
)

logger = logging.getLogger(__name__)

# Strict domain allowlists
ALLOWLISTED_LABELS = frozenset(["Entity", "Paper", "Author", "Method"])
ALLOWLISTED_REL_TYPES = frozenset(["AUTHORED", "USES_METHOD"])


class ScientificGraphRetrieverError(Exception):
    """Base exception for scientific graph retriever."""


class ScientificInvalidTemplateError(ScientificGraphRetrieverError):
    """Raised when an unapproved template is requested."""


@dataclass(frozen=True)
class ScientificGraphQueryResult:
    """Structured result from executing a scientific Cypher query template."""

    template_id: str
    template_name: str
    parameters: dict[str, Any]
    records: list[dict[str, Any]]
    record_count: int
    provenance_chunk_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.record_count == 0


class ScientificGraphRetriever:
    """Manages read-only execution of approved scientific Cypher templates."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        catalog: type[ScientificCypherTemplateCatalog] = ScientificCypherTemplateCatalog,
    ) -> None:
        self.uri = uri
        self.user = user
        self.catalog = catalog

        # Guard targets
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        if scheme not in ("bolt", "neo4j", "bolt+s", "neo4j+s"):
            raise TargetIsolationError(f"Invalid Neo4j scheme '{scheme}'")

        hostname = (parsed.hostname or "").lower()
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            raise TargetIsolationError(f"Non-local Neo4j host '{hostname}'")

        port = parsed.port
        if port in ENTERPRISE_PORTS or port == 7687 or port != REQUIRED_NEO4J_BOLT_PORT:
            raise TargetIsolationError(
                f"Forbidden Neo4j port {port}; must be {REQUIRED_NEO4J_BOLT_PORT}"
            )

        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def execute_template(
        self,
        template_id: str,
        parameters: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> ScientificGraphQueryResult:
        """Execute an approved template with validated parameters."""
        spec: ScientificCypherTemplateSpec
        try:
            spec = self.catalog.get(template_id)
        except KeyError as exc:
            raise ScientificInvalidTemplateError(f"Unknown template: {template_id}") from exc

        params = dict(parameters or {})
        effective_limit = limit if limit is not None else spec.default_limit
        params["limit"] = effective_limit

        records: list[dict[str, Any]] = []
        provenance_cids: set[str] = set()
        entities: set[str] = set()

        try:
            with self._driver.session() as session:
                result = session.run(spec.cypher, **params)
                for record in result:
                    rec_dict = record.data()
                    records.append(rec_dict)

                    # Extract chunk IDs
                    for key in ("source_chunk_ids", "r1_chunk_ids", "r2_chunk_ids"):
                        val = rec_dict.get(key)
                        if isinstance(val, list):
                            for cid in val:
                                if cid:
                                    provenance_cids.add(str(cid))
                        elif isinstance(val, str) and val:
                            provenance_cids.add(val)

                    # Extract entity IDs
                    for key in (
                        "paper_id",
                        "author_id",
                        "method_id",
                        "source_author_id",
                        "coauthor_id",
                        "neighbor_id",
                        "entity_id",
                    ):
                        eid = rec_dict.get(key)
                        if eid:
                            entities.add(str(eid))

        except Exception as exc:
            logger.error("Error executing scientific template %s: %s", template_id, exc)
            raise ScientificGraphRetrieverError(f"Failed to execute {template_id}: {exc}") from exc

        return ScientificGraphQueryResult(
            template_id=spec.template_id,
            template_name=spec.name,
            parameters=params,
            records=records,
            record_count=len(records),
            provenance_chunk_ids=sorted(provenance_cids),
            entity_ids=sorted(entities),
        )

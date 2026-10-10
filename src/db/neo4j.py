"""Neo4j driver lifecycle, schema initialisation, and idempotent batch writes.

Design constraints (see docs/SECURITY.md and docs/ARCHITECTURE.md)
-------------------------------------------------------------------
* **Zero dynamic Cypher**: every Cypher string in this module is a module-level
  constant.  No f-strings, ``%`` formatting, or ``str.format`` calls touch query
  text.  All untrusted values are passed via the parameter map.
* **Ontology-locked templates**: the 7 node-label templates and 10 relationship
  templates are derived directly from ``docs/ONTOLOGY.md``.  Unknown types are
  rejected with ``UnsupportedEntityTypeError`` / ``UnsupportedRelationshipTypeError``
  *before* a Cypher statement is executed.
* **Idempotency**: nodes use ``MERGE … ON CREATE SET … ON MATCH SET …``.
  Relationships use ``MERGE`` on the composite
  ``(relation_type, provenance_chunk_id)`` key so that re-ingesting the same
  chunk is a no-op.
* **Bounded batches**: ``write_entities`` and ``write_relationships`` accept an
  explicit ``batch_size`` parameter (default 100).  Each batch runs in its own
  explicit transaction; a failure in one batch does not silently corrupt earlier
  batches.
* **Resource lifecycle**: ``Neo4jClient`` is a context manager.  Users must call
  ``close()`` or use the ``with`` statement; the driver is never left open.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC
from typing import Any

from neo4j import Driver as Neo4jDriver
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import Settings, get_settings
from src.extraction.schemas import (
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class Neo4jClientError(Exception):
    """Base class for all Neo4j client errors."""


class Neo4jConnectionError(Neo4jClientError):
    """Raised when the driver cannot reach Neo4j."""


class UnsupportedEntityTypeError(Neo4jClientError):
    """Raised when an entity type is not in the ontology allowlist."""


class UnsupportedRelationshipTypeError(Neo4jClientError):
    """Raised when a relationship type is not in the ontology allowlist."""


class BatchWriteError(Neo4jClientError):
    """Raised when a transactional batch write fails and is rolled back."""


# ---------------------------------------------------------------------------
# Ontology-locked allowlists (mirrors docs/ONTOLOGY.md)
# ---------------------------------------------------------------------------

#: The 7 entity labels from ONTOLOGY.md §1.
_ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset(
    {"Service", "Team", "API", "Database", "Repository", "Vulnerability", "Infrastructure"}
)

#: The 10 directed relationship types from ONTOLOGY.md §2.
_ALLOWED_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {
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
)

# ---------------------------------------------------------------------------
# Schema DDL — uniqueness constraint and fulltext index
# ---------------------------------------------------------------------------

#: Uniqueness constraint on canonical entity ID (DATA_MODEL.md §3).
_CONSTRAINT_UNIQUE_ENTITY_ID = (
    "CREATE CONSTRAINT unique_entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
)

#: Fulltext search index on name and description (DATA_MODEL.md §3).
_INDEX_ENTITY_FULLTEXT = (
    "CREATE FULLTEXT INDEX entity_text_index IF NOT EXISTS "
    "FOR (n:Entity) ON EACH [n.name, n.description]"
)

# ---------------------------------------------------------------------------
# Node upsert templates — one per entity type (label is a literal)
# ---------------------------------------------------------------------------
#
# Secondary label (e.g. :Service, :Team, …) cannot be passed as a parameter in
# Cypher — labels are identifiers, not values.  We therefore maintain one fixed
# template per entity type so the label is always a literal in the query text.
# No user-supplied or LLM-generated string ever enters the query text.

_UPSERT_ENTITY_SERVICE = """
MERGE (e:Entity:Service {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

_UPSERT_ENTITY_TEAM = """
MERGE (e:Entity:Team {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

_UPSERT_ENTITY_API = """
MERGE (e:Entity:API {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

_UPSERT_ENTITY_DATABASE = """
MERGE (e:Entity:Database {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

_UPSERT_ENTITY_REPOSITORY = """
MERGE (e:Entity:Repository {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

_UPSERT_ENTITY_VULNERABILITY = """
MERGE (e:Entity:Vulnerability {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

_UPSERT_ENTITY_INFRASTRUCTURE = """
MERGE (e:Entity:Infrastructure {id: $id})
ON CREATE SET
    e.name        = $name,
    e.type        = $entity_type,
    e.description = $description,
    e.aliases     = $aliases,
    e.created_at  = datetime($created_at),
    e.updated_at  = datetime($updated_at)
ON MATCH SET
    e.name        = $name,
    e.description = $description,
    e.aliases     = $aliases,
    e.updated_at  = datetime($updated_at)
RETURN e.id AS id
"""

#: Dispatch map: entity_type -> fixed Cypher template string.
_ENTITY_UPSERT_TEMPLATES: dict[str, str] = {
    "Service": _UPSERT_ENTITY_SERVICE,
    "Team": _UPSERT_ENTITY_TEAM,
    "API": _UPSERT_ENTITY_API,
    "Database": _UPSERT_ENTITY_DATABASE,
    "Repository": _UPSERT_ENTITY_REPOSITORY,
    "Vulnerability": _UPSERT_ENTITY_VULNERABILITY,
    "Infrastructure": _UPSERT_ENTITY_INFRASTRUCTURE,
}

# ---------------------------------------------------------------------------
# Relationship upsert templates — one fixed UNWIND template per rel type
# ---------------------------------------------------------------------------
#
# Relationship types ARE identifiers in Cypher (not values), so they cannot be
# parameterised.  We pre-render one constant per type at module load time.
# The UNWIND pattern lets us submit an entire batch in one round-trip.
#
# MERGE key: (source.id, target.id, r.relation_type, r.provenance_chunk_id)
# This means the same relationship extracted from two *different* chunks
# produces two distinct edges (each with its own provenance), while
# re-ingesting the same chunk is a no-op.

_REL_BATCH_OWNED_BY = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:OWNED_BY {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_DEPENDS_ON = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:DEPENDS_ON {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_EXPOSES_API = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:EXPOSES_API {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_CALLS_API = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:CALLS_API {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_READS_FROM = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:READS_FROM {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_WRITES_TO = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:WRITES_TO {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_STORES_CODE = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:STORES_CODE {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_AFFECTED_BY = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:AFFECTED_BY {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_DEPLOYED_ON = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:DEPLOYED_ON {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_PRODUCES_EVENTS = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:PRODUCES_EVENTS {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

#: Dispatch map: relation_type -> fixed UNWIND Cypher template.
_RELATIONSHIP_BATCH_TEMPLATES: dict[str, str] = {
    "OWNED_BY": _REL_BATCH_OWNED_BY,
    "DEPENDS_ON": _REL_BATCH_DEPENDS_ON,
    "EXPOSES_API": _REL_BATCH_EXPOSES_API,
    "CALLS_API": _REL_BATCH_CALLS_API,
    "READS_FROM": _REL_BATCH_READS_FROM,
    "WRITES_TO": _REL_BATCH_WRITES_TO,
    "STORES_CODE": _REL_BATCH_STORES_CODE,
    "AFFECTED_BY": _REL_BATCH_AFFECTED_BY,
    "DEPLOYED_ON": _REL_BATCH_DEPLOYED_ON,
    "PRODUCES_EVENTS": _REL_BATCH_PRODUCES_EVENTS,
}

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

_PING_QUERY = "RETURN 1 AS ping"

# ---------------------------------------------------------------------------
# Helper: current UTC datetime string for Neo4j datetime()
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string accepted by Neo4j ``datetime()``."""
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------


class Neo4jClient:
    """Managed Neo4j driver providing schema initialisation and batch writes.

    Usage::

        with Neo4jClient(settings) as client:
            client.initialise_schema()
            client.write_entities(entities)
            client.write_relationships(relationships)

    All Cypher in this class uses module-level constant templates; no string
    formatting is ever applied to user-supplied or LLM-generated data.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the client from *settings* (defaults to ``get_settings()``)."""
        cfg = settings or get_settings()
        password = (
            cfg.neo4j_password.get_secret_value()
            if cfg.neo4j_password is not None
            else ""
        )
        self._driver: Neo4jDriver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, password),
        )
        logger.debug("Neo4j driver created for URI %s", cfg.neo4j_uri)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying Neo4j driver and release all connections."""
        self._driver.close()
        logger.debug("Neo4j driver closed")

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------

    def verify_connectivity(self) -> None:
        """Ping Neo4j to confirm the driver can reach the server.

        Raises
        ------
        Neo4jConnectionError
            If the server is unreachable or the driver is misconfigured.
        """
        try:
            with self._driver.session() as session:
                result = session.run(_PING_QUERY)
                record = result.single()
                if record is None or record["ping"] != 1:
                    raise Neo4jConnectionError("Connectivity ping returned unexpected result")
            logger.info("Neo4j connectivity verified")
        except ServiceUnavailable as exc:
            raise Neo4jConnectionError(
                f"Cannot connect to Neo4j: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Schema DDL
    # ------------------------------------------------------------------

    def initialise_schema(self) -> None:
        """Create uniqueness constraints and the fulltext index if they do not exist.

        Idempotent: safe to call on every startup because both statements use
        ``IF NOT EXISTS``.
        """
        with self._driver.session() as session:
            logger.info("Applying Neo4j schema: uniqueness constraint")
            session.run(_CONSTRAINT_UNIQUE_ENTITY_ID)
            logger.info("Applying Neo4j schema: fulltext index")
            session.run(_INDEX_ENTITY_FULLTEXT)
        logger.info("Neo4j schema initialisation complete")

    # ------------------------------------------------------------------
    # Entity batch write
    # ------------------------------------------------------------------

    def write_entities(
        self,
        entities: Sequence[ExtractedEntity],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upsert *entities* into Neo4j in bounded transactional batches.

        Each entity maps to the fixed Cypher template for its type (dispatched
        via ``_ENTITY_UPSERT_TEMPLATES``).  Unknown entity types raise
        ``UnsupportedEntityTypeError`` *before* any write is attempted.

        Parameters
        ----------
        entities:
            Sequence of validated ``ExtractedEntity`` objects.
        batch_size:
            Maximum number of entities per transaction.

        Returns
        -------
        int
            Total number of entities successfully written.

        Raises
        ------
        UnsupportedEntityTypeError
            If any entity carries a type not in the ontology allowlist.
        BatchWriteError
            If a transactional batch fails; the transaction is rolled back.
        """
        if not entities:
            return 0

        # Validate all types upfront — fail fast before touching the DB.
        for entity in entities:
            if entity.entity_type not in _ALLOWED_ENTITY_TYPES:
                raise UnsupportedEntityTypeError(
                    f"Entity type '{entity.entity_type}' is not in the ontology allowlist. "
                    f"Allowed types: {sorted(_ALLOWED_ENTITY_TYPES)}"
                )

        now = _utc_now_iso()
        total_written = 0

        for batch_start in range(0, len(entities), batch_size):
            batch = entities[batch_start : batch_start + batch_size]
            try:
                with self._driver.session() as session:
                    with session.begin_transaction() as tx:
                        for entity in batch:
                            template = _ENTITY_UPSERT_TEMPLATES[entity.entity_type]
                            params: dict[str, Any] = {
                                "id": entity.canonical_id,
                                "name": entity.name,
                                "entity_type": entity.entity_type,
                                "description": entity.description,
                                "aliases": list(entity.aliases),
                                "created_at": now,
                                "updated_at": now,
                            }
                            tx.run(template, params)
                        tx.commit()
                total_written += len(batch)
                logger.debug(
                    "Wrote entity batch [%d:%d] (%d entities)",
                    batch_start,
                    batch_start + len(batch),
                    len(batch),
                )
            except Exception as exc:
                raise BatchWriteError(
                    f"Entity batch [{batch_start}:{batch_start + len(batch)}] "
                    f"failed and was rolled back: {exc}"
                ) from exc

        logger.info("write_entities complete: %d entities written", total_written)
        return total_written

    # ------------------------------------------------------------------
    # Relationship batch write
    # ------------------------------------------------------------------

    def write_relationships(
        self,
        relationships: Sequence[ExtractedRelationship],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upsert *relationships* into Neo4j in bounded transactional batches.

        Uses the pre-rendered UNWIND batch template for the relationship type.
        Unknown relationship types raise ``UnsupportedRelationshipTypeError``
        before any write is attempted.

        Nodes referenced by the relationship **must already exist** in the graph
        (call :meth:`write_entities` first).  If a ``MATCH`` clause finds no
        node, that row is silently skipped by Neo4j — the batch does not fail,
        but ``merged`` will be lower than ``len(batch)``.

        Parameters
        ----------
        relationships:
            Sequence of validated ``ExtractedRelationship`` objects.
        batch_size:
            Maximum number of relationships per transaction.

        Returns
        -------
        int
            Total number of relationship rows submitted (merged or created).

        Raises
        ------
        UnsupportedRelationshipTypeError
            If any relationship carries a type not in the ontology allowlist.
        BatchWriteError
            If a transactional batch fails; the transaction is rolled back.
        """
        if not relationships:
            return 0

        # Validate all types upfront.
        for rel in relationships:
            if rel.relation_type not in _ALLOWED_RELATIONSHIP_TYPES:
                raise UnsupportedRelationshipTypeError(
                    f"Relationship type '{rel.relation_type}' is not in the ontology allowlist. "
                    f"Allowed types: {sorted(_ALLOWED_RELATIONSHIP_TYPES)}"
                )

        now = _utc_now_iso()
        total_written = 0

        for batch_start in range(0, len(relationships), batch_size):
            batch = relationships[batch_start : batch_start + batch_size]

            # Group the batch by relation_type: one UNWIND call per type.
            grouped: dict[str, list[dict[str, Any]]] = {}
            for rel in batch:
                row: dict[str, Any] = {
                    "source_id": rel.source_canonical_id,
                    "target_id": rel.target_canonical_id,
                    "relation_type": rel.relation_type,
                    "description": rel.description,
                    "confidence": rel.confidence,
                    "source_chunk_ids": list(rel.source_chunk_ids),
                    "provenance_chunk_id": rel.provenance.chunk_id,
                    "created_at": now,
                    "updated_at": now,
                }
                grouped.setdefault(rel.relation_type, []).append(row)

            try:
                with self._driver.session() as session:
                    with session.begin_transaction() as tx:
                        for rel_type, rows in grouped.items():
                            template = _RELATIONSHIP_BATCH_TEMPLATES[rel_type]
                            tx.run(template, {"rows": rows})
                        tx.commit()
                total_written += len(batch)
                logger.debug(
                    "Wrote relationship batch [%d:%d] (%d relationships)",
                    batch_start,
                    batch_start + len(batch),
                    len(batch),
                )
            except Exception as exc:
                raise BatchWriteError(
                    f"Relationship batch [{batch_start}:{batch_start + len(batch)}] "
                    f"failed and was rolled back: {exc}"
                ) from exc

        logger.info("write_relationships complete: %d relationships written", total_written)
        return total_written

    # ------------------------------------------------------------------
    # Convenience: write a full ChunkExtractionResult
    # ------------------------------------------------------------------

    def write_chunk_result(
        self,
        result: ChunkExtractionResult,
        *,
        entity_batch_size: int = 100,
        relationship_batch_size: int = 100,
    ) -> tuple[int, int]:
        """Write all entities then all relationships from a single chunk result.

        Parameters
        ----------
        result:
            Fully validated ``ChunkExtractionResult`` from TASK-103/104.
        entity_batch_size:
            Batch size forwarded to :meth:`write_entities`.
        relationship_batch_size:
            Batch size forwarded to :meth:`write_relationships`.

        Returns
        -------
        tuple[int, int]
            ``(entities_written, relationships_written)``.
        """
        entities_written = self.write_entities(
            result.entities, batch_size=entity_batch_size
        )
        relationships_written = self.write_relationships(
            result.relationships, batch_size=relationship_batch_size
        )
        return entities_written, relationships_written

    # ------------------------------------------------------------------
    # Expose the raw driver for advanced/integration use-cases
    # ------------------------------------------------------------------

    @property
    def driver(self) -> Neo4jDriver:
        """Return the underlying ``neo4j.Driver`` instance (read-only)."""
        return self._driver

"""Isolated Neo4j Graph Writer for the Scientific-Paper Experiment.

Strict Constraints:
- Connects ONLY to the isolated Neo4j endpoint (host port 7688).
- Rejects default or enterprise Neo4j ports (7687) fail-closed.
- Writes scientific entity nodes with dual labels: ``(:Entity:ScientificType)`` (e.g. ``(:Entity:Paper)``).
- Reuses the existing generic uniqueness constraint on ``(e:Entity {id})``.
- Uses strictly constant, parameterized Cypher templates with zero string formatting on data.
- Bounded transactional batch execution with rollback safety.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC
from typing import Any
from urllib.parse import urlparse

from neo4j import Driver as Neo4jDriver
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
    ScientificExtractedEntity,
    ScientificExtractedRelationship,
)
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
)

logger = logging.getLogger(__name__)


class ScientificNeo4jError(Exception):
    """Base exception for scientific Neo4j graph operations."""


class TargetIsolationError(ScientificNeo4jError):
    """Raised when the Neo4j endpoint targets enterprise or unverified ports."""


class UnsupportedScientificTypeError(ScientificNeo4jError):
    """Raised when an unknown entity or relationship type is submitted."""


def _utc_now_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat()


# Pre-rendered parameterized Cypher templates for the 8 scientific entity types
# Node always has primary label :Entity and secondary scientific label :<Type>
_UPSERT_ENTITY_PAPER = """
MERGE (e:Entity:Paper {id: $id})
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

_UPSERT_ENTITY_AUTHOR = """
MERGE (e:Entity:Author {id: $id})
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

_UPSERT_ENTITY_MODEL = """
MERGE (e:Entity:Model {id: $id})
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

_UPSERT_ENTITY_DATASET = """
MERGE (e:Entity:Dataset {id: $id})
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

_UPSERT_ENTITY_METRIC = """
MERGE (e:Entity:Metric {id: $id})
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

_UPSERT_ENTITY_TASK = """
MERGE (e:Entity:Task {id: $id})
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

_UPSERT_ENTITY_METHOD = """
MERGE (e:Entity:Method {id: $id})
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

_UPSERT_ENTITY_RESEARCH_ORG = """
MERGE (e:Entity:ResearchOrganization {id: $id})
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

SCIENTIFIC_ENTITY_TEMPLATES: dict[str, str] = {
    "Paper": _UPSERT_ENTITY_PAPER,
    "Author": _UPSERT_ENTITY_AUTHOR,
    "Model": _UPSERT_ENTITY_MODEL,
    "Dataset": _UPSERT_ENTITY_DATASET,
    "Metric": _UPSERT_ENTITY_METRIC,
    "Task": _UPSERT_ENTITY_TASK,
    "Method": _UPSERT_ENTITY_METHOD,
    "ResearchOrganization": _UPSERT_ENTITY_RESEARCH_ORG,
}

# Parameterized UNWIND templates for the 7 scientific relationships
_REL_BATCH_AUTHORED = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:AUTHORED {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_PROPOSES = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:PROPOSES {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_EVALUATED_ON = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:EVALUATED_ON {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_REPORTS_METRIC = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:REPORTS_METRIC {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_USES_METHOD = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:USES_METHOD {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_AFFILIATED_WITH = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:AFFILIATED_WITH {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

_REL_BATCH_CITES = """
UNWIND $rows AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
MERGE (source)-[r:CITES {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
ON CREATE SET r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.created_at = datetime(row.created_at), r.updated_at = datetime(row.updated_at)
ON MATCH SET  r.description = row.description, r.confidence = row.confidence, r.source_chunk_ids = row.source_chunk_ids, r.updated_at = datetime(row.updated_at)
RETURN count(r) AS merged
"""

SCIENTIFIC_RELATIONSHIP_TEMPLATES: dict[str, str] = {
    "AUTHORED": _REL_BATCH_AUTHORED,
    "PROPOSES": _REL_BATCH_PROPOSES,
    "EVALUATED_ON": _REL_BATCH_EVALUATED_ON,
    "REPORTS_METRIC": _REL_BATCH_REPORTS_METRIC,
    "USES_METHOD": _REL_BATCH_USES_METHOD,
    "AFFILIATED_WITH": _REL_BATCH_AFFILIATED_WITH,
    "CITES": _REL_BATCH_CITES,
}


class IsolatedScientificNeo4jWriter:
    """Manages transactional, idempotent writes of scientific entities and relationships to isolated Neo4j."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self._password = password

        # Strict target guard
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        if scheme not in ("bolt", "neo4j", "bolt+s", "neo4j+s"):
            raise TargetIsolationError(
                f"Refusing to connect: Invalid Neo4j scheme '{scheme}'. Must be bolt or neo4j."
            )

        hostname = (parsed.hostname or "").lower()
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            raise TargetIsolationError(
                f"Refusing to connect: Non-local host '{hostname}'. Isolated experiments must use loopback (localhost or 127.0.0.1)."
            )

        port = parsed.port
        if port in ENTERPRISE_PORTS or port == 7687:
            raise TargetIsolationError(
                f"Refusing to connect: URI '{uri}' targets enterprise port {port}. Must be {REQUIRED_NEO4J_BOLT_PORT}."
            )
        if port != REQUIRED_NEO4J_BOLT_PORT:
            raise TargetIsolationError(
                f"Refusing to connect: Target port is {port}; expected isolated port {REQUIRED_NEO4J_BOLT_PORT}."
            )

        self._driver: Neo4jDriver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def __enter__(self) -> IsolatedScientificNeo4jWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def verify_connectivity(self) -> None:
        """Ping isolated Neo4j to confirm driver connectivity."""
        try:
            with self._driver.session() as session:
                rec = session.run("RETURN 1 AS ping").single()
                if not rec or rec["ping"] != 1:
                    raise ScientificNeo4jError("Neo4j ping returned unexpected value")
        except ServiceUnavailable as exc:
            raise ScientificNeo4jError(f"Cannot connect to isolated Neo4j: {exc}") from exc

    def write_entities(
        self,
        entities: Sequence[ScientificExtractedEntity],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upsert scientific entities into Neo4j in bounded transactional batches."""
        if not entities:
            return 0

        # Validate types upfront
        for ent in entities:
            if ent.entity_type not in SCIENTIFIC_ENTITY_TEMPLATES:
                raise UnsupportedScientificTypeError(
                    f"Entity type '{ent.entity_type}' is not supported by scientific ontology"
                )

        now = _utc_now_iso()
        total_written = 0

        for batch_start in range(0, len(entities), batch_size):
            batch = entities[batch_start : batch_start + batch_size]
            try:
                with self._driver.session() as session:
                    with session.begin_transaction() as tx:
                        for entity in batch:
                            template = SCIENTIFIC_ENTITY_TEMPLATES[entity.entity_type]
                            params = {
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
            except Exception as exc:
                raise ScientificNeo4jError(
                    f"Entity batch write failed and rolled back: {exc}"
                ) from exc

        return total_written

    def write_relationships(
        self,
        relationships: Sequence[ScientificExtractedRelationship],
        *,
        batch_size: int = 100,
    ) -> int:
        """Upsert scientific relationships into Neo4j in bounded transactional batches."""
        if not relationships:
            return 0

        for rel in relationships:
            if rel.relation_type not in SCIENTIFIC_RELATIONSHIP_TEMPLATES:
                raise UnsupportedScientificTypeError(
                    f"Relationship type '{rel.relation_type}' is not supported"
                )

        now = _utc_now_iso()
        total_written = 0

        for batch_start in range(0, len(relationships), batch_size):
            batch = relationships[batch_start : batch_start + batch_size]

            grouped: dict[str, list[dict[str, Any]]] = {}
            for rel in batch:
                row = {
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
                            template = SCIENTIFIC_RELATIONSHIP_TEMPLATES[rel_type]
                            tx.run(template, {"rows": rows})
                        tx.commit()
                total_written += len(batch)
            except Exception as exc:
                raise ScientificNeo4jError(
                    f"Relationship batch write failed and rolled back: {exc}"
                ) from exc

        return total_written

    def write_chunk_result(
        self,
        result: ScientificChunkExtractionResult,
        *,
        entity_batch_size: int = 100,
        relationship_batch_size: int = 100,
    ) -> tuple[int, int]:
        """Write entities followed by relationships for a single chunk result."""
        e_cnt = self.write_entities(result.entities, batch_size=entity_batch_size)
        r_cnt = self.write_relationships(result.relationships, batch_size=relationship_batch_size)
        return e_cnt, r_cnt

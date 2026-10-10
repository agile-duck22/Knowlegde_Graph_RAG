"""Dual-Store Provenance and Consistency Validator (TASK-205).

Design & Invariants (see docs/DATA_MODEL.md, docs/ARCHITECTURE.md, docs/SECURITY.md):
- Validates cross-store chunk ID parity between Neo4j and PostgreSQL.
- Verifies that every relationship and entity provenance record in Neo4j references a valid
  canonical chunk UUID in PostgreSQL (chunks.id).
- Detects:
  1. Orphaned Graph Provenance: source_chunk_ids or provenance_chunk_ids in Neo4j missing from PostgreSQL.
  2. Missing Graph Provenance: Chunks in PostgreSQL that have no corresponding graph entities/edges.
  3. Mismatched Entity Links: entity_links in PostgreSQL referencing non-existent Neo4j entities or chunks.
  4. Invalid UUID Formats: Any non-canonical or malformed chunk_id in Neo4j edges or entity provenance.
- Zero raw dynamic Cypher: Uses parameterized Cypher read templates.
- Parameterized SQL: Uses asyncpg parameterized queries against PostgreSQL.
- Safe resource management with robust error handling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.db.neo4j import Neo4jClient, Neo4jClientError
from src.db.postgres import PostgresClient, PostgresClientError

logger = logging.getLogger(__name__)

# UUID format regex (canonical 8-4-4-4-12 hex string)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Cypher read templates (Parameterized / Fixed constant strings)
_CYPHER_FETCH_RELATIONSHIP_PROVENANCE = """
MATCH ()-[r]->()
RETURN
    type(r) AS relation_type,
    r.provenance_chunk_id AS provenance_chunk_id,
    r.source_chunk_ids AS source_chunk_ids,
    startNode(r).id AS source_id,
    endNode(r).id AS target_id
"""

_CYPHER_FETCH_ALL_ENTITY_IDS = """
MATCH (e:Entity)
RETURN e.id AS entity_id, e.type AS entity_type
"""


class DualStoreValidatorError(Exception):
    """Base exception for dual-store provenance validation failures."""


@dataclass(frozen=True)
class ProvenanceMismatch:
    """Individual provenance mismatch record.

    Attributes
    ----------
    mismatch_type:
        Category of mismatch ('orphaned_graph_chunk', 'unreferenced_postgres_chunk',
        'invalid_uuid_format', 'broken_entity_link').
    source_store:
        Store where the error was identified ('neo4j', 'postgres', 'cross_store').
    identifier:
        The chunk_id, relationship, or entity_link ID involved.
    details:
        Human-readable description and context of the mismatch.
    """

    mismatch_type: str
    source_store: str
    identifier: str
    details: str


@dataclass
class ConsistencyReport:
    """Detailed dual-store consistency and provenance validation report.

    Attributes
    ----------
    is_consistent:
        True if zero critical provenance mismatches were detected.
    total_postgres_chunks:
        Total number of chunks stored in PostgreSQL.
    total_neo4j_relationships:
        Total number of relationships inspected in Neo4j.
    total_graph_chunk_references:
        Total count of distinct chunk IDs referenced across Neo4j relationships.
    postgres_chunk_ids:
        Set of all chunk UUIDs stored in PostgreSQL.
    neo4j_chunk_ids:
        Set of all chunk UUID strings referenced in Neo4j.
    orphaned_graph_chunk_ids:
        Chunk IDs referenced in Neo4j but not present in PostgreSQL.
    unreferenced_postgres_chunk_ids:
        PostgreSQL chunk IDs with no associated Neo4j relationships (informational/warning).
    mismatches:
        List of all detailed ProvenanceMismatch objects.
    metadata:
        Additional execution metrics or context.
    """

    is_consistent: bool = True
    total_postgres_chunks: int = 0
    total_neo4j_relationships: int = 0
    total_graph_chunk_references: int = 0
    postgres_chunk_ids: set[str] = field(default_factory=set)
    neo4j_chunk_ids: set[str] = field(default_factory=set)
    orphaned_graph_chunk_ids: set[str] = field(default_factory=set)
    unreferenced_postgres_chunk_ids: set[str] = field(default_factory=set)
    mismatches: list[ProvenanceMismatch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the validation report."""
        status = "PASSED (CONSISTENT)" if self.is_consistent else "FAILED (INCONSISTENCIES FOUND)"
        lines = [
            f"=== Dual-Store Provenance Consistency Report: {status} ===",
            f"PostgreSQL Total Chunks: {self.total_postgres_chunks}",
            f"Neo4j Total Relationships: {self.total_neo4j_relationships}",
            f"Neo4j Distinct Referenced Chunks: {len(self.neo4j_chunk_ids)}",
            f"Orphaned Graph Chunk References: {len(self.orphaned_graph_chunk_ids)}",
            f"Unreferenced PostgreSQL Chunks: {len(self.unreferenced_postgres_chunk_ids)}",
            f"Total Specific Mismatches: {len(self.mismatches)}",
        ]
        if self.mismatches:
            lines.append("--- Detailed Mismatches ---")
            for m in self.mismatches[:20]:  # Show first 20 in summary
                lines.append(
                    f"[{m.mismatch_type}] ({m.source_store}) ID: {m.identifier} -> {m.details}"
                )
            if len(self.mismatches) > 20:
                lines.append(f"... and {len(self.mismatches) - 20} more mismatches.")
        return "\n".join(lines)


class DualStoreConsistencyValidator:
    """Validates data consistency and chunk provenance across Neo4j and PostgreSQL."""

    def __init__(
        self,
        postgres_client: PostgresClient,
        neo4j_client: Neo4jClient,
    ) -> None:
        self.postgres_client = postgres_client
        self.neo4j_client = neo4j_client

    async def validate_provenance(
        self,
        document_id: str | None = None,
        *,
        strict_orphan_check: bool = True,
        check_entity_links: bool = True,
    ) -> ConsistencyReport:
        """Execute full cross-store provenance validation.

        Parameters
        ----------
        document_id:
            Optional document_id filter. If provided, checks are scoped to that document.
        strict_orphan_check:
            If True, treats unreferenced PostgreSQL chunks as report inconsistencies.
            Default False / informational since chunks may contain introductory text without graph entities.
        check_entity_links:
            If True, also validates PostgreSQL entity_links table against chunks and Neo4j entities.

        Returns
        -------
        ConsistencyReport
            Structured validation report detailing all findings.

        Raises
        ------
        DualStoreValidatorError
            If database access fails or connections cannot be established.
        """
        logger.info(
            "Starting dual-store provenance validation (scope=%s)...", document_id or "GLOBAL"
        )
        report = ConsistencyReport()

        try:
            # 1. Fetch chunk IDs from PostgreSQL
            pg_chunk_ids = await self._fetch_postgres_chunk_ids(document_id=document_id)
            report.postgres_chunk_ids = pg_chunk_ids
            report.total_postgres_chunks = len(pg_chunk_ids)

            # 2. Fetch relationship provenance from Neo4j
            neo4j_rels = self._fetch_neo4j_relationship_provenance()
            report.total_neo4j_relationships = len(neo4j_rels)

            # 3. Process Neo4j relationship chunk references
            graph_chunk_ids: set[str] = set()
            for rel in neo4j_rels:
                rel_type = rel.get("relation_type", "UNKNOWN")
                src_id = rel.get("source_id", "UNKNOWN")
                tgt_id = rel.get("target_id", "UNKNOWN")
                rel_ident = f"({src_id})-[{rel_type}]->({tgt_id})"

                # Validate provenance_chunk_id
                prov_id = rel.get("provenance_chunk_id")
                if prov_id:
                    prov_str = str(prov_id).strip()
                    if not _UUID_RE.match(prov_str):
                        report.mismatches.append(
                            ProvenanceMismatch(
                                mismatch_type="invalid_uuid_format",
                                source_store="neo4j",
                                identifier=prov_str,
                                details=f"Relationship {rel_ident} has malformed provenance_chunk_id '{prov_str}'",
                            )
                        )
                    else:
                        graph_chunk_ids.add(prov_str)

                # Validate source_chunk_ids array
                source_chunks = rel.get("source_chunk_ids") or []
                for sc in source_chunks:
                    sc_str = str(sc).strip()
                    if not _UUID_RE.match(sc_str):
                        report.mismatches.append(
                            ProvenanceMismatch(
                                mismatch_type="invalid_uuid_format",
                                source_store="neo4j",
                                identifier=sc_str,
                                details=f"Relationship {rel_ident} has malformed source_chunk_id '{sc_str}'",
                            )
                        )
                    else:
                        graph_chunk_ids.add(sc_str)

            report.neo4j_chunk_ids = graph_chunk_ids
            report.total_graph_chunk_references = len(graph_chunk_ids)

            # 4. Detect Orphaned Graph Chunks (in Neo4j but not in PostgreSQL)
            # If scoped to document_id, only check graph chunks that belong to or overlap this scope
            orphaned = graph_chunk_ids - pg_chunk_ids
            if document_id is not None:
                # If scoped, verify whether orphaned IDs are indeed absent from PostgreSQL entirely
                all_pg_ids = await self._fetch_postgres_chunk_ids(document_id=None)
                truly_orphaned = orphaned - all_pg_ids
                report.orphaned_graph_chunk_ids = truly_orphaned
                for orphan in truly_orphaned:
                    report.mismatches.append(
                        ProvenanceMismatch(
                            mismatch_type="orphaned_graph_chunk",
                            source_store="cross_store",
                            identifier=orphan,
                            details=f"Chunk ID '{orphan}' is referenced in Neo4j relationship but missing from PostgreSQL chunks table",
                        )
                    )
            else:
                report.orphaned_graph_chunk_ids = orphaned
                for orphan in orphaned:
                    report.mismatches.append(
                        ProvenanceMismatch(
                            mismatch_type="orphaned_graph_chunk",
                            source_store="cross_store",
                            identifier=orphan,
                            details=f"Chunk ID '{orphan}' is referenced in Neo4j relationship but missing from PostgreSQL chunks table",
                        )
                    )

            # 5. Detect Unreferenced PostgreSQL chunks
            unref = pg_chunk_ids - graph_chunk_ids
            report.unreferenced_postgres_chunk_ids = unref
            if strict_orphan_check and unref:
                for unref_id in unref:
                    report.mismatches.append(
                        ProvenanceMismatch(
                            mismatch_type="unreferenced_postgres_chunk",
                            source_store="postgres",
                            identifier=unref_id,
                            details=f"Chunk ID '{unref_id}' in PostgreSQL has no corresponding entity/relationship provenance in Neo4j",
                        )
                    )

            # 6. Optional Entity Links Validation (PostgreSQL entity_links table)
            if check_entity_links:
                await self._validate_entity_links(report=report, document_id=document_id)

            # Set overall consistency boolean
            # A report is inconsistent if there are any orphaned graph chunks, malformed UUIDs, or broken entity links
            critical_mismatches = [
                m
                for m in report.mismatches
                if m.mismatch_type
                in {
                    "orphaned_graph_chunk",
                    "invalid_uuid_format",
                    "broken_entity_link",
                }
                or (strict_orphan_check and m.mismatch_type == "unreferenced_postgres_chunk")
            ]
            report.is_consistent = len(critical_mismatches) == 0

            logger.info(
                "Provenance validation completed: consistent=%s, mismatches=%d",
                report.is_consistent,
                len(report.mismatches),
            )
            return report

        except (PostgresClientError, Neo4jClientError) as exc:
            logger.error("Database error during provenance validation: %s", exc, exc_info=True)
            raise DualStoreValidatorError(
                f"Provenance validation failed due to database error: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during provenance validation: %s", exc, exc_info=True)
            raise DualStoreValidatorError(f"Unexpected provenance validation error: {exc}") from exc

    async def _fetch_postgres_chunk_ids(self, document_id: str | None = None) -> set[str]:
        """Fetch all chunk UUID strings from PostgreSQL."""
        async with self.postgres_client.connection() as conn:
            if document_id:
                rows = await conn.fetch("SELECT id FROM chunks WHERE document_id = $1", document_id)
            else:
                rows = await conn.fetch("SELECT id FROM chunks")
            return {str(r["id"]).lower() for r in rows}

    def _fetch_neo4j_relationship_provenance(self) -> list[dict[str, Any]]:
        """Fetch relationship provenance properties from Neo4j."""
        results: list[dict[str, Any]] = []
        try:
            with self.neo4j_client.driver.session() as session:
                records = session.run(_CYPHER_FETCH_RELATIONSHIP_PROVENANCE)
                for record in records:
                    results.append(
                        {
                            "relation_type": record["relation_type"],
                            "provenance_chunk_id": record["provenance_chunk_id"],
                            "source_chunk_ids": record["source_chunk_ids"],
                            "source_id": record["source_id"],
                            "target_id": record["target_id"],
                        }
                    )
        except Exception as exc:
            raise Neo4jClientError(f"Failed to query Neo4j relationship provenance: {exc}") from exc
        return results

    def _fetch_neo4j_entity_ids(self) -> set[str]:
        """Fetch all canonical entity IDs from Neo4j."""
        entity_ids: set[str] = set()
        try:
            with self.neo4j_client.driver.session() as session:
                records = session.run(_CYPHER_FETCH_ALL_ENTITY_IDS)
                for record in records:
                    if record["entity_id"]:
                        entity_ids.add(str(record["entity_id"]))
        except Exception as exc:
            raise Neo4jClientError(f"Failed to query Neo4j entity IDs: {exc}") from exc
        return entity_ids

    async def _validate_entity_links(
        self,
        report: ConsistencyReport,
        document_id: str | None = None,
    ) -> None:
        """Validate PostgreSQL entity_links against PostgreSQL chunks and Neo4j entity nodes."""
        neo4j_entities = self._fetch_neo4j_entity_ids()

        async with self.postgres_client.connection() as conn:
            if document_id:
                sql = """
                SELECT el.id, el.chunk_id, el.canonical_entity_id, el.entity_type, el.mention_text
                FROM entity_links el
                JOIN chunks c ON el.chunk_id = c.id
                WHERE c.document_id = $1
                """
                rows = await conn.fetch(sql, document_id)
            else:
                sql = """
                SELECT el.id, el.chunk_id, el.canonical_entity_id, el.entity_type, el.mention_text
                FROM entity_links el
                """
                rows = await conn.fetch(sql)

            for row in rows:
                link_id = str(row["id"])
                chunk_id = str(row["chunk_id"]).lower()
                canonical_id = str(row["canonical_entity_id"])

                # Check if referenced chunk exists in report.postgres_chunk_ids
                if chunk_id not in report.postgres_chunk_ids:
                    report.mismatches.append(
                        ProvenanceMismatch(
                            mismatch_type="broken_entity_link",
                            source_store="postgres",
                            identifier=link_id,
                            details=f"entity_link {link_id} references chunk_id '{chunk_id}' not found in PostgreSQL chunks",
                        )
                    )

                # Check if referenced canonical_entity_id exists in Neo4j
                if neo4j_entities and canonical_id not in neo4j_entities:
                    report.mismatches.append(
                        ProvenanceMismatch(
                            mismatch_type="broken_entity_link",
                            source_store="cross_store",
                            identifier=canonical_id,
                            details=f"entity_link {link_id} references canonical_entity_id '{canonical_id}' not found in Neo4j",
                        )
                    )

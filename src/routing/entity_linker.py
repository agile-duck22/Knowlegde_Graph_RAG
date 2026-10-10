r"""Entity Linker for Knowledge Graph RAG (TASK-304).

Maps raw entity mentions and question terms to canonical Neo4j entity IDs
conforming strictly to the domain ontology (``docs/ONTOLOGY.md``).

Design Constraints & Invariants (see docs/SECURITY.md, docs/DATA_MODEL.md, docs/RETRIEVAL_STRATEGY.md):
--------------------------------------------------------------------------------------------------------
* **Strict Ontology Compliance**: Maps entities to the 7 core entity types:
  ``Service``, ``Team``, ``API``, ``Database``, ``Repository``, ``Vulnerability``, ``Infrastructure``.
* **Dual Resolution Engine**:
  1. Primary In-Memory Entity Resolver (``EntityResolver``): Fast deterministic alias mapping,
     case-insensitive normalization, canonical prefix checking, and CVE pattern recognition.
  2. Graph Database Verification / Search (``Neo4jClient`` / ``SafeGraphRetriever``):
     - Uses template ``T-08`` (``TEMPLATE_ENTITY_LOOKUP_BY_ID``) to verify canonical existence.
     - Performs parameterized exact name or alias lookup against Neo4j property graph.
* **Deterministic & Safe Disambiguation**:
  - If a mention resolves to multiple distinct canonical entities without sufficient context/hint,
    it records the status as ``ambiguous`` with candidate list rather than silently guessing.
  - If a mention cannot be resolved to any known canonical entity, it records the status as ``unresolved``.
  - Never invents synthetic canonical IDs for unresolvable entities unless explicit fallback is enabled.
* **Predictable, Typed Result Schema**:
  - Emits ``EntityLink`` for individual mentions and ``EntityLinkingResult`` for query-level linking results.
* **Zero Raw / Dynamic Cypher**:
  - All database interactions use parameterized Cypher queries or audited templates.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from src.db.neo4j import Neo4jClient, Neo4jClientError, Neo4jConnectionError
from src.extraction.resolver import (
    AmbiguousEntityError,
    EntityResolver,
    ResolvedEntityTarget,
    UnknownEntityError,
)
from src.extraction.schemas import (
    _CANONICAL_ID_PREFIX,
    EntityType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class LinkResolutionStatus(StrEnum):
    """Resolution outcome status for an entity mention."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    DATABASE_ERROR = "database_error"


class EntityLink(BaseModel):
    """Structured representation of an individual resolved or attempted entity link."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mention: str = Field(..., description="Original raw mention or search term from user query.")
    canonical_id: str | None = Field(
        default=None,
        description="Resolved canonical entity ID (e.g., 'service:payment-gateway') if resolved.",
    )
    entity_type: EntityType | None = Field(
        default=None,
        description="Ontology entity type if resolved.",
    )
    primary_name: str | None = Field(
        default=None,
        description="Display name of the resolved entity.",
    )
    status: LinkResolutionStatus = Field(
        ...,
        description="Resolution status: resolved, ambiguous, unresolved, or database_error.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the resolution [0.0, 1.0].",
    )
    candidate_ids: list[str] = Field(
        default_factory=list,
        description="List of candidate canonical IDs if ambiguous.",
    )
    verified_in_graph: bool = Field(
        default=False,
        description="True if the resolved entity was verified to exist in the Neo4j graph store.",
    )
    error_message: str | None = Field(
        default=None,
        description="Sanitized error description if database lookup or resolution failed.",
    )


class EntityLinkingResult(BaseModel):
    """Aggregate result containing all entity links for a given query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(..., description="The original query from which mentions were linked.")
    links: list[EntityLink] = Field(
        default_factory=list,
        description="List of all extracted and attempted entity links.",
    )

    @property
    def resolved_links(self) -> list[EntityLink]:
        """Return only successfully resolved links."""
        return [link for link in self.links if link.status == LinkResolutionStatus.RESOLVED]

    @property
    def canonical_ids(self) -> list[str]:
        """Return a deduplicated list of resolved canonical IDs in deterministic order."""
        seen: set[str] = set()
        ids: list[str] = []
        for link in self.resolved_links:
            if link.canonical_id and link.canonical_id not in seen:
                seen.add(link.canonical_id)
                ids.append(link.canonical_id)
        return ids

    @property
    def has_unresolved(self) -> bool:
        """True if any mention could not be resolved or was ambiguous."""
        return any(
            link.status in (LinkResolutionStatus.UNRESOLVED, LinkResolutionStatus.AMBIGUOUS)
            for link in self.links
        )


# ---------------------------------------------------------------------------
# Parameterized Cypher Query for Graph Entity Lookup
# ---------------------------------------------------------------------------

#: Parameterized Cypher lookup by canonical ID or exact name / alias matching
_CYPHER_FIND_ENTITY_IN_GRAPH: Final[str] = """
MATCH (e:Entity)
WHERE e.id = $id_or_name
   OR toLower(e.name) = toLower($id_or_name)
   OR toLower($id_or_name) IN [a IN coalesce(e.aliases, []) | toLower(a)]
RETURN
    e.id AS entity_id,
    e.name AS name,
    e.type AS entity_type,
    e.aliases AS aliases
LIMIT 5
"""


# ---------------------------------------------------------------------------
# Entity Linker Class
# ---------------------------------------------------------------------------


class EntityLinker:
    """Links entity mentions in user queries to canonical Neo4j entity IDs.

    Features:
    - In-memory deterministic resolution using domain ontology and alias catalog.
    - Optional Neo4j graph store verification and discovery via parameterized Cypher.
    - Mention extraction heuristic targeting canonical IDs, CVE identifiers, and domain entities.
    - Safe handling of ambiguous terms, unresolvable mentions, and database failures.
    """

    def __init__(
        self,
        resolver: EntityResolver | None = None,
        neo4j_client: Neo4jClient | None = None,
        verify_graph: bool = False,
    ) -> None:
        """Initialize the EntityLinker.

        Parameters
        ----------
        resolver:
            Optional EntityResolver instance. If None, a standard default resolver is created.
        neo4j_client:
            Optional Neo4jClient instance for graph database verification and lookups.
        verify_graph:
            If True, attempts to verify resolved entities against Neo4j when client is available.
        """
        self._resolver = resolver if resolver is not None else EntityResolver()
        self._neo4j_client = neo4j_client
        self._verify_graph = verify_graph

    @property
    def resolver(self) -> EntityResolver:
        """Underlying EntityResolver instance."""
        return self._resolver

    @property
    def neo4j_client(self) -> Neo4jClient | None:
        """Optional Neo4jClient instance."""
        return self._neo4j_client

    def link_mention(
        self,
        mention: str,
        hint_type: EntityType | None = None,
    ) -> EntityLink:
        """Link a single entity mention to a canonical entity ID.

        Parameters
        ----------
        mention:
            The raw mention, alias, or candidate identifier.
        hint_type:
            Optional hint for expected EntityType to assist disambiguation.

        Returns
        -------
        EntityLink:
            Typed resolution outcome.
        """
        raw_mention = mention.strip()
        if not raw_mention:
            return EntityLink(
                mention=mention,
                status=LinkResolutionStatus.UNRESOLVED,
                confidence=0.0,
                error_message="Empty or whitespace mention cannot be linked.",
            )

        # 1. Attempt deterministic resolution via EntityResolver
        resolved_target: ResolvedEntityTarget | None = None
        try:
            resolved_target = self._resolver.resolve_mention(
                raw_mention,
                hint_type=hint_type,
                allow_fallback=False,
            )
        except AmbiguousEntityError as exc:
            # Extract candidate IDs from ambiguity error message
            cand_matches = re.findall(
                r"'(?:service|team|database|api|repo|infra|vuln):[a-z0-9-]+'", str(exc)
            )
            cand_ids = sorted({m.strip("'") for m in cand_matches})
            return EntityLink(
                mention=raw_mention,
                status=LinkResolutionStatus.AMBIGUOUS,
                confidence=0.5,
                candidate_ids=cand_ids,
                error_message=str(exc),
            )
        except UnknownEntityError:
            resolved_target = None

        # 2. If resolved via in-memory catalog
        if resolved_target is not None:
            verified = False
            if self._verify_graph and self._neo4j_client is not None:
                verified = self._verify_in_graph(resolved_target.canonical_id)

            return EntityLink(
                mention=raw_mention,
                canonical_id=resolved_target.canonical_id,
                entity_type=resolved_target.entity_type,
                primary_name=resolved_target.primary_name,
                status=LinkResolutionStatus.RESOLVED,
                confidence=1.0 if verified or not self._verify_graph else 0.95,
                verified_in_graph=verified,
            )

        # 3. If unresolved in memory, attempt Neo4j lookup if client is configured
        if self._neo4j_client is not None:
            return self._lookup_in_neo4j(raw_mention, hint_type)

        # 4. Unresolvable entity
        return EntityLink(
            mention=raw_mention,
            status=LinkResolutionStatus.UNRESOLVED,
            confidence=0.0,
            error_message=f"Entity mention '{raw_mention}' could not be resolved to any canonical entity.",
        )

    def link_mentions(
        self,
        mentions: Sequence[str],
        hint_types: Sequence[EntityType | None] | None = None,
    ) -> list[EntityLink]:
        """Link a sequence of entity mentions.

        Parameters
        ----------
        mentions:
            Sequence of raw mention strings.
        hint_types:
            Optional parallel sequence of EntityType hints.

        Returns
        -------
        list[EntityLink]:
            List of EntityLink results corresponding to input mentions.
        """
        links: list[EntityLink] = []
        for i, mention in enumerate(mentions):
            hint = hint_types[i] if hint_types and i < len(hint_types) else None
            links.append(self.link_mention(mention, hint_type=hint))
        return links

    def link_query(
        self,
        query: str,
        extracted_entities: Sequence[str] | None = None,
    ) -> EntityLinkingResult:
        """Extract and link all candidate entity mentions in a user query.

        Parameters
        ----------
        query:
            The raw user question or prompt.
        extracted_entities:
            Optional pre-extracted entity candidates (e.g., from IntentClassifier).
            If None or empty, candidates are extracted via domain regex heuristic.

        Returns
        -------
        EntityLinkingResult:
            Aggregate linking result.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return EntityLinkingResult(query=clean_query, links=[])

        # 1. Determine candidate mentions
        candidates: list[str] = []
        if extracted_entities:
            candidates.extend([e.strip() for e in extracted_entities if e.strip()])
        else:
            candidates.extend(self.extract_candidate_mentions(clean_query))

        # Deduplicate while preserving order
        unique_candidates: list[str] = []
        seen_cand: set[str] = set()
        for c in candidates:
            norm_c = c.lower()
            if norm_c not in seen_cand:
                seen_cand.add(norm_c)
                unique_candidates.append(c)

        # 2. Link each candidate
        links = self.link_mentions(unique_candidates)
        return EntityLinkingResult(query=clean_query, links=links)

    def extract_candidate_mentions(self, text: str) -> list[str]:
        """Extract potential entity mentions from free-form query text.

        Uses regex rules targeting:
        - Canonical ID patterns (e.g., ``service:payment-gateway``)
        - CVE vulnerability identifiers (e.g., ``CVE-2024-21626``, ``cve-2023-38545``)
        - Known seed aliases registered in the resolver
        """
        if not text:
            return []

        candidates: list[str] = []
        seen_lower: set[str] = set()

        def _add(m: str) -> None:
            m_clean = m.strip()
            if m_clean and m_clean.lower() not in seen_lower:
                seen_lower.add(m_clean.lower())
                candidates.append(m_clean)

        # 1. Match canonical IDs (prefix:kebab-name)
        for match in re.finditer(
            r"\b(?:service|team|database|api|repo|infra|vuln):[a-z0-9-]+\b", text, re.IGNORECASE
        ):
            _add(match.group(0))

        # 2. Match CVE identifiers
        for match in re.finditer(r"\b(?:cve[-_\s]*)?\d{4}[-_]\d{4,7}\b", text, re.IGNORECASE):
            _add(match.group(0))

        # 3. Match known seed aliases from DEFAULT_ALIAS_MAP
        text_lower = text.lower()
        from src.extraction.resolver import DEFAULT_ALIAS_MAP

        # Sort aliases by length descending so longer phrases match first
        sorted_aliases = sorted(DEFAULT_ALIAS_MAP.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            # Match whole-word boundary for aliases
            escaped_alias = re.escape(alias)
            if re.search(rf"\b{escaped_alias}\b", text_lower):
                _add(alias)

        return candidates

    def _verify_in_graph(self, canonical_id: str) -> bool:
        """Verify that a canonical ID exists in Neo4j."""
        if self._neo4j_client is None:
            return False

        cypher = "MATCH (e:Entity {id: $id}) RETURN count(e) AS cnt"
        try:
            with self._neo4j_client.driver.session() as session:
                result = session.run(cypher, {"id": canonical_id})
                record = result.single()
                if record and record.get("cnt", 0) > 0:
                    return True
        except Exception as exc:
            logger.warning("Graph verification query failed for ID '%s': %s", canonical_id, exc)
            return False
        return False

    def _lookup_in_neo4j(
        self,
        mention: str,
        hint_type: EntityType | None = None,
    ) -> EntityLink:
        """Perform a parameterized Cypher search in Neo4j for an unresolved mention."""
        if self._neo4j_client is None:
            return EntityLink(
                mention=mention,
                status=LinkResolutionStatus.UNRESOLVED,
                confidence=0.0,
            )

        try:
            with self._neo4j_client.driver.session() as session:
                result = session.run(_CYPHER_FIND_ENTITY_IN_GRAPH, {"id_or_name": mention})
                records = list(result)
        except (Neo4jConnectionError, Neo4jClientError, Exception) as exc:
            logger.error("Neo4j entity lookup failed for mention '%s': %s", mention, exc)
            return EntityLink(
                mention=mention,
                status=LinkResolutionStatus.DATABASE_ERROR,
                confidence=0.0,
                error_message="Graph database error during entity lookup.",
            )

        if not records:
            return EntityLink(
                mention=mention,
                status=LinkResolutionStatus.UNRESOLVED,
                confidence=0.0,
                error_message=f"Entity '{mention}' not found in graph database.",
            )

        # Filter by hint_type if provided
        if hint_type:
            filtered = [r for r in records if r.get("entity_type") == hint_type]
            if filtered:
                records = filtered

        if len(records) == 1:
            rec = records[0]
            canon_id = str(rec["entity_id"])
            e_type = str(rec["entity_type"])
            p_name = str(rec.get("name", canon_id))
            valid_e_type = e_type if e_type in _CANONICAL_ID_PREFIX else None
            return EntityLink(
                mention=mention,
                canonical_id=canon_id,
                entity_type=valid_e_type,
                primary_name=p_name,
                status=LinkResolutionStatus.RESOLVED,
                confidence=0.95,
                verified_in_graph=True,
            )

        # Multiple matches in graph -> Ambiguous
        candidate_ids = sorted({str(r["entity_id"]) for r in records})
        return EntityLink(
            mention=mention,
            status=LinkResolutionStatus.AMBIGUOUS,
            confidence=0.5,
            candidate_ids=candidate_ids,
            verified_in_graph=True,
            error_message=f"Mention '{mention}' matched multiple graph entities: {candidate_ids}",
        )

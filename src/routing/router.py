r"""Router Engine coordinating intent classification, entity linking, and retrieval delegation (TASK-305).

Design Constraints & Routing Architecture (see docs/RETRIEVAL_STRATEGY.md, docs/API_SPEC.md, docs/SECURITY.md):
-----------------------------------------------------------------------------------------------------------------
* **Four Authoritative Routing Paths**:
  1. `GRAPH_ONLY`:
     - Resolves canonical entity IDs from query terms via ``EntityLinker``.
     - Executes approved parameterized Cypher query template(s) via ``SafeGraphRetriever``.
     - Returns structured graph records, paths, and extracted provenance ``chunk_id``s without vector calls.
     - If no entities could be resolved, falls back to ``HYBRID`` dual-retrieval to ensure coverage.
  2. `VECTOR_ONLY`:
     - Embeds the query text via ``EmbeddingGenerator`` (or accepts pre-computed embedding vector).
     - Searches top-$k$ text chunks in PostgreSQL/pgvector via ``VectorRetriever`` with cosine similarity.
     - No graph traversal or Cypher queries are executed.
  3. `HYBRID`:
     - Concurrently or sequentially executes both Graph retrieval (if entities linkable) and Vector retrieval.
     - Fetches supporting text chunks for graph provenance IDs if needed for downstream hydration (Phase 4).
     - Returns combined graph results and vector chunks.
  4. `OUT_OF_SCOPE`:
     - Immediately halts retrieval without database calls (Neo4j or PostgreSQL).
     - Returns an explicit empty/refusal routing payload.
* **Strict Safety & Parameterization**:
  - Zero dynamic Cypher: all graph queries select strictly from ``CypherTemplateCatalog`` (T-01..T-08).
  - Validates parameters before executing templates.
  - Redacts sensitive details from error messages.
* **Typed & Predictable Results**:
  - Emits ``RoutingRequest``, ``RoutingResult``, ``RetrievalPayload`` structures.
* **Test Isolation**:
  - Accepts dependency injection for all components (classifier, linker, graph retriever, vector retriever, embedding client).
  - Offline unit tests with mocks by default.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.retrieval.cypher_templates import (
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    TEMPLATE_DIRECT_NEIGHBORHOOD,
    TEMPLATE_ENTITY_LOOKUP_BY_ID,
    TEMPLATE_INCOMING_SERVICE_DEPENDENTS,
    TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
    TEMPLATE_MULTI_HOP_DEPENDENCY,
    TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
    TEMPLATE_TEAM_OWNERSHIP_SERVICES,
    TEMPLATE_VULNERABILITY_BLAST_RADIUS,
    CypherTemplateCatalog,
)
from src.retrieval.graph_retriever import (
    GraphQueryResult,
    SafeGraphRetriever,
)
from src.retrieval.vector_retriever import (
    RetrievedChunk,
    VectorRetriever,
)
from src.routing.entity_linker import (
    EntityLinker,
    EntityLinkingResult,
)
from src.routing.intent_classifier import (
    IntentClassificationResult,
    QueryIntent,
    QueryIntentClassifier,
)

if TYPE_CHECKING:
    from src.embeddings.client import EmbeddingGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request & Result Models
# ---------------------------------------------------------------------------


class RoutingRequest(BaseModel):
    """Input parameters for the Router Engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(..., description="The user question or search prompt.")
    retrieval_override: QueryIntent | str | None = Field(
        default=None,
        description="Optional manual override for execution mode (GRAPH_ONLY, VECTOR_ONLY, HYBRID, OUT_OF_SCOPE).",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum vector chunks to retrieve (overrides settings if provided).",
    )
    graph_limit: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum graph nodes/paths to retrieve (overrides settings if provided).",
    )
    similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold for vector search.",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional document ID filter for vector search.",
    )
    doc_type: str | None = Field(
        default=None,
        description="Optional doc_type filter for vector search.",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSONB metadata key-value filter for vector search.",
    )
    include_provenance: bool = Field(
        default=True,
        description="Whether to extract and include provenance chunk IDs in results.",
    )


@dataclass(frozen=True)
class RoutingResult:
    """Consolidated retrieval result produced by the Router Engine."""

    query: str
    mode: QueryIntent
    classification: IntentClassificationResult
    entity_linking: EntityLinkingResult
    graph_results: list[GraphQueryResult] = field(default_factory=list)
    vector_chunks: list[RetrievedChunk] = field(default_factory=list)
    selected_template_id: str | None = None
    all_provenance_chunk_ids: list[str] = field(default_factory=list)
    error_message: str | None = None

    @property
    def has_graph_data(self) -> bool:
        """Return True if any graph records were successfully retrieved."""
        return any(not res.is_empty for res in self.graph_results)

    @property
    def has_vector_data(self) -> bool:
        """Return True if any vector chunks were retrieved."""
        return len(self.vector_chunks) > 0

    @property
    def is_empty(self) -> bool:
        """Return True if no graph records or vector chunks were retrieved."""
        return not self.has_graph_data and not self.has_vector_data


# ---------------------------------------------------------------------------
# Template Parameter Mapping Helper
# ---------------------------------------------------------------------------

_TEMPLATE_PARAM_KEYS: dict[str, str] = {
    TEMPLATE_DIRECT_NEIGHBORHOOD: "entity_id",
    TEMPLATE_MULTI_HOP_DEPENDENCY: "entity_id",
    TEMPLATE_TEAM_OWNERSHIP_SERVICES: "team_id",
    TEMPLATE_VULNERABILITY_BLAST_RADIUS: "vuln_id",
    TEMPLATE_SERVICE_DEPENDENCY_CHAIN: "service_id",
    TEMPLATE_INCOMING_SERVICE_DEPENDENTS: "service_id",
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS: "database_id",
    TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS: "infra_id",
    TEMPLATE_ENTITY_LOOKUP_BY_ID: "entity_id",
}


def _build_template_params(
    template_id: str, canonical_id: str, limit: int | None = None
) -> dict[str, Any]:
    """Build the parameter dictionary for a given Cypher template and canonical ID."""
    param_key = _TEMPLATE_PARAM_KEYS.get(template_id, "entity_id")
    params: dict[str, Any] = {param_key: canonical_id}
    if limit is not None:
        params["limit"] = limit
    return params


# Map templates to allowed canonical entity ID prefixes based on their Cypher signatures
_TEMPLATE_COMPATIBLE_PREFIXES: dict[str, set[str]] = {
    TEMPLATE_TEAM_OWNERSHIP_SERVICES: {"team"},
    TEMPLATE_VULNERABILITY_BLAST_RADIUS: {"vuln"},
    TEMPLATE_SERVICE_DEPENDENCY_CHAIN: {"service"},
    TEMPLATE_INCOMING_SERVICE_DEPENDENTS: {"service"},
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS: {"database"},
    TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS: {"infra"},
    # Generic entity templates that accept any entity type
    TEMPLATE_DIRECT_NEIGHBORHOOD: {"service", "team", "api", "database", "repo", "vuln", "infra"},
    TEMPLATE_MULTI_HOP_DEPENDENCY: {"service", "team", "api", "database", "repo", "vuln", "infra"},
    TEMPLATE_ENTITY_LOOKUP_BY_ID: {"service", "team", "api", "database", "repo", "vuln", "infra"},
}

# Default canonical prefix mapping to preferred template
_PREFIX_TO_DEFAULT_TEMPLATE: dict[str, str] = {
    "team": TEMPLATE_TEAM_OWNERSHIP_SERVICES,
    "vuln": TEMPLATE_VULNERABILITY_BLAST_RADIUS,
    "database": TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    "infra": TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
    "service": TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
}


def _is_entity_compatible_with_template(canonical_id: str, template_id: str) -> bool:
    """Check if the canonical entity ID's type prefix is compatible with the Cypher template."""
    prefix = canonical_id.split(":", 1)[0].lower() if ":" in canonical_id else ""
    allowed_prefixes = _TEMPLATE_COMPATIBLE_PREFIXES.get(template_id)
    if allowed_prefixes is None:
        return False
    # If the entity has no prefix, it is only compatible with generic templates (T-01, T-02, T-08)
    if not prefix:
        return template_id in (
            TEMPLATE_DIRECT_NEIGHBORHOOD,
            TEMPLATE_MULTI_HOP_DEPENDENCY,
            TEMPLATE_ENTITY_LOOKUP_BY_ID,
        )
    return prefix in allowed_prefixes


def _select_best_template_for_entity(
    canonical_id: str, suggested_template: str | None = None
) -> str | None:
    """Choose the most appropriate Cypher template based on entity type and suggested template.

    Returns None if no compatible template exists for the entity.
    """
    prefix = canonical_id.split(":", 1)[0].lower() if ":" in canonical_id else ""

    # 1. If LLM suggested a template, check if it exists and is compatible with entity type
    if suggested_template and CypherTemplateCatalog.has_template(suggested_template):
        if _is_entity_compatible_with_template(canonical_id, suggested_template):
            return suggested_template
        logger.debug(
            "Suggested template '%s' is incompatible with entity '%s' (prefix '%s'). Falling back.",
            suggested_template,
            canonical_id,
            prefix,
        )

    # 2. Check prefix-specific default template
    if prefix in _PREFIX_TO_DEFAULT_TEMPLATE:
        candidate = _PREFIX_TO_DEFAULT_TEMPLATE[prefix]
        if _is_entity_compatible_with_template(canonical_id, candidate):
            return candidate

    # 3. Compatible fallback to generic neighborhood template (T-01)
    if _is_entity_compatible_with_template(canonical_id, TEMPLATE_DIRECT_NEIGHBORHOOD):
        return TEMPLATE_DIRECT_NEIGHBORHOOD

    # 4. No compatible template found; safe skip
    logger.warning("No compatible Cypher template found for entity '%s'.", canonical_id)
    return None


# ---------------------------------------------------------------------------
# Router Engine Class
# ---------------------------------------------------------------------------


class RouterEngine:
    """Coordinates query classification, entity linking, and retrieval delegation.

    Architecture:
    - Intent Classifier (`QueryIntentClassifier`): Determines mode (`GRAPH_ONLY`, `VECTOR_ONLY`, `HYBRID`, `OUT_OF_SCOPE`).
    - Entity Linker (`EntityLinker`): Matches question terms to canonical Neo4j IDs.
    - Graph Retriever (`SafeGraphRetriever`): Executes audited parameterized Cypher templates.
    - Vector Retriever (`VectorRetriever`): Performs dense cosine vector search on PostgreSQL.
    - Embedding Generator (`EmbeddingGenerator`): Computes dense query vectors for semantic search.
    """

    def __init__(
        self,
        classifier: QueryIntentClassifier,
        entity_linker: EntityLinker,
        graph_retriever: SafeGraphRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the RouterEngine with component dependencies."""
        self._classifier = classifier
        self._entity_linker = entity_linker
        self._graph_retriever = graph_retriever
        self._vector_retriever = vector_retriever
        self._embedding_generator = embedding_generator
        self._settings = settings or get_settings()

    @property
    def classifier(self) -> QueryIntentClassifier:
        return self._classifier

    @property
    def entity_linker(self) -> EntityLinker:
        return self._entity_linker

    @property
    def graph_retriever(self) -> SafeGraphRetriever | None:
        return self._graph_retriever

    @property
    def vector_retriever(self) -> VectorRetriever | None:
        return self._vector_retriever

    @property
    def embedding_generator(self) -> EmbeddingGenerator | None:
        return self._embedding_generator

    async def route_and_retrieve(
        self,
        request: RoutingRequest | str,
        query_vector: Sequence[float] | None = None,
    ) -> RoutingResult:
        """Route an incoming question and delegate retrieval to appropriate stores.

        Parameters
        ----------
        request:
            The input `RoutingRequest` or raw question string.
        query_vector:
            Optional pre-computed embedding vector (avoids extra embedding API call if provided).

        Returns
        -------
        RoutingResult:
            Consolidated structured graph subgraphs and vector chunks.
        """
        # 1. Normalize request
        if isinstance(request, str):
            req = RoutingRequest(query=request)
        else:
            req = request

        query_text = req.query.strip()

        # 2. Step 1: Intent Classification
        classification = self._classifier.classify(
            query=query_text,
            retrieval_override=req.retrieval_override,
        )
        mode = classification.mode

        # Fast-exit: OUT_OF_SCOPE inquiries require zero database retrievals
        if mode == QueryIntent.OUT_OF_SCOPE:
            logger.info("Query '%s' classified as OUT_OF_SCOPE; skipping retrieval.", query_text)
            return RoutingResult(
                query=query_text,
                mode=QueryIntent.OUT_OF_SCOPE,
                classification=classification,
                entity_linking=EntityLinkingResult(query=query_text, links=[]),
                graph_results=[],
                vector_chunks=[],
                selected_template_id=None,
                all_provenance_chunk_ids=[],
            )

        # 3. Step 2: Entity Linking
        # Use extracted entities from classifier if available to guide linking
        extracted_mentions = classification.extracted_entities
        linking_result = self._entity_linker.link_query(
            query=query_text,
            extracted_entities=extracted_mentions if extracted_mentions else None,
        )

        # If GRAPH_ONLY was requested but NO entities could be resolved, fall back to HYBRID
        effective_mode = mode
        if effective_mode == QueryIntent.GRAPH_ONLY and not linking_result.canonical_ids:
            logger.warning(
                "Query '%s' classified as GRAPH_ONLY but no canonical entities could be resolved. "
                "Falling back to HYBRID mode to ensure vector search coverage.",
                query_text,
            )
            effective_mode = QueryIntent.HYBRID
        elif (
            effective_mode == QueryIntent.GRAPH_ONLY
            and self._vector_retriever is not None
            and req.retrieval_override is None
        ):
            # Multi-hop and relationship queries in enterprise systems frequently require supporting
            # textual documentation, specifications, or architectural context. When both retrievers are
            # available and no explicit caller override was provided, run dual retrieval.
            effective_mode = QueryIntent.HYBRID

        # 4. Step 3: Retrieval Delegation
        graph_results: list[GraphQueryResult] = []
        vector_chunks: list[RetrievedChunk] = []
        selected_template: str | None = None
        provenance_ids: list[str] = []
        error_msg: str | None = None

        # A. Execute Graph Retrieval if mode is GRAPH_ONLY or HYBRID
        if effective_mode in (QueryIntent.GRAPH_ONLY, QueryIntent.HYBRID):
            if self._graph_retriever is not None and linking_result.canonical_ids:
                try:
                    graph_results, selected_template = self._execute_graph_retrieval(
                        canonical_ids=linking_result.canonical_ids,
                        suggested_template=classification.suggested_template_id,
                        limit=req.graph_limit or self._settings.graph_node_limit,
                    )
                    for gr in graph_results:
                        provenance_ids.extend(gr.source_chunk_ids)
                except Exception as exc:
                    logger.error("Graph retrieval execution failed: %s", exc)
                    error_msg = f"Graph retrieval failed: {exc}"
            elif self._graph_retriever is None and effective_mode == QueryIntent.GRAPH_ONLY:
                error_msg = "Graph retriever component is not configured on RouterEngine."

        # B. Execute Vector Retrieval if mode is VECTOR_ONLY or HYBRID
        if effective_mode in (QueryIntent.VECTOR_ONLY, QueryIntent.HYBRID):
            if self._vector_retriever is not None:
                try:
                    # Compute embedding vector if not supplied
                    vec = query_vector
                    if vec is None and self._embedding_generator is not None:
                        vec = self._embedding_generator.generate_single(query_text)

                    if vec is not None:
                        # Select top_k depending on mode
                        top_k = req.top_k
                        if top_k is None:
                            top_k = (
                                self._settings.vector_top_k
                                if effective_mode == QueryIntent.VECTOR_ONLY
                                else self._settings.hybrid_vector_top_k
                            )

                        vector_chunks = await self._vector_retriever.search(
                            query_vector=vec,
                            top_k=top_k,
                            similarity_threshold=req.similarity_threshold
                            or self._settings.vector_similarity_threshold,
                            document_id=req.document_id,
                            doc_type=req.doc_type,
                            metadata_filter=req.metadata_filter,
                        )
                        for chunk in vector_chunks:
                            provenance_ids.append(chunk.chunk_id)
                    else:
                        logger.warning(
                            "No embedding vector available for query '%s'; skipping vector search.",
                            query_text,
                        )
                except Exception as exc:
                    logger.error("Vector retrieval execution failed: %s", exc)
                    if error_msg:
                        error_msg += f" | Vector retrieval failed: {exc}"
                    else:
                        error_msg = f"Vector retrieval failed: {exc}"
            elif self._vector_retriever is None and effective_mode == QueryIntent.VECTOR_ONLY:
                error_msg = "Vector retriever component is not configured on RouterEngine."

        # Deduplicate all collected provenance chunk IDs
        seen_chunks: set[str] = set()
        deduped_provenance_ids: list[str] = []
        for cid in provenance_ids:
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                deduped_provenance_ids.append(cid)

        return RoutingResult(
            query=query_text,
            mode=effective_mode,
            classification=classification,
            entity_linking=linking_result,
            graph_results=graph_results,
            vector_chunks=vector_chunks,
            selected_template_id=selected_template,
            all_provenance_chunk_ids=deduped_provenance_ids,
            error_message=error_msg,
        )

    def _execute_graph_retrieval(
        self,
        canonical_ids: list[str],
        suggested_template: str | None = None,
        limit: int = 50,
    ) -> tuple[list[GraphQueryResult], str | None]:
        """Execute approved Cypher template(s) for the resolved canonical entity IDs."""
        if not self._graph_retriever or not canonical_ids:
            return [], None

        results: list[GraphQueryResult] = []
        last_template_used: str | None = None

        for canon_id in canonical_ids:
            template_id = _select_best_template_for_entity(canon_id, suggested_template)
            if template_id is None:
                logger.warning(
                    "Skipping graph query for entity '%s': no compatible template found.",
                    canon_id,
                )
                continue

            last_template_used = template_id
            params = _build_template_params(template_id, canon_id, limit=limit)

            query_res = self._graph_retriever.execute_template(
                template_id=template_id,
                parameters=params,
                limit=limit,
            )
            results.append(query_res)

        return results, last_template_used

r"""Audited Parameterized Cypher Query Template Catalog (TASK-301).

Design Constraints & Security Invariants (see docs/SECURITY.md, docs/RETRIEVAL_STRATEGY.md):
--------------------------------------------------------------------------------------------
* **Zero dynamic Cypher**: All Cypher statements are fixed, audited module-level string constants.
  No f-strings, `%` formatting, or `str.format` calls are used to build query text.
* **Parameterized execution**: All runtime values (entity IDs, limits, thresholds, labels) must be passed
  via parameter dictionaries.
* **Bounded traversals and result sets**:
  - Traversal depth is strictly limited ($D \le 3$).
  - Every query enforces a parameterized or hard-capped `LIMIT` to prevent out-of-memory or graph traversal explosion.
* **Ontology Alignment**: All node labels and relationship types strictly adhere to `docs/ONTOLOGY.md`.
* **Provenance Preserved**: Every template fetching relationships or paths returns `source_chunk_ids` for dual-store
  citation verification downstream (Phase 4).
* **Typed Specifications**: Each template is packaged into a `CypherTemplateSpec` containing its unique identifier,
  name, purpose, Cypher query string, required parameter names, and default parameter limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

# ---------------------------------------------------------------------------
# Core Template Identifiers
# ---------------------------------------------------------------------------

TEMPLATE_DIRECT_NEIGHBORHOOD: Final[str] = "T-01"
TEMPLATE_MULTI_HOP_DEPENDENCY: Final[str] = "T-02"
TEMPLATE_TEAM_OWNERSHIP_SERVICES: Final[str] = "T-03"
TEMPLATE_VULNERABILITY_BLAST_RADIUS: Final[str] = "T-04"
TEMPLATE_SERVICE_DEPENDENCY_CHAIN: Final[str] = "T-05"
TEMPLATE_DATABASE_PRODUCERS_CONSUMERS: Final[str] = "T-06"
TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS: Final[str] = "T-07"
TEMPLATE_ENTITY_LOOKUP_BY_ID: Final[str] = "T-08"

# ---------------------------------------------------------------------------
# Approved Parameterized Cypher Queries (docs/RETRIEVAL_STRATEGY.md §2 & ONTOLOGY.md)
# ---------------------------------------------------------------------------

# T-01: Direct Entity Neighborhood (1-hop traversal)
# Retrieves 1-hop connected entities, edge types, relationship descriptions, and supporting chunk IDs.
_CYPHER_DIRECT_NEIGHBORHOOD = """
MATCH (e:Entity {id: $entity_id})-[r]-(neighbor:Entity)
RETURN
    e.id AS source_id,
    e.name AS source_name,
    e.type AS source_type,
    type(r) AS rel_type,
    r.description AS rel_desc,
    r.confidence AS confidence,
    r.source_chunk_ids AS source_chunk_ids,
    neighbor.id AS target_id,
    neighbor.name AS target_name,
    neighbor.type AS target_type,
    labels(neighbor) AS target_labels
LIMIT $limit
"""

# T-02: Multi-Hop Dependency Path (2 to 3 hops)
# Traces directional DEPENDS_ON and CALLS_API dependencies between services and downstream targets.
_CYPHER_MULTI_HOP_DEPENDENCY = """
MATCH path = (e:Entity {id: $entity_id})-[:DEPENDS_ON|CALLS_API*1..3]->(target:Entity)
WITH path, relationships(path) AS rels, nodes(path) AS nds
UNWIND rels AS r
RETURN
    [n IN nds | {id: n.id, name: n.name, type: n.type}] AS path_nodes,
    type(r) AS rel_type,
    r.description AS rel_desc,
    r.confidence AS confidence,
    r.source_chunk_ids AS source_chunk_ids,
    startNode(r).id AS rel_start_id,
    endNode(r).id AS rel_end_id
LIMIT $limit
"""

# T-03: Team Ownership & Service Impact
# Finds all services owned by a given team, their exposed APIs, and any active security vulnerabilities.
_CYPHER_TEAM_OWNERSHIP_SERVICES = """
MATCH (t:Team {id: $team_id})<-[r_own:OWNED_BY]-(s:Service)
OPTIONAL MATCH (s)-[r_api:EXPOSES_API]->(a:API)
OPTIONAL MATCH (s)-[r_vuln:AFFECTED_BY]->(v:Vulnerability)
RETURN
    t.id AS team_id,
    t.name AS team_name,
    s.id AS service_id,
    s.name AS service_name,
    r_own.source_chunk_ids AS ownership_chunk_ids,
    collect(DISTINCT {id: a.id, name: a.name, chunk_ids: r_api.source_chunk_ids}) AS exposed_apis,
    collect(DISTINCT {id: v.id, name: v.name, chunk_ids: r_vuln.source_chunk_ids}) AS vulnerabilities
LIMIT $limit
"""

# T-04: Vulnerability Blast Radius
# Identifies all services directly or transitively impacted by a security vulnerability.
_CYPHER_VULNERABILITY_BLAST_RADIUS = """
MATCH (v:Vulnerability {id: $vuln_id})<-[r_vuln:AFFECTED_BY]-(impacted:Service)
OPTIONAL MATCH (dependent:Service)-[r_dep:DEPENDS_ON*1..2]->(impacted)
RETURN
    v.id AS vuln_id,
    v.name AS vuln_name,
    impacted.id AS vulnerable_service_id,
    impacted.name AS vulnerable_service_name,
    r_vuln.source_chunk_ids AS vuln_chunk_ids,
    collect(DISTINCT {
        id: dependent.id,
        name: dependent.name
    }) AS affected_dependents
LIMIT $limit
"""

# T-05: Strict Downstream Service Dependency Chain
# Direct and transitive runtime downstream dependencies of a service (up to 3 hops).
_CYPHER_SERVICE_DEPENDENCY_CHAIN = """
MATCH path = (s:Service {id: $service_id})-[:DEPENDS_ON*1..3]->(downstream:Service)
WITH path, relationships(path) AS rels, nodes(path) AS nds
UNWIND rels AS r
RETURN
    s.id AS root_service_id,
    [n IN nds | {id: n.id, name: n.name, type: n.type}] AS dependency_chain,
    downstream.id AS downstream_service_id,
    downstream.name AS downstream_service_name,
    type(r) AS rel_type,
    r.source_chunk_ids AS source_chunk_ids
LIMIT $limit
"""

# T-06: Database Producers and Consumers
# Identifies all services that read from or write to a specific database.
_CYPHER_DATABASE_PRODUCERS_CONSUMERS = """
MATCH (d:Database {id: $database_id})
OPTIONAL MATCH (reader:Service)-[r_read:READS_FROM]->(d)
OPTIONAL MATCH (writer:Service)-[r_write:WRITES_TO]->(d)
RETURN
    d.id AS database_id,
    d.name AS database_name,
    collect(DISTINCT {
        service_id: reader.id,
        service_name: reader.name,
        rel: 'READS_FROM',
        chunk_ids: r_read.source_chunk_ids
    }) AS readers,
    collect(DISTINCT {
        service_id: writer.id,
        service_name: writer.name,
        rel: 'WRITES_TO',
        chunk_ids: r_write.source_chunk_ids
    }) AS writers
LIMIT $limit
"""

# T-07: Infrastructure Deployments and Event Producers
# Identifies all services and databases deployed on or publishing events to an infrastructure resource.
_CYPHER_INFRASTRUCTURE_DEPLOYMENTS = """
MATCH (infra:Infrastructure {id: $infra_id})
OPTIONAL MATCH (deployed:Entity)-[r_dep:DEPLOYED_ON]->(infra)
OPTIONAL MATCH (producer:Service)-[r_pub:PRODUCES_EVENTS]->(infra)
RETURN
    infra.id AS infra_id,
    infra.name AS infra_name,
    collect(DISTINCT {
        entity_id: deployed.id,
        entity_name: deployed.name,
        entity_type: deployed.type,
        chunk_ids: r_dep.source_chunk_ids
    }) AS deployed_components,
    collect(DISTINCT {
        service_id: producer.id,
        service_name: producer.name,
        chunk_ids: r_pub.source_chunk_ids
    }) AS event_producers
LIMIT $limit
"""

# T-08: Exact Canonical Entity Lookup by ID
# Fetches node properties, aliases, and metadata for a given canonical ID.
_CYPHER_ENTITY_LOOKUP_BY_ID = """
MATCH (e:Entity {id: $entity_id})
RETURN
    e.id AS entity_id,
    e.name AS name,
    e.type AS entity_type,
    e.description AS description,
    e.aliases AS aliases,
    labels(e) AS labels,
    e.created_at AS created_at,
    e.updated_at AS updated_at
LIMIT 1
"""


# ---------------------------------------------------------------------------
# Template Specification Data Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CypherTemplateSpec:
    """Specification and metadata for an approved parameterized Cypher query template.

    Attributes
    ----------
    template_id:
        Unique alphanumeric template identifier (e.g. 'T-01').
    name:
        Human-readable descriptor of the query template.
    description:
        Detailed explanation of traversal intent and usage.
    cypher:
        Audited, parameterized Cypher query string.
    required_parameters:
        Tuple of parameter keys that MUST be provided when invoking the query.
    optional_parameters:
        Tuple of parameter keys that have default fallback values (e.g., 'limit').
    default_limit:
        Default maximum records returned if not specified.
    max_limit:
        Hard upper-bound for limit parameter to prevent runaway memory usage.
    """

    template_id: str
    name: str
    description: str
    cypher: str
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...] = ("limit",)
    default_limit: int = 50
    max_limit: int = 200

    def validate_and_build_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate input parameters against specification and enforce safety boundaries.

        Parameters
        ----------
        params:
            Raw parameter mapping provided by caller.

        Returns
        -------
        dict[str, Any]
            Validated and sanitized parameter dictionary safe for Neo4j driver execution.

        Raises
        ------
        ValueError:
            If a required parameter is missing, empty, or an invalid limit is supplied.
        """
        validated: dict[str, Any] = {}

        # 1. Validate required parameters
        for req in self.required_parameters:
            if req not in params or params[req] is None:
                raise ValueError(
                    f"Missing required parameter '{req}' for Cypher template '{self.template_id}' ({self.name})"
                )
            val = params[req]
            if isinstance(val, str) and not val.strip():
                raise ValueError(
                    f"Parameter '{req}' cannot be empty string for Cypher template '{self.template_id}'"
                )
            validated[req] = val

        # 2. Validate and enforce limit
        if "limit" in params and params["limit"] is not None:
            try:
                limit_val = int(params["limit"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid 'limit' value {params['limit']!r}: must be an integer"
                ) from exc

            if limit_val < 1:
                raise ValueError(f"'limit' must be >= 1, got {limit_val}")
            # Clamp limit to max_limit
            validated["limit"] = min(limit_val, self.max_limit)
        else:
            validated["limit"] = self.default_limit

        # 3. Copy any additional supported optional parameters
        for opt in self.optional_parameters:
            if opt != "limit" and opt in params:
                validated[opt] = params[opt]

        return validated


# ---------------------------------------------------------------------------
# Centralized Template Catalog Registry
# ---------------------------------------------------------------------------


class CypherTemplateCatalog:
    """Audited catalog registry managing approved parameterized Cypher query templates."""

    _TEMPLATES: Final[dict[str, CypherTemplateSpec]] = {
        TEMPLATE_DIRECT_NEIGHBORHOOD: CypherTemplateSpec(
            template_id=TEMPLATE_DIRECT_NEIGHBORHOOD,
            name="Direct Entity Neighborhood",
            description="Retrieves 1-hop connected entities, edge types, descriptions, and provenance chunk IDs.",
            cypher=_CYPHER_DIRECT_NEIGHBORHOOD.strip(),
            required_parameters=("entity_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_MULTI_HOP_DEPENDENCY: CypherTemplateSpec(
            template_id=TEMPLATE_MULTI_HOP_DEPENDENCY,
            name="Multi-Hop Dependency Path",
            description="Traces 1 to 3 hop directional DEPENDS_ON and CALLS_API paths from a source entity.",
            cypher=_CYPHER_MULTI_HOP_DEPENDENCY.strip(),
            required_parameters=("entity_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_TEAM_OWNERSHIP_SERVICES: CypherTemplateSpec(
            template_id=TEMPLATE_TEAM_OWNERSHIP_SERVICES,
            name="Team Ownership & Service Impact",
            description="Finds all services owned by a team, their exposed APIs, and active security vulnerabilities.",
            cypher=_CYPHER_TEAM_OWNERSHIP_SERVICES.strip(),
            required_parameters=("team_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_VULNERABILITY_BLAST_RADIUS: CypherTemplateSpec(
            template_id=TEMPLATE_VULNERABILITY_BLAST_RADIUS,
            name="Vulnerability Blast Radius",
            description="Identifies all services directly or transitively impacted by a security vulnerability.",
            cypher=_CYPHER_VULNERABILITY_BLAST_RADIUS.strip(),
            required_parameters=("vuln_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_SERVICE_DEPENDENCY_CHAIN: CypherTemplateSpec(
            template_id=TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
            name="Service Dependency Chain",
            description="Retrieves downstream service dependency chains up to 3 hops deep.",
            cypher=_CYPHER_SERVICE_DEPENDENCY_CHAIN.strip(),
            required_parameters=("service_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_DATABASE_PRODUCERS_CONSUMERS: CypherTemplateSpec(
            template_id=TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
            name="Database Producers & Consumers",
            description="Finds all services that read from or write to a specific database.",
            cypher=_CYPHER_DATABASE_PRODUCERS_CONSUMERS.strip(),
            required_parameters=("database_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS: CypherTemplateSpec(
            template_id=TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
            name="Infrastructure Deployments & Events",
            description="Finds services/databases deployed on or publishing events to an infrastructure resource.",
            cypher=_CYPHER_INFRASTRUCTURE_DEPLOYMENTS.strip(),
            required_parameters=("infra_id",),
            default_limit=50,
            max_limit=200,
        ),
        TEMPLATE_ENTITY_LOOKUP_BY_ID: CypherTemplateSpec(
            template_id=TEMPLATE_ENTITY_LOOKUP_BY_ID,
            name="Exact Entity Lookup",
            description="Fetches properties, metadata, and aliases for a single canonical entity ID.",
            cypher=_CYPHER_ENTITY_LOOKUP_BY_ID.strip(),
            required_parameters=("entity_id",),
            default_limit=1,
            max_limit=1,
        ),
    }

    @classmethod
    def get_template(cls, template_id: str) -> CypherTemplateSpec:
        """Retrieve an approved template specification by its template ID.

        Parameters
        ----------
        template_id:
            Identifier matching one of the approved catalog constants (e.g. 'T-01').

        Returns
        -------
        CypherTemplateSpec
            The validated template specification.

        Raises
        ------
        KeyError:
            If the requested template ID is not present in the catalog.
        """
        template_id_clean = template_id.strip().upper()
        if template_id_clean not in cls._TEMPLATES:
            raise KeyError(
                f"Unknown Cypher template ID '{template_id}'. Approved templates: {cls.list_template_ids()}"
            )
        return cls._TEMPLATES[template_id_clean]

    @classmethod
    def list_templates(cls) -> list[CypherTemplateSpec]:
        """Return all approved template specifications in the catalog."""
        return list(cls._TEMPLATES.values())

    @classmethod
    def list_template_ids(cls) -> list[str]:
        """Return the list of all approved template IDs."""
        return sorted(cls._TEMPLATES.keys())

    @classmethod
    def has_template(cls, template_id: str) -> bool:
        """Check whether a template ID exists in the catalog."""
        return template_id.strip().upper() in cls._TEMPLATES

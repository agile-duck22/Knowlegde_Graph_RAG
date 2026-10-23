"""Parameterized Cypher Query Templates for Scientific Paper Knowledge Graph (Phase E).

Audited, parameterized Cypher templates adhering to the scientific ontology:
Nodes: :Paper, :Author, :Method (all inherit :Entity)
Relationships: :AUTHORED (Author -> Paper), :USES_METHOD (Paper -> Method)

Design Constraints:
1. Zero dynamic Cypher: all queries are module-level parameterized string constants.
2. Allowlisted labels and relationships (:Entity, :Paper, :Author, :Method; AUTHORED, USES_METHOD).
3. Parameterized execution with bounded LIMIT.
4. Provenance preservation: always return source_chunk_ids for dual-store alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Template Identifiers
TEMPLATE_FIND_PAPER_BY_ID_OR_NAME: Final[str] = "SCI-T01"
TEMPLATE_FIND_AUTHORS_OF_PAPER: Final[str] = "SCI-T02"
TEMPLATE_FIND_PAPERS_BY_AUTHOR: Final[str] = "SCI-T03"
TEMPLATE_FIND_COAUTHORS: Final[str] = "SCI-T04"
TEMPLATE_FIND_METHODS_OF_PAPER: Final[str] = "SCI-T05"
TEMPLATE_FIND_PAPERS_BY_METHOD: Final[str] = "SCI-T06"
TEMPLATE_TRAVERSE_PAPER_NEIGHBORHOOD: Final[str] = "SCI-T07"

# Approved Parameterized Cypher Queries

# SCI-T01: Find Paper by ID or Name
_CYPHER_FIND_PAPER_BY_ID_OR_NAME = """
MATCH (p:Entity:Paper)
WHERE p.id = $paper_id OR p.name = $paper_name OR toLower(p.name) = toLower($paper_name)
RETURN
    p.id AS entity_id,
    p.name AS name,
    p.type AS type,
    p.description AS description,
    labels(p) AS labels
LIMIT $limit
"""

# SCI-T02: Find Authors of a Paper (returns AUTHORED edges with provenance)
_CYPHER_FIND_AUTHORS_OF_PAPER = """
MATCH (a:Entity:Author)-[r:AUTHORED]->(p:Entity:Paper)
WHERE p.id = $paper_id OR p.name = $paper_name OR toLower(p.name) = toLower($paper_name)
RETURN
    p.id AS paper_id,
    p.name AS paper_name,
    type(r) AS rel_type,
    r.source_chunk_ids AS source_chunk_ids,
    a.id AS author_id,
    a.name AS author_name,
    labels(a) AS author_labels
LIMIT $limit
"""

# SCI-T03: Find Papers by an Author
_CYPHER_FIND_PAPERS_BY_AUTHOR = """
MATCH (a:Entity:Author)-[r:AUTHORED]->(p:Entity:Paper)
WHERE a.id = $author_id OR a.name = $author_name OR toLower(a.name) = toLower($author_name)
RETURN
    a.id AS author_id,
    a.name AS author_name,
    type(r) AS rel_type,
    r.source_chunk_ids AS source_chunk_ids,
    p.id AS paper_id,
    p.name AS paper_name,
    labels(p) AS paper_labels
LIMIT $limit
"""

# SCI-T04: Find Co-authors (Author -> Paper <- Author2)
_CYPHER_FIND_COAUTHORS = """
MATCH (a1:Entity:Author)-[r1:AUTHORED]->(p:Entity:Paper)<-[r2:AUTHORED]-(a2:Entity:Author)
WHERE (a1.id = $author_id OR a1.name = $author_name OR toLower(a1.name) = toLower($author_name))
  AND a1.id <> a2.id
RETURN
    a1.id AS source_author_id,
    a1.name AS source_author_name,
    p.id AS shared_paper_id,
    p.name AS shared_paper_name,
    r1.source_chunk_ids AS r1_chunk_ids,
    r2.source_chunk_ids AS r2_chunk_ids,
    a2.id AS coauthor_id,
    a2.name AS coauthor_name
LIMIT $limit
"""

# SCI-T05: Find Methods associated with a Paper
_CYPHER_FIND_METHODS_OF_PAPER = """
MATCH (p:Entity:Paper)-[r:USES_METHOD]->(m:Entity:Method)
WHERE p.id = $paper_id OR p.name = $paper_name OR toLower(p.name) = toLower($paper_name)
RETURN
    p.id AS paper_id,
    p.name AS paper_name,
    type(r) AS rel_type,
    r.source_chunk_ids AS source_chunk_ids,
    m.id AS method_id,
    m.name AS method_name,
    labels(m) AS method_labels
LIMIT $limit
"""

# SCI-T06: Find Papers by Method
_CYPHER_FIND_PAPERS_BY_METHOD = """
MATCH (p:Entity:Paper)-[r:USES_METHOD]->(m:Entity:Method)
WHERE m.id = $method_id OR m.name = $method_name OR toLower(m.name) = toLower($method_name)
RETURN
    m.id AS method_id,
    m.name AS method_name,
    type(r) AS rel_type,
    r.source_chunk_ids AS source_chunk_ids,
    p.id AS paper_id,
    p.name AS paper_name,
    labels(p) AS paper_labels
LIMIT $limit
"""

# SCI-T07: Traverse Paper 1-hop / 2-hop neighborhood
_CYPHER_TRAVERSE_PAPER_NEIGHBORHOOD = """
MATCH (p:Entity:Paper)-[r]-(neighbor:Entity)
WHERE p.id = $paper_id OR p.name = $paper_name OR toLower(p.name) = toLower($paper_name)
RETURN
    p.id AS paper_id,
    p.name AS paper_name,
    type(r) AS rel_type,
    r.source_chunk_ids AS source_chunk_ids,
    neighbor.id AS neighbor_id,
    neighbor.name AS neighbor_name,
    labels(neighbor) AS neighbor_labels
LIMIT $limit
"""


@dataclass(frozen=True)
class ScientificCypherTemplateSpec:
    """Specification for an audited scientific Cypher query template."""

    template_id: str
    name: str
    purpose: str
    cypher: str
    required_params: tuple[str, ...]
    default_limit: int = 50


class ScientificCypherTemplateCatalog:
    """Catalog of audited scientific Cypher templates."""

    _CATALOG: dict[str, ScientificCypherTemplateSpec] = {
        TEMPLATE_FIND_PAPER_BY_ID_OR_NAME: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_FIND_PAPER_BY_ID_OR_NAME,
            name="Find Paper by ID or Name",
            purpose="Lookup paper metadata by identifier or title",
            cypher=_CYPHER_FIND_PAPER_BY_ID_OR_NAME,
            required_params=("paper_id", "paper_name"),
            default_limit=20,
        ),
        TEMPLATE_FIND_AUTHORS_OF_PAPER: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_FIND_AUTHORS_OF_PAPER,
            name="Find Authors of Paper",
            purpose="Retrieve authors and AUTHORED provenance chunks for a paper",
            cypher=_CYPHER_FIND_AUTHORS_OF_PAPER,
            required_params=("paper_id", "paper_name"),
            default_limit=50,
        ),
        TEMPLATE_FIND_PAPERS_BY_AUTHOR: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_FIND_PAPERS_BY_AUTHOR,
            name="Find Papers by Author",
            purpose="Retrieve papers authored by a specific researcher",
            cypher=_CYPHER_FIND_PAPERS_BY_AUTHOR,
            required_params=("author_id", "author_name"),
            default_limit=50,
        ),
        TEMPLATE_FIND_COAUTHORS: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_FIND_COAUTHORS,
            name="Find Coauthors",
            purpose="Retrieve researchers who coauthored papers with a target author",
            cypher=_CYPHER_FIND_COAUTHORS,
            required_params=("author_id", "author_name"),
            default_limit=50,
        ),
        TEMPLATE_FIND_METHODS_OF_PAPER: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_FIND_METHODS_OF_PAPER,
            name="Find Methods of Paper",
            purpose="Retrieve methods and USES_METHOD provenance chunks for a paper",
            cypher=_CYPHER_FIND_METHODS_OF_PAPER,
            required_params=("paper_id", "paper_name"),
            default_limit=50,
        ),
        TEMPLATE_FIND_PAPERS_BY_METHOD: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_FIND_PAPERS_BY_METHOD,
            name="Find Papers by Method",
            purpose="Retrieve papers that utilize a specific scientific method",
            cypher=_CYPHER_FIND_PAPERS_BY_METHOD,
            required_params=("method_id", "method_name"),
            default_limit=50,
        ),
        TEMPLATE_TRAVERSE_PAPER_NEIGHBORHOOD: ScientificCypherTemplateSpec(
            template_id=TEMPLATE_TRAVERSE_PAPER_NEIGHBORHOOD,
            name="Traverse Paper Neighborhood",
            purpose="Retrieve 1-hop connected entities and provenance edges for a paper",
            cypher=_CYPHER_TRAVERSE_PAPER_NEIGHBORHOOD,
            required_params=("paper_id", "paper_name"),
            default_limit=50,
        ),
    }

    @classmethod
    def get(cls, template_id: str) -> ScientificCypherTemplateSpec:
        """Get template spec by template ID, or raise KeyError."""
        if template_id not in cls._CATALOG:
            raise KeyError(f"Unknown scientific Cypher template: '{template_id}'")
        return cls._CATALOG[template_id]

    @classmethod
    def list_templates(cls) -> list[ScientificCypherTemplateSpec]:
        """List all approved scientific Cypher templates."""
        return list(cls._CATALOG.values())

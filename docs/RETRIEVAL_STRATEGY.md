# Retrieval & Routing Strategy

This document defines the query classification taxonomy, entity linking, approved parameterized Cypher templates, vector search parameters, and hybrid context ranking strategy.

---

## 1. Query Routing Taxonomy & Intent Classification

Incoming user queries are evaluated by the **Intent Router** (`src/routing/intent_classifier.py`) and classified into one of four execution modes:

```
                          ┌──────────────────────────┐
                          │   Incoming User Query    │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Intent Classifier (LLM)  │
                          └─────────────┬────────────┘
                                        │
         ┌──────────────────┬───────────┴──────────┬──────────────────┐
         │                  │                      │                  │
         ▼                  ▼                      ▼                  ▼
┌─────────────────┐┌─────────────────┐  ┌──────────────────┐┌──────────────────┐
│   GRAPH_ONLY    ││   VECTOR_ONLY   │  │      HYBRID      ││   OUT_OF_SCOPE   │
│ (Multi-hop,     ││ (Semantic facts,│  │(Ambiguous, broad,││ (General gossip, │
│  dependencies,  ││  definitions,   │  │ multi-aspect, or ││  unsupported     │
│  aggregations)  ││  isolated text) │  │ low-confidence)  ││  domains)        │
└─────────────────┘└─────────────────┘  └──────────────────┘└──────────────────┘
```

### Classification Rules & Decision Matrix

| Query Pattern | Example Question | Selected Mode | Primary Engine |
| :--- | :--- | :--- | :--- |
| **Multi-Hop Dependency** | *"Which services depend on a service owned by the Checkout team?"* | `GRAPH_ONLY` | Neo4j Cypher Traversal |
| **Direct Semantic Lookup** | *"What is the retry policy described in the payment gateway spec?"* | `VECTOR_ONLY` | PostgreSQL pgvector HNSW |
| **Relationship Aggregation** | *"List all databases that are written to by services vulnerable to CVE-2024-21626"* | `GRAPH_ONLY` | Neo4j Cypher Aggregation |
| **Broad / Architecture Question** | *"Describe the payment architecture and which teams are responsible for it"* | `HYBRID` | Neo4j + pgvector Combined |
| **Low Routing Confidence ($< 0.65$)** | *"Tell me about checkout and postgres"* | `HYBRID` | Fallback Dual Retrieval |
| **Out-of-Scope / Non-Enterprise** | *"What is the weather in Tokyo?"* | `OUT_OF_SCOPE` | Immediate Refusal |

---

## 2. Approved Parameterized Cypher Query Templates

> [!IMPORTANT]
> Zero raw model-generated Cypher is permitted. All graph retrievals select from this audited template library.

### Template 1: Direct Entity Neighborhood (`T-01`)
*Retrieves 1-hop connected entities, relationships, and supporting chunk IDs.*
```cypher
MATCH (e:Entity {id: $entity_id})-[r]-(neighbor:Entity)
RETURN e.id AS source_id, e.name AS source_name,
       type(r) AS rel_type, r.description AS rel_desc,
       r.source_chunk_ids AS source_chunk_ids,
       neighbor.id AS target_id, neighbor.name AS target_name,
       labels(neighbor) AS target_labels
LIMIT $limit
```

### Template 2: Multi-Hop Dependency Path (`T-02`)
*Traces 2-to-3 hop directional dependency paths between services and downstream resources.*
```cypher
MATCH path = (e:Entity {id: $entity_id})-[:DEPENDS_ON|CALLS_API*1..3]->(target:Entity)
WITH path, relationships(path) AS rels, nodes(path) AS nds
UNWIND rels AS r
RETURN [n IN nds | {id: n.id, name: n.name, type: n.type}] AS path_nodes,
       type(r) AS rel_type,
       r.source_chunk_ids AS source_chunk_ids
LIMIT $limit
```

### Template 3: Team Ownership & Service Impact (`T-03`)
*Finds all services owned by a team and their active vulnerabilities.*
```cypher
MATCH (t:Team {id: $team_id})<-[:OWNED_BY]-(s:Service)
OPTIONAL MATCH (s)-[:AFFECTED_BY]->(v:Vulnerability)
RETURN t.name AS team_name, s.id AS service_id, s.name AS service_name,
       collect(DISTINCT v.id) AS vulnerability_ids
LIMIT $limit
```

### Template 4: Vulnerability Blast Radius (`T-04`)
*Identifies all services directly or transitively depending on a vulnerable component.*
```cypher
MATCH (v:Vulnerability {id: $vuln_id})<-[:AFFECTED_BY]-(impacted:Service)
OPTIONAL MATCH (dependent:Service)-[:DEPENDS_ON*1..2]->(impacted)
RETURN v.id AS vuln_id,
       impacted.name AS vulnerable_service,
       collect(DISTINCT dependent.name) AS affected_dependents
LIMIT $limit
```

---

## 3. PostgreSQL / pgvector Retrieval Strategy

### Vector Search Query
```sql
SELECT 
    c.id AS chunk_id,
    c.document_id,
    c.content,
    c.metadata,
    1 - (c.embedding <=> $query_embedding) AS similarity_score
FROM chunks c
WHERE 1 - (c.embedding <=> $query_embedding) >= $similarity_threshold
ORDER BY c.embedding <=> $query_embedding ASC
LIMIT $top_k;
```

### Tunable Vector Search Parameters
- `similarity_threshold`: default `0.65` (cosine similarity).
- `top_k`: default `5` (vector mode), `3` (hybrid mode).
- `hnsw_ef_search`: default `40` (configurable via `SET LOCAL hnsw.ef_search = 40`).

---

## 4. Context Normalization & Hybrid Ranking

When executing in `HYBRID` mode:
1. **Graph Traversal Phase**: Retrieves structured triples and supporting `chunk_id`s from Neo4j.
2. **Vector Retrieval Phase**: Retrieves top-$k$ semantic chunks from PostgreSQL.
3. **Hydration & Deduplication**:
   - Supporting `chunk_id`s from the graph phase are fetched from PostgreSQL `chunks` table.
   - Chunks appearing in both graph and vector results are deduplicated and prioritized.
4. **Context Formatting**:
   - Structured graph relationships are formatted as Markdown relationship tables.
   - Text chunks are formatted as distinct `<chunk id="...">...</chunk>` blocks.
5. **Token Budget Enforcement**: Context is truncated safely to fit within `MAX_CONTEXT_TOKENS` (default 6,000 tokens).

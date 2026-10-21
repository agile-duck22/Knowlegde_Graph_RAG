# Data Model & Storage Contracts

This document defines the storage schemas, database contracts, indexes, and constraints for both Neo4j and PostgreSQL/pgvector.

---

## 1. Dual-Store Invariant & ID Mapping

```
┌────────────────────────────────────────────────────────┐
│                   PostgreSQL Database                  │
│  TABLE chunks (                                        │
│      id UUID PRIMARY KEY, ──┐                          │
│      document_id UUID,      │                          │
│      content TEXT,          │                          │
│      embedding VECTOR(1536) │                          │
│  )                          │                          │
└─────────────────────────────┼──────────────────────────┘
                              │ Shared Canonical
                              │ UUID chunk_id
┌─────────────────────────────┼──────────────────────────┐
│                   Neo4j Graph Database                 │
│  (:Entity {id: '...' })     │                          │
│       │                     │                          │
│  -[r:DEPENDS_ON { ──────────┘                          │
│      source_chunk_ids: ['a1b2c3...'],                  │
│      confidence: 0.95                                  │
│  }]->                                                  │
│  (:Entity {id: '...' })                                │
└────────────────────────────────────────────────────────┘
```

---

## 2. PostgreSQL / pgvector Schema

### Table: `documents`
Stores metadata and checksums for ingested documents.

```sql
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uri VARCHAR(1024) NOT NULL UNIQUE,
    title VARCHAR(512) NOT NULL,
    doc_type VARCHAR(64) NOT NULL DEFAULT 'technical_doc',
    content_hash CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
```

### Table: `chunks`
Stores raw chunk text, metadata, and dense vector embeddings.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),         -- 1536-dim vector (OpenAI text-embedding-3-small)
    embedding_384 VECTOR(384),       -- 384-dim vector (local sentence-transformers all-MiniLM-L6-v2)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
);

-- HNSW Cosine Distance Vector Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_384_hnsw
ON chunks USING hnsw (embedding_384 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
```

### Table: `entity_links`
Relational bridge between PostgreSQL chunks and extracted canonical entity identifiers.

```sql
CREATE TABLE IF NOT EXISTS entity_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    canonical_entity_id VARCHAR(256) NOT NULL,
    entity_type VARCHAR(64) NOT NULL,
    mention_text VARCHAR(256) NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entity_links_chunk_id ON entity_links(chunk_id);
CREATE INDEX IF NOT EXISTS idx_entity_links_canonical_id ON entity_links(canonical_entity_id);
```

---

## 3. Neo4j Property Graph Schema

### Node Labels and Properties

All nodes implement the primary label `Entity` plus their specific domain label (e.g., `Service`, `Team`, `API`, `Database`, `Vulnerability`, `Repository`).

```
(e:Entity:Service {
    id: "service:payment-gateway",        // Canonical ID: type:normalized_name
    name: "Payment Gateway",               // Display name
    type: "Service",                       // Entity type string
    description: "Processes card and ACH payments",
    aliases: ["payment-svc", "pay-gw"],   // Normalization aliases
    created_at: datetime(),
    updated_at: datetime()
})
```

### Relationship Types and Properties

```
(source:Entity)-[r:DEPENDS_ON {
    relation_type: "DEPENDS_ON",
    description: "Synchronous gRPC calls during checkout",
    confidence: 0.95,
    source_chunk_ids: ["a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"],
    created_at: datetime(),
    updated_at: datetime()
}]->(target:Entity)
```

### Neo4j Uniqueness Constraints & Indexes

```cypher
// Canonical Entity ID Uniqueness Constraint
CREATE CONSTRAINT unique_entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Fulltext Search Index on Name and Aliases
CREATE FULLTEXT INDEX entity_text_index IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.description];
```

---

## 4. Ingestion Consistency, Idempotency & Reconciliation

1. **Upsert Logic**:
   - In PostgreSQL: `INSERT ... ON CONFLICT (document_id, chunk_index) DO UPDATE ...`
   - In Neo4j: `MERGE (e:Entity {id: $id}) ON CREATE SET ... ON MATCH SET ...`
2. **Re-ingestion Behavior**:
   - If a document is updated, existing chunks in PostgreSQL are updated and obsolete entity links are purged via foreign key cascade.
   - Neo4j edges append the new `chunk_id` to `source_chunk_ids` or update timestamps. If all supporting `source_chunk_ids` for an edge are deleted, a reconciliation background query removes orphaned edges.

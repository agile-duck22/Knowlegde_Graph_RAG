# Proposed Migration Plan: Full Corpus Re-Ingestion (Phase H Proposal)

> **Document Status**: Proposed & Awaiting Explicit User Authorization  
> **Target Scope**: Remaining 119 scientific papers (or full 120-paper corpus idempotently)  
> **Safety Boundary**: Zero execution without explicit user sign-off. Strict isolation to PostgreSQL port `5433` and Neo4j port `7688`. `ALLOW_PAID_EMBEDDINGS=false`. Zero remote API calls.

---

## 1. Context & Objectives

In Phase G, single-paper integration on `paper-arxiv-2411.18583` verified that the deterministic `OfflineScientificExtractor` successfully extracts and writes 8 ontology entity types and 7 relationship types into Neo4j and PostgreSQL without error, without duplicates, and without regressing the Phase E hybrid retrieval benchmark.

The purpose of this migration plan is to describe the safe, reproducible, and verifiable re-ingestion of the remaining 119 papers in the corpus (`experiments/scipaper/data/corpus_scipaper.jsonl`).

---

## 2. Schema and Constraint Preparation

### 2.1 Graph Schema (Neo4j)
- **Existing Constraint**: `CONSTRAINT unique_entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE`
  - Already active and covers all 8 scientific types (`Paper`, `Author`, `Method`, `Model`, `Dataset`, `Metric`, `Task`, `ResearchOrganization`) because all nodes carry the primary label `:Entity`.
- **Recommended Additive Indexes**:
  - `CREATE INDEX index_entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);`
  - Facilitates filtered multi-hop traversals by entity type (e.g. finding all `Dataset` or `Model` nodes connected to a paper).
- **Zero Destructive Cypher**: No dropping of existing constraints or database recreation.

### 2.2 Relational Schema (PostgreSQL)
- **Table `documents`**: 120 rows already present. Documents remain untouched or updated via `ON CONFLICT (document_id) DO UPDATE`.
- **Table `chunks`**: 474 rows already present with 384-dimensional embeddings in `embedding_384`. Chunks and embeddings are preserved; no re-chunking or re-embedding is required because chunk boundaries and IDs are deterministic.
- **Table `entity_links`**: Stores the cross-reference from text chunk to canonical entity ID. Existing links will be preserved, and new entity mentions will be inserted idempotently (`ON CONFLICT DO NOTHING`).

---

## 3. Migration Execution Architecture

### 3.1 Bounded Batching & Order
- **Corpus Size**: 120 papers total (1 paper already integrated in Phase G; 119 papers pending).
- **Batch Size**: 20 documents per batch ($\approx 80$ chunks per batch).
- **Batch Count**: 6 batches total.
- **Deterministic Ordering**: Sorted by `document_id` ascending.

### 3.2 Processing Pipeline Per Batch
1. **Document Loading**: Read from `corpus_scipaper.jsonl`.
2. **Deterministic Chunking Verification**: Generate chunks via `DocumentChunker(chunk_size=500, chunk_overlap=50)` and assert that `chunk_id` matches the existing PostgreSQL database records.
3. **Extraction**: Run `OfflineScientificExtractor(chunk)` to extract entities and relationships with chunk-level provenance.
4. **Neo4j Upsert**:
   - `write_entities()` in batches of 100 using parameterized `MERGE (e:Entity:<Type> {id: $id})`.
   - `write_relationships()` in batches of 100 using parameterized `MERGE (source)-[r:<RelType> {provenance_chunk_id: $prov}]->(target)`.
5. **PostgreSQL Entity Links Upsert**:
   - Insert new `(chunk_id, canonical_entity_id, entity_type, mention_text, confidence)` records.
6. **Per-Batch Audit & Health Check**:
   - Query node and relationship deltas.
   - Assert zero partial write errors.
   - Assert PostgreSQL chunk count remains exactly 474 and embedding count remains 474.

---

## 4. Idempotency & Rollback Strategy

### 4.1 Idempotent Write Semantics
- In Neo4j: `MERGE (e:Entity:<Type> {id: $id})` updates existing nodes on match.
- Relationships use:
  ```cypher
  MERGE (source)-[r:<TYPE> {relation_type: row.relation_type, provenance_chunk_id: row.provenance_chunk_id}]->(target)
  ON CREATE SET ...
  ON MATCH SET ...
  ```
  This guarantees that repeated runs on the same chunk produce zero duplicate edges.
- In PostgreSQL: `entity_links` uses unique constraint `(chunk_id, canonical_entity_id)` or pre-existence checks before insert.

### 4.2 Transaction Boundaries & Targeted Rollback
- Every batch runs inside explicit database transactions.
- If a batch fails:
  - Neo4j transaction rolls back automatically.
  - PostgreSQL transaction rolls back.
  - State is recorded in a migration ledger file (`migration_ledger.json`).
  - **No full database wipes**: Recovery is handled via re-running the specific failed batch.

---

## 5. Runtime & Resource Estimates

Based on the measured execution of `paper-arxiv-2411.18583` during Phase G:
- **Extraction Time**: $\approx 0.05$ seconds per chunk ($0.2$s per paper).
- **Neo4j Write Time**: $\approx 0.08$ seconds per paper.
- **PostgreSQL Entity Links Write**: $\approx 0.04$ seconds per paper.
- **Total Expected Migration Time**:
  - $119 \text{ papers} \times \approx 0.35 \text{ seconds} \approx \mathbf{42 \text{ seconds}}$.
- **Resource Footprint**: Minimal CPU / memory overhead; zero remote network traffic.

---

## 6. Verification Criteria & Stop Conditions

### Stop Conditions (Migration Halted Immediately If):
1. Target isolation failure: Any attempt to connect to ports `5432` or `7687`.
2. Model download or API invocation: Any attempt to download models or invoke external LLMs.
3. PostgreSQL count distortion: Any alteration to total document count ($120$) or chunk count ($474$).
4. Vector embedding degradation: Any nullification of `embedding_384` vectors.
5. Ingestion transaction error: Any unhandled exception during a batch write.

### Success Verification Criteria:
1. All 120 papers processed.
2. PostgreSQL: 120 documents, 474 chunks, 474 `embedding_384` vectors intact.
3. Neo4j: Nodes expanded to include `Model`, `Dataset`, `Metric`, `Task`, `ResearchOrganization`.
4. Idempotency test: Re-running batch 1 yields 0 node and 0 relationship deltas.
5. Retrieval benchmark: Re-running Phase E benchmark confirms no regression on the 50 questions.

---

## 7. Approval Checkpoint

**Current State**: 1 paper (`paper-arxiv-2411.18583`) integrated. 119 papers awaiting migration.  
**Action Required**: Await explicit user authorization before executing Phase H migration.

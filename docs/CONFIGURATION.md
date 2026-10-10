# Configuration Reference & Environment Variables

This document defines all configuration settings, environment variables, default values, and empirical tuning parameters for the system.

---

## 1. Environment Variable Reference (`.env`)

| Variable Name | Type | Default Value | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | string | `development` | No | Runtime environment (`development`, `test`, `production`) |
| `LOG_LEVEL` | string | `INFO` | No | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ANTHROPIC_API_KEY` | secret | — | **Yes** | Anthropic Claude API Key |
| `LLM_MODEL` | string | `claude-3-5-sonnet-20241022`| No | Primary LLM model for synthesis and extraction |
| `ROUTING_MODEL` | string | `claude-3-haiku-20240307` | No | Fast model for query intent routing |
| `EMBEDDING_PROVIDER` | string | `openai` | No | Embedding provider (`openai`, `huggingface`, `cohere`) |
| `EMBEDDING_MODEL` | string | `text-embedding-3-small` | No | Embedding model name |
| `EMBEDDING_DIMENSIONS`| int | `1536` | No | Vector dimension (must match PostgreSQL schema) |
| `NEO4J_URI` | string | `bolt://localhost:7687` | No | Neo4j Bolt protocol URI |
| `NEO4J_USER` | string | `neo4j` | No | Neo4j database username |
| `NEO4J_PASSWORD` | secret | `neo4j_password` | **Yes** | Neo4j database password |
| `POSTGRES_HOST` | string | `localhost` | No | PostgreSQL host address |
| `POSTGRES_PORT` | int | `5432` | No | PostgreSQL port |
| `POSTGRES_DB` | string | `kgr_db` | No | PostgreSQL database name |
| `POSTGRES_USER` | string | `kgr_user` | No | PostgreSQL username |
| `POSTGRES_PASSWORD` | secret | `kgr_password` | **Yes** | PostgreSQL password |

---

## 2. Ingestion & Chunking Parameters

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `CHUNK_SIZE` | int | `500` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | int | `50` | Overlap between consecutive chunks in tokens |
| `EXTRACTION_BATCH_SIZE` | int | `10` | Chunks processed concurrently during LLM entity extraction |
| `EXTRACTION_MAX_RETRIES`| int | `3` | Maximum retry attempts for failed LLM extraction calls |

---

## 3. Retrieval & Routing Tuning Knobs

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ROUTER_CONFIDENCE_THRESHOLD` | float | `0.65` | Confidence below which query falls back to `HYBRID` mode |
| `VECTOR_TOP_K` | int | `5` | Number of chunks returned in `VECTOR_ONLY` mode |
| `HYBRID_VECTOR_TOP_K` | int | `3` | Number of chunks returned from vector search in `HYBRID` mode |
| `VECTOR_SIMILARITY_THRESHOLD`| float | `0.60` | Minimum cosine similarity score for retrieved chunks |
| `GRAPH_MAX_HOPS` | int | `2` | Maximum graph traversal depth ($D \le 3$) |
| `GRAPH_NODE_LIMIT` | int | `25` | Maximum number of graph nodes returned per query |
| `MAX_CONTEXT_TOKENS` | int | `6000` | Maximum token limit for merged synthesis context |

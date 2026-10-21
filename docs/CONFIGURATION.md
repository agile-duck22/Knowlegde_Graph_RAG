# Configuration Reference & Environment Variables

This document defines all configuration settings, environment variables, default values, and empirical tuning parameters for the system.

---

## 1. Environment Variable Reference (`.env`)

| Variable Name | Type | Default Value | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | string | `development` | No | Runtime environment (`development`, `test`, `production`) |
| `LOG_LEVEL` | string | `INFO` | No | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LLM_PROVIDER` | string | `anthropic` | No | Default LLM provider backend (`groq`, `gemini`, `openrouter`, `anthropic`) |
| `EXTRACTION_PROVIDER` | string | `None` (falls back to `LLM_PROVIDER`) | No | Provider override for entity extraction |
| `ROUTING_PROVIDER` | string | `None` (falls back to `LLM_PROVIDER`) | No | Provider override for query intent routing |
| `SYNTHESIS_PROVIDER` | string | `None` (falls back to `LLM_PROVIDER`) | No | Provider override for answer synthesis |
| `GROQ_API_KEY` | secret | — | If `groq` | Groq API key for free-tier inference |
| `GEMINI_API_KEY` | secret | — | If `gemini` | Google Gemini API key for free-tier inference |
| `OPENROUTER_API_KEY` | secret | — | If `openrouter` | OpenRouter API key for `:free` models |
| `ANTHROPIC_API_KEY` | secret | — | If `anthropic` | Anthropic Claude API Key (optional if using free provider) |
| `LLM_MODEL` | string | `claude-3-5-sonnet-20241022`| No | Primary LLM model for synthesis and extraction |
| `ROUTING_MODEL` | string | `claude-3-haiku-20240307` | No | Model for query intent routing (e.g. `openai/gpt-oss-20b`) |
| `EXTRACTION_MODEL` | string | `None` (falls back to `LLM_MODEL`) | No | Separate model override for entity extraction (e.g. `gemini-3.1-flash-lite`) |
| `SYNTHESIS_MODEL` | string | `None` (falls back to `LLM_MODEL`) | No | Separate model override for answer synthesis (e.g. `gemini-3.1-flash-lite`) |
| `EMBEDDING_PROVIDER` | string | `openai` | No | Embedding provider (`openai`, `sentence_transformers`, `huggingface`, `local`, `cohere`) |
| `EMBEDDING_MODEL` | string | `text-embedding-3-small` | No | Embedding model name (e.g. `all-MiniLM-L6-v2` for zero-cost local, `text-embedding-3-small` for OpenAI) |
| `EMBEDDING_DIMENSIONS`| int | `1536` | No | Vector dimension matching model output (384 for `all-MiniLM-L6-v2`, 1536 for OpenAI `text-embedding-3-small`). Must match PostgreSQL schema |
| `ALLOW_PAID_EMBEDDINGS` | bool | `False` | No | Explicit authorization flag required for paid OpenAI embedding generation |
| `NEO4J_URI` | string | `bolt://localhost:7687` | No | Neo4j Bolt protocol URI |
| `NEO4J_USER` | string | `neo4j` | No | Neo4j database username |
| `NEO4J_PASSWORD` | secret | `neo4j_password` | **Yes** | Neo4j database password |
| `POSTGRES_HOST` | string | `localhost` | No | PostgreSQL host address |
| `POSTGRES_PORT` | int | `5432` | No | PostgreSQL port |
| `POSTGRES_DB` | string | `kgr_db` | No | PostgreSQL database name |
| `POSTGRES_USER` | string | `kgr_user` | No | PostgreSQL username |
| `POSTGRES_PASSWORD` | secret | `kgr_password` | **Yes** | PostgreSQL password |

---

## 1.1 Multi-Provider LLM & Free-Tier Constraints

The system supports free-tier LLM inference via an OpenAI-compatible adapter without requiring external SDK installations:

| Provider | Canonical Base URL | Configured Allowlist Models | Requirements |
| :--- | :--- | :--- | :--- |
| **Groq** | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | Set `GROQ_API_KEY`. Free-tier inference via open reasoning and versatile models. Fail-closed on unauthorized model IDs. |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`, `gemini-1.5-pro` | Set `GEMINI_API_KEY`. High rate-limit free tier via Google AI Studio. Fail-closed on unauthorized model IDs. |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3.5-lightning:free`, `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`, `mistralai/mistral-7b-instruct:free` (all models must explicitly end with `:free`) | Set `OPENROUTER_API_KEY`. Strict `:free` suffix check enforced (fails closed without `:free`). |
| **Anthropic** | Native SDK | `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`, `claude-3-opus-20240229` | Set `ANTHROPIC_API_KEY`. Paid API. Never fallen back to automatically. |

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

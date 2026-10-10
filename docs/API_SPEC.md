# REST API Specification (OpenAPI Contract)

This document defines the REST API exposed by the FastAPI gateway for the Knowledge Graph RAG platform.

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Basic liveness check | No |
| `GET` | `/ready` | Deep readiness check verifying Neo4j and PostgreSQL connectivity | No |
| `POST` | `/api/v1/query` | Primary RAG question answering endpoint | Bearer (Prod) |
| `POST` | `/api/v1/ingest` | Document ingestion endpoint (MVP batch processing) | Bearer (Prod) |

---

## 2. Endpoint Specifications

### `GET /health`
Liveness probe indicating the HTTP service is running.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-10-09T12:00:00Z"
}
```

---

### `GET /ready`
Readiness probe verifying active connections to Neo4j and PostgreSQL.

#### Response `200 OK`
```json
{
  "status": "ready",
  "databases": {
    "neo4j": "connected",
    "postgres": "connected"
  },
  "timestamp": "2026-10-09T12:00:00Z"
}
```

#### Response `503 Service Unavailable`
```json
{
  "status": "unavailable",
  "databases": {
    "neo4j": "disconnected",
    "postgres": "connected"
  },
  "error": "Failed to connect to Neo4j database"
}
```

---

### `POST /api/v1/query`
Executes hybrid retrieval, answer synthesis, and citation verification.

#### Request Body
```json
{
  "question": "Which services depend on a component maintained by the Checkout team?",
  "retrieval_override": null,  // Optional: "GRAPH_ONLY" | "VECTOR_ONLY" | "HYBRID"
  "top_k": 5,                  // Optional: 1 to 20 (default: 5)
  "include_provenance": true   // Optional: return retrieved chunks and graph triples
}
```

#### Response `200 OK`
```json
{
  "request_id": "req-98765432-abcd",
  "question": "Which services depend on a component maintained by the Checkout team?",
  "answer": "The Checkout Service is maintained by the Checkout Team [chunk-1111]. The Payment Gateway depends directly on the Checkout Service for order validation [chunk-2222].",
  "retrieval_mode": "HYBRID",
  "confidence": 0.92,
  "citations": [
    {
      "chunk_id": "chunk-1111",
      "document_id": "doc-checkout-arch",
      "excerpt": "The Checkout Service is owned and operated by the Checkout Team..."
    },
    {
      "chunk_id": "chunk-2222",
      "document_id": "doc-payment-gateway-spec",
      "excerpt": "Payment Gateway makes synchronous gRPC calls to the Checkout Service..."
    }
  ],
  "graph_paths": [
    {
      "source": "team:checkout-engineering",
      "relation": "OWNED_BY (inverse)",
      "target": "service:checkout-service"
    },
    {
      "source": "service:payment-gateway",
      "relation": "DEPENDS_ON",
      "target": "service:checkout-service"
    }
  ],
  "execution_time_ms": 1420.5,
  "token_usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 180,
    "total_tokens": 1430,
    "estimated_cost_usd": 0.00645
  },
  "warnings": []
}
```

#### Response `400 Bad Request`
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Question cannot exceed 1000 characters.",
  "request_id": "req-98765432-abcd"
}
```

---

## 3. Error Codes & Handling

| HTTP Status | Error Code | Description |
| :--- | :--- | :--- |
| `400` | `INVALID_INPUT` | Malformed JSON or input validation constraint failed |
| `422` | `UNPROCESSABLE_ENTITY` | Pydantic model validation failure |
| `429` | `RATE_LIMIT_EXCEEDED` | Request rate limit exceeded |
| `500` | `INTERNAL_SERVER_ERROR` | Unhandled internal exception (secrets sanitized) |
| `503` | `SERVICE_UNAVAILABLE` | Database or upstream LLM provider failure |

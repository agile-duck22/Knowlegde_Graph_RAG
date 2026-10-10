# Observability, Structured Logging & Metrics

This document specifies the logging formats, telemetry middleware, metrics collection, and sensitive data redaction policies.

---

## 1. Structured Logging Specification

All application logs must be emitted as single-line JSON objects to standard output (`stdout`) to facilitate ingestion by log collectors (e.g., Datadog, Grafana Loki, CloudWatch).

### JSON Log Schema
```json
{
  "timestamp": "2026-10-09T12:00:00.123Z",
  "level": "INFO",
  "logger": "src.routing.router",
  "request_id": "req-98765432-abcd",
  "message": "Query routed successfully",
  "event": "query_routed",
  "retrieval_mode": "HYBRID",
  "routing_confidence": 0.94,
  "execution_duration_ms": 42.1,
  "environment": "development"
}
```

---

## 2. Redaction & Data Sanitization Rules

The custom logging formatter (`src/observability/logging.py`) enforces automatic redaction of:
1. **API Keys & Authorization Headers**: Values matching `Bearer .*`, `sk-ant-.*`, or `ANTHROPIC_API_KEY`.
2. **Database Credentials**: Connection strings containing passwords (`bolt://neo4j:***@...`, `postgresql://user:***@...`).
3. **Full Raw Document Text**: Logs may contain `chunk_id` and token counts, but must never dump full raw document bodies to general info logs.

---

## 3. Telemetry Metrics

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `rag_query_requests_total` | Counter | `mode`, `status_code` | Total incoming query requests |
| `rag_query_duration_seconds` | Histogram | `mode` | End-to-end query response latency |
| `rag_retrieval_duration_seconds` | Histogram | `engine` (`neo4j` \| `pgvector`) | Duration spent in database retrieval |
| `rag_token_usage_total` | Counter | `model`, `token_type` (`prompt` \| `completion`) | Cumulative token consumption |
| `rag_citation_validation_failures_total` | Counter | `failure_type` | Total detected hallucinated citations |
| `rag_ingest_documents_total` | Counter | `status` | Total documents processed during ingestion |

---

## 4. Token & Cost Tracking Middleware

Every LLM call to Anthropic records token usage. The request telemetry middleware computes estimated query cost:

$$\text{Cost} = (\text{Prompt Tokens} \times \$3.00 / 1\text{M}) + (\text{Completion Tokens} \times \$15.00 / 1\text{M})$$
*(Calculated using standard Claude 3.5 Sonnet pricing tiers; configurable in settings).*

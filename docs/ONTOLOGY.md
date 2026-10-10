# Domain Ontology Specification: Enterprise Architecture & Software Systems

This document defines the formal entity types, relationship types, property schemas, canonical normalization rules, and resolution test cases for the enterprise architecture reference corpus.

---

## 1. Entity Types (Nodes)

The ontology defines **7 core entity types**:

| Entity Type | Description | Canonical ID Format | Example Instance |
| :--- | :--- | :--- | :--- |
| `Service` | A deployed software application or microservice | `service:<kebab-name>` | `service:payment-gateway` |
| `Team` | An engineering or business ownership team | `team:<kebab-name>` | `team:checkout-engineering` |
| `API` | A defined REST, gRPC, or GraphQL interface | `api:<kebab-name>` | `api:process-charge-v2` |
| `Database` | A persistent datastore or cache | `database:<kebab-name>` | `database:customer-orders-pg` |
| `Repository` | A source code repository or mono-repo | `repo:<kebab-name>` | `repo:payment-service-repo` |
| `Vulnerability` | A security CVE or internal security advisory | `vuln:<cve-id>` | `vuln:cve-2024-21626` |
| `Infrastructure` | Cloud resources, clusters, or message brokers | `infra:<kebab-name>` | `infra:kafka-event-bus` |

---

## 2. Relationship Types (Edges)

The ontology defines **10 directed relationship types**:

| Relationship | Source Label | Target Label | Description | Cardinality |
| :--- | :--- | :--- | :--- | :--- |
| `OWNED_BY` | `Service` \| `Repository` | `Team` | Identifies team ownership and operational responsibility | $N:1$ |
| `DEPENDS_ON` | `Service` | `Service` | Downstream runtime service dependency | $N:M$ |
| `EXPOSES_API` | `Service` | `API` | Service exposes an API contract | $1:N$ |
| `CALLS_API` | `Service` | `API` | Service invokes a remote API contract | $N:M$ |
| `READS_FROM` | `Service` | `Database` | Service performs read queries against database | $N:M$ |
| `WRITES_TO` | `Service` | `Database` | Service performs write transactions against database | $N:M$ |
| `STORES_CODE` | `Repository` | `Service` | Repository contains source code for service | $1:N$ |
| `AFFECTED_BY` | `Service` \| `Repository` | `Vulnerability` | Service is impacted by a security vulnerability | $N:M$ |
| `DEPLOYED_ON` | `Service` \| `Database` | `Infrastructure` | Component is deployed on infrastructure resource | $N:M$ |
| `PRODUCES_EVENTS`| `Service` | `Infrastructure` | Service publishes event messages to infrastructure | $N:M$ |

---

## 3. Entity Resolution & Normalization Rules

### Canonical Naming Standard
- Entity names are converted to lowercase kebab-case for the canonical ID prefix (e.g., `"Payment Gateway API"` $\rightarrow$ `api:payment-gateway-api`).
- Punctuation and version indicators in display names are preserved (e.g., name: `"Payment Gateway v2"`).

### Alias Normalization Table
Common aliases must automatically resolve to identical canonical entities:

```
"Payment Svc", "pay-svc", "payment-service"         ==> service:payment-service
"Postgres", "PostgreSQL", "PostgreSQL 16"           ==> database:postgres-16
"Core Checkout", "Checkout Team", "checkout-eng"    ==> team:checkout-engineering
```

---

## 4. Entity Resolution Test Cases

| Raw Mention in Text | Expected Canonical ID | Resolved Entity Type | Normalization Rule Applied |
| :--- | :--- | :--- | :--- |
| `"The pay-gw service crashes"` | `service:payment-gateway` | `Service` | Alias dictionary lookup |
| `"Maintained by the Checkout Eng team"` | `team:checkout-engineering` | `Team` | Suffix stripping & team mapping |
| `"Vulnerable to CVE-2024-21626"` | `vuln:cve-2024-21626` | `Vulnerability` | Uppercase CVE regex normalization |
| `"Publishes to our Kafka event cluster"` | `infra:kafka-event-bus` | `Infrastructure` | Synonym mapping |
| `"Calls /v2/charge endpoint"` | `api:process-charge-v2` | `API` | Path to API contract mapping |

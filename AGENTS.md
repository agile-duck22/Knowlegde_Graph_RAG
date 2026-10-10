# AGENTS.md — Multi-Agent Engineering Guidelines

> **Target Audience**: All autonomous coding agents, LLM assistants, and tool-driven CLI agents operating in this repository.

This repository enforces a strict **Documentation-First, Test-Driven, Architecture-Bound** workflow.

---

## 1. Authoritative Source Hierarchy

When making decisions or inspecting the codebase, agents must treat documentation files as the authoritative source in this exact precedence order:

1. [`docs/SECURITY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md) — Security policies, query injection rules, and secret protection.
2. [`docs/ARCHITECTURE.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ARCHITECTURE.md) — System boundaries, components, and communication protocols.
3. [`docs/DATA_MODEL.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md) & [`docs/ONTOLOGY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ONTOLOGY.md) — Data schemas, entity definitions, and database constraints.
4. [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md) & [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md) — Active task state, implementation backlog, and session facts.
5. [`docs/API_SPEC.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/API_SPEC.md) & [`docs/CONFIGURATION.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CONFIGURATION.md) — External interfaces and configuration variables.

---

## 2. Non-Negotiable Development Rules

1. **Check Task & Memory First**: Always review [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md) and [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md) prior to taking action. Do not invent new tasks without recording them.
2. **Dual-Store ID Consistency**: The canonical identifier for any chunk in PostgreSQL (`chunks.id`) must match the `chunk_id` stored on provenance nodes and relationships in Neo4j.
3. **No Unsafe Dynamic Cypher**: Never construct Cypher queries via raw string interpolation or direct LLM output. Use the parameterized template catalog defined in [`docs/RETRIEVAL_STRATEGY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/RETRIEVAL_STRATEGY.md).
4. **Mocked Testing by Default**: Unit tests and standard CI suites must run without live database dependencies or external paid API calls using mocks. Integration tests requiring live services must be opt-in via pytest marks (`@pytest.mark.integration`).
5. **Verified Verification**: Do not report tasks as `DONE` unless tests have been executed and verified in the current session.

---

## 3. Interaction Protocol & Phase Boundaries

- Work in small, isolated steps corresponding to a single Task ID.
- At the end of each task, update [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md) and [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md).
- **Phase Gate**: When all tasks in an implementation phase are complete, pause and request user confirmation before initiating the subsequent phase.

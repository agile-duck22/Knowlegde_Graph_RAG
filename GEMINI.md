# GEMINI / ANTIGRAVITY AGENT INSTRUCTIONS

> **Project**: Knowledge Graph RAG for Enterprise Data  
> **Workspace**: `c:\Users\workp\OneDrive\Desktop\kgr`

As an AI coding agent working in this repository, you must adhere strictly to the engineering rules, architectural invariants, and operational workflows detailed below.

---

## 1. Mandatory Pre-Flight Checklist

Before starting **any** implementation, refactoring, or testing task:

1. **Read Core Documentation**:
   - [`docs/INDEX.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/INDEX.md): System map and authority hierarchy.
   - [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md): Current verified state, active blockers, and last completed task.
   - [`docs/ARCHITECTURE.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ARCHITECTURE.md): System components, data flows, and design constraints.
   - [`docs/SECURITY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md): Threat model, credential handling, and query safety rules.
   - [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md): Task backlog, dependencies, and Definition of Done.
2. **Identify the Target Task ID**: Match your work to an explicit Task ID (e.g., `TASK-101`) in [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md).
3. **Verify Preconditions**: Ensure all prerequisite tasks are marked `DONE` and verified.
4. **State Intent**: State your planned edits, affected files, and test plan clearly before modifying code.

---

## 2. Core Architectural & Operational Invariants

1. **One Chunk ID Across Both Stores**: Every Neo4j entity/relationship provenance record and PostgreSQL vector embedding record **MUST** share the identical canonical `chunk_id` (UUID format).
2. **No Unrestricted Raw Cypher Generation**: Never execute LLM-generated dynamic Cypher strings directly against Neo4j. All graph operations must use parameterized Cypher templates defined in the query catalog.
3. **Idempotent Ingestion**: Ingesting the same document or chunk multiple times must produce identical database state without duplicate nodes, duplicate edges, or duplicate embeddings.
4. **Grounded Synthesis & Citation Integrity**: Every generated factual statement must cite a valid `chunk_id` retrieved in context. The system must validate citations against the retrieved candidate set before returning responses.
5. **No Hallucinated Test Results or Statuses**: Never report that tests passed or features work without executing the test commands and inspecting real outputs.
6. **No Secrets in Code or Logs**: Never commit API keys, database passwords, or auth tokens. Redact sensitive values from logs and error payloads.
7. **Small, Atomic, Testable Units**: Work in single task increments. Stop at phase boundaries to allow user review.
8. **Preserve Existing Working Code**: Do not refactor unrelated working components without explicit instructions.

---

## 3. Post-Task Protocol

Upon finishing code changes for a task:

1. **Execute Tests**: Run `pytest` or relevant test suite commands to verify changes.
2. **Security & Lint Inspection**: Check for credential leaks, parameter injection risks, and lint violations (`ruff`, `mypy`).
3. **Update Documentation & State**:
   - Update task status in [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md).
   - Record verified progress, next recommendations, and changes in [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md).
   - If architectural decisions were made, append to [`docs/DECISIONS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md).
   - If API contracts or schemas changed, update [`docs/API_SPEC.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/API_SPEC.md) or [`docs/DATA_MODEL.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md).
4. **Report Summary**: Provide a clear breakdown of files modified, tests executed with exact outcomes, and the next recommended task ID.

---

## 4. Escalation & Clarification Triggers

Halt and request user clarification when:
- An architectural decision requires picking an unconfirmed technology or service (e.g., embedding provider selection).
- A schema modification introduces breaking changes across stores.
- External API costs could increase significantly due to unbudgeted batch processing.
- Security boundaries or authentication models are unclear.

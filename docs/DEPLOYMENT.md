# Deployment & Operations Runbook

This document details the local development setup, containerized services (Docker Compose), database initialization, migrations, and operational runbooks.

---

## 1. Local Development Setup (Planned)

### Prerequisites
- Python 3.11+
- Docker Engine 24+ & Docker Compose v2+
- Git

### Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

---

## 2. Docker Compose Infrastructure (`docker-compose.yml`)

The platform utilizes two primary persistent services:
1. **Neo4j 5.x Community**: Graph database on ports `7474` (HTTP) and `7687` (Bolt).
2. **PostgreSQL 16 with pgvector**: Relational and vector store on port `5432`.

### Docker Services Specification
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: kgr-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-kgr_db}
      POSTGRES_USER: ${POSTGRES_USER:-kgr_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-kgr_password}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - kgr_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-kgr_user} -d ${POSTGRES_DB:-kgr_db}"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s

  neo4j:
    image: neo4j:5.18-community
    container_name: kgr-neo4j
    restart: unless-stopped
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-neo4j_password}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_initial__size: 512m
      NEO4J_dbms_memory_heap_max__size: 1G
    ports:
      - "127.0.0.1:7474:7474" # HTTP Browser UI (Localhost only)
      - "127.0.0.1:7687:7687" # Bolt Protocol (Localhost only)
    volumes:
      - kgr_neo4jdata:/data
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-neo4j_password} 'RETURN 1'"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 15s

volumes:
  kgr_pgdata:
    name: kgr_pgdata
  kgr_neo4jdata:
    name: kgr_neo4jdata
```

---

## 3. Database Initialization & Migrations

```bash
# Start storage containers
docker compose up -d

# Run PostgreSQL DDL migrations and Neo4j constraint initialization
python -m src.db.migrate

# Verify database readiness
python -m src.db.healthcheck
```

---

## 4. Production Hardening Checklist

- [ ] Change default database passwords in `.env` to high-entropy secrets.
- [ ] Bind database container ports to `127.0.0.1` or internal Docker networks rather than `0.0.0.0`.
- [ ] Enable TLS/SSL termination at the reverse proxy (Nginx / Cloudflare).
- [ ] Configure volume backup scripts for PostgreSQL WAL archives and Neo4j dump files.
- [ ] Enforce request rate limiting and token consumption limits per client IP / API key.

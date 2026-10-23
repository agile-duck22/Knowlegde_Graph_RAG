"""Offline unit tests for scientific-paper experiment preflight and infrastructure isolation.

Verifies:
- Rejection of production ports (5432, 7687) and database names (kgr_db).
- Rejection of production container names and volume names.
- Correct isolated configuration acceptance.
- Rejection of missing or placeholder credentials in strict mode.
- Rejection of paid embedding configurations.
- Absence of silent fallback to production .env.
- Absence of secret leakage in validation messages.
"""

from __future__ import annotations

from pathlib import Path

from experiments.scipaper.scripts.preflight_check import (
    parse_env_file_isolated,
    run_preflight_checks,
    validate_compose_file,
    validate_environment_dict,
)

BASE_DIR = Path(__file__).resolve().parent.parent
COMPOSE_PATH = BASE_DIR / "docker-compose.scipaper.yml"
EXAMPLE_ENV_PATH = BASE_DIR / ".env.scipaper.example"


def test_compose_file_exists_and_is_valid() -> None:
    """Verify that docker-compose.scipaper.yml exists and passes all isolation checks."""
    assert COMPOSE_PATH.exists()
    errors = validate_compose_file(COMPOSE_PATH)
    assert not errors, f"Compose isolation check failed: {errors}"


def test_example_env_file_exists_and_passes_permissive_preflight() -> None:
    """Verify that .env.scipaper.example passes preflight with allow_placeholders=True."""
    assert EXAMPLE_ENV_PATH.exists()
    passed, errors = run_preflight_checks(
        compose_path=COMPOSE_PATH,
        env_path=EXAMPLE_ENV_PATH,
        allow_placeholders=True,
    )
    assert passed, f"Example env failed preflight: {errors}"


def test_example_env_fails_strict_preflight_due_to_placeholders() -> None:
    """Verify that .env.scipaper.example fails strict preflight because of placeholder passwords."""
    passed, errors = run_preflight_checks(
        compose_path=COMPOSE_PATH,
        env_path=EXAMPLE_ENV_PATH,
        allow_placeholders=False,
    )
    assert not passed
    assert any("placeholder" in err.lower() for err in errors)


def test_rejects_production_postgres_port() -> None:
    """Verify preflight rejects POSTGRES_PORT=5432."""
    env = {
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "EMBEDDING_PROVIDER": "sentence_transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSIONS": "384",
        "ALLOW_PAID_EMBEDDINGS": "false",
        "POSTGRES_PASSWORD": "secure_isolated_password",
        "NEO4J_PASSWORD": "secure_isolated_password",
    }
    errors = validate_environment_dict(env)
    assert any("production/enterprise port 5432" in err for err in errors)


def test_rejects_production_neo4j_port() -> None:
    """Verify preflight rejects NEO4J_URI=bolt://localhost:7687."""
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7687",
        "EMBEDDING_PROVIDER": "sentence_transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSIONS": "384",
        "ALLOW_PAID_EMBEDDINGS": "false",
        "POSTGRES_PASSWORD": "secure_isolated_password",
        "NEO4J_PASSWORD": "secure_isolated_password",
    }
    errors = validate_environment_dict(env)
    assert any("production/enterprise port 7687" in err for err in errors)


def test_rejects_production_db_name() -> None:
    """Verify preflight rejects POSTGRES_DB=kgr_db."""
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "kgr_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "EMBEDDING_PROVIDER": "sentence_transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSIONS": "384",
        "ALLOW_PAID_EMBEDDINGS": "false",
        "POSTGRES_PASSWORD": "secure_isolated_password",
        "NEO4J_PASSWORD": "secure_isolated_password",
    }
    errors = validate_environment_dict(env)
    assert any("enterprise database 'kgr_db'" in err for err in errors)


def test_rejects_paid_embeddings_enabled() -> None:
    """Verify preflight rejects ALLOW_PAID_EMBEDDINGS=true."""
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "EMBEDDING_PROVIDER": "sentence_transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSIONS": "384",
        "ALLOW_PAID_EMBEDDINGS": "true",
        "POSTGRES_PASSWORD": "secure_isolated_password",
        "NEO4J_PASSWORD": "secure_isolated_password",
    }
    errors = validate_environment_dict(env)
    assert any("ALLOW_PAID_EMBEDDINGS is 'true'" in err for err in errors)


def test_rejects_openai_embedding_provider() -> None:
    """Verify preflight rejects non-local embedding provider."""
    env = {
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "scipaper_db",
        "NEO4J_URI": "bolt://localhost:7688",
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "1536",
        "ALLOW_PAID_EMBEDDINGS": "false",
        "POSTGRES_PASSWORD": "secure_isolated_password",
        "NEO4J_PASSWORD": "secure_isolated_password",
    }
    errors = validate_environment_dict(env)
    assert any(
        "EMBEDDING_PROVIDER 'openai' must be 'sentence_transformers'" in err for err in errors
    )


def test_compose_rejects_production_container_name(tmp_path: Path) -> None:
    """Verify compose validator detects and rejects enterprise container name."""
    bad_compose = tmp_path / "docker-compose.bad.yml"
    bad_compose.write_text(
        """
services:
  scipaper-postgres:
    image: pgvector/pgvector:pg16
    container_name: kgr-postgres
    ports:
      - "127.0.0.1:5433:5432"
    volumes:
      - kgr_scipaper_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
volumes:
  kgr_scipaper_pgdata:
    name: kgr_scipaper_pgdata
networks:
  kgr-scipaper-net:
    name: kgr-scipaper-net
""",
        encoding="utf-8",
    )
    errors = validate_compose_file(bad_compose)
    assert any("forbidden enterprise container name 'kgr-postgres'" in err for err in errors)


def test_compose_rejects_production_volume_name(tmp_path: Path) -> None:
    """Verify compose validator detects and rejects enterprise volume name."""
    bad_compose = tmp_path / "docker-compose.bad.yml"
    bad_compose.write_text(
        """
services:
  scipaper-postgres:
    image: pgvector/pgvector:pg16
    container_name: kgr-scipaper-postgres
    ports:
      - "127.0.0.1:5433:5432"
    volumes:
      - kgr_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
volumes:
  kgr_pgdata:
    name: kgr_pgdata
networks:
  kgr-scipaper-net:
    name: kgr-scipaper-net
""",
        encoding="utf-8",
    )
    errors = validate_compose_file(bad_compose)
    assert any("forbidden enterprise volume 'kgr_pgdata'" in err for err in errors)


def test_no_secret_leakage_in_error_messages() -> None:
    """Verify that password secrets are never printed in error messages."""
    super_secret_pg_pass = "super_secret_unmasked_pg_password_123"
    super_secret_neo_pass = "super_secret_unmasked_neo_password_456"

    env = {
        "POSTGRES_PORT": "5432",  # trigger error
        "POSTGRES_DB": "kgr_db",  # trigger error
        "NEO4J_URI": "bolt://localhost:7687",  # trigger error
        "EMBEDDING_PROVIDER": "openai",  # trigger error
        "ALLOW_PAID_EMBEDDINGS": "true",  # trigger error
        "POSTGRES_PASSWORD": super_secret_pg_pass,
        "NEO4J_PASSWORD": super_secret_neo_pass,
    }
    errors = validate_environment_dict(env)
    assert len(errors) > 0

    joined_errors = " ".join(errors)
    assert super_secret_pg_pass not in joined_errors
    assert super_secret_neo_pass not in joined_errors


def test_isolated_parser_does_not_fall_back_to_root_env(tmp_path: Path) -> None:
    """Verify that parse_env_file_isolated only reads the target file and does not inspect root .env."""
    custom_env = tmp_path / ".env.test"
    custom_env.write_text(
        """
# Isolated test env
POSTGRES_PORT=5433
POSTGRES_DB=scipaper_db
MY_ISOLATED_FLAG=true
""",
        encoding="utf-8",
    )
    parsed = parse_env_file_isolated(custom_env)
    assert parsed.get("POSTGRES_PORT") == "5433"
    assert parsed.get("POSTGRES_DB") == "scipaper_db"
    assert parsed.get("MY_ISOLATED_FLAG") == "true"
    # Should not have variables from root .env or other defaults
    assert "LLM_PROVIDER" not in parsed
    assert "OPENAI_API_KEY" not in parsed


def test_scipaper_ddl_generation_and_safety() -> None:
    """Verify generated DDL statements for PostgreSQL and Neo4j conform to expected schemas."""
    from experiments.scipaper.scripts.init_scipaper_schema import (
        build_scipaper_postgres_ddl,
        get_scipaper_neo4j_ddl,
    )

    pg_ddl = build_scipaper_postgres_ddl(embedding_dim=384)
    assert "VECTOR(384)" in pg_ddl
    assert "CREATE TABLE IF NOT EXISTS documents" in pg_ddl
    assert "CREATE TABLE IF NOT EXISTS chunks" in pg_ddl
    assert "CREATE TABLE IF NOT EXISTS entity_links" in pg_ddl
    assert "idx_chunks_embedding_hnsw" in pg_ddl
    assert "idx_chunks_embedding_384_hnsw" in pg_ddl

    neo_ddl = get_scipaper_neo4j_ddl()
    assert len(neo_ddl) == 2
    assert any("unique_entity_id" in s for s in neo_ddl)
    assert any("entity_text_index" in s for s in neo_ddl)

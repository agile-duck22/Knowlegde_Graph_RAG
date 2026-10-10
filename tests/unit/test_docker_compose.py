"""Unit tests for Docker Compose specification syntax and security policies."""

from pathlib import Path


def test_docker_compose_file_exists() -> None:
    """Verify docker-compose.yml exists at repo root."""
    compose_file = Path(__file__).parents[2] / "docker-compose.yml"
    assert compose_file.exists()


def test_services_defined() -> None:
    """Verify required storage services exist."""
    compose_file = Path(__file__).parents[2] / "docker-compose.yml"
    content = compose_file.read_text(encoding="utf-8")
    assert "postgres:" in content
    assert "neo4j:" in content


def test_pinned_image_versions() -> None:
    """Verify images use pinned compatible versions, not generic latest."""
    compose_file = Path(__file__).parents[2] / "docker-compose.yml"
    content = compose_file.read_text(encoding="utf-8")
    assert "image: pgvector/pgvector:pg16" in content
    assert "image: neo4j:5.18-community" in content


def test_localhost_port_bindings() -> None:
    """Security: Verify database ports are bound strictly to 127.0.0.1."""
    compose_file = Path(__file__).parents[2] / "docker-compose.yml"
    content = compose_file.read_text(encoding="utf-8")
    assert "127.0.0.1:5432:5432" in content
    assert "127.0.0.1:7474:7474" in content
    assert "127.0.0.1:7687:7687" in content


def test_persistent_named_volumes() -> None:
    """Reliability: Verify named persistent volumes are defined."""
    compose_file = Path(__file__).parents[2] / "docker-compose.yml"
    content = compose_file.read_text(encoding="utf-8")
    assert "kgr_pgdata:" in content
    assert "kgr_neo4jdata:" in content


def test_healthchecks_configured() -> None:
    """Reliability: Verify health checks are defined for both services."""
    compose_file = Path(__file__).parents[2] / "docker-compose.yml"
    content = compose_file.read_text(encoding="utf-8")
    assert "pg_isready" in content
    assert "cypher-shell" in content

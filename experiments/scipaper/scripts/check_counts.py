import asyncio

import asyncpg
from neo4j import GraphDatabase


async def run() -> None:
    pg = await asyncpg.connect(
        "postgresql://scipaper_user:7rHk2ahfnAIflKiBvnBzLVjCSwHh0amM@127.0.0.1:5433/scipaper_db"
    )
    docs = await pg.fetchval("SELECT count(*) FROM documents;")
    chunks = await pg.fetchval("SELECT count(*) FROM chunks;")
    elinks = await pg.fetchval("SELECT count(*) FROM entity_links;")
    emb384 = await pg.fetchval("SELECT count(*) FROM chunks WHERE embedding_384 IS NOT NULL;")
    await pg.close()
    print("PG docs:", docs, "chunks:", chunks, "elinks:", elinks, "emb384:", emb384)

    driver = GraphDatabase.driver(
        "bolt://127.0.0.1:7688", auth=("neo4j", "qFOYJSDGF8vOW4TV8Otzu47iNkftJwaf")
    )
    with driver.session() as s:
        rec_n = s.run("MATCH (n) RETURN count(n) AS cnt").single()
        rec_r = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
        nodes = rec_n["cnt"] if rec_n is not None else 0
        rels = rec_r["cnt"] if rec_r is not None else 0
    driver.close()
    print("Neo4j nodes:", nodes, "rels:", rels)


if __name__ == "__main__":
    asyncio.run(run())

"""
Provision Databricks Vector Search infrastructure for MaintBot.

Creates:
1. A Vector Search endpoint (STANDARD type)
2. A Direct Vector Access index for maintenance_knowledge

Usage:
    uv run python scripts/setup_vector_search.py [--migrate]

    --migrate: Copy existing memories from pgvector to Vector Search
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VS_ENDPOINT_NAME = os.getenv("VS_ENDPOINT_NAME", "maint-bot-vs-endpoint")
VS_CATALOG = os.getenv("VS_CATALOG", "ankit_yadav")
VS_SCHEMA = os.getenv("VS_SCHEMA", "maint_bot")
VS_INDEX_NAME = f"{VS_CATALOG}.{VS_SCHEMA}.maintenance_knowledge_vs"
EMBEDDING_DIM = 1024


def create_endpoint(client):
    """Create VS endpoint if it doesn't exist."""
    try:
        endpoint = client.get_endpoint(name=VS_ENDPOINT_NAME)
        state = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
        logger.info(f"Endpoint '{VS_ENDPOINT_NAME}' exists, state: {state}")
        return state
    except Exception:
        logger.info(f"Creating endpoint '{VS_ENDPOINT_NAME}'...")
        client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
        logger.info("Endpoint creation initiated")
        return "PROVISIONING"


def wait_for_endpoint(client, timeout=600):
    """Wait for endpoint to reach ONLINE state."""
    logger.info(f"Waiting for endpoint to be ONLINE (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            endpoint = client.get_endpoint(name=VS_ENDPOINT_NAME)
            state = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
            if state == "ONLINE":
                logger.info("Endpoint is ONLINE")
                return True
            logger.info(f"  State: {state}, waiting...")
        except Exception as e:
            logger.warning(f"  Error checking endpoint: {e}")
        time.sleep(15)
    logger.error(f"Endpoint did not reach ONLINE within {timeout}s")
    return False


def create_index(client):
    """Create the Direct Vector Access index if it doesn't exist."""
    try:
        client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
        logger.info(f"Index '{VS_INDEX_NAME}' already exists")
        return
    except Exception:
        pass

    logger.info(f"Creating Direct Vector Access index '{VS_INDEX_NAME}'...")
    client.create_direct_access_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        primary_key="id",
        embedding_dimension=EMBEDDING_DIM,
        embedding_vector_column="embedding",
        schema={
            "id": "string",
            "technician_id": "string",
            "content": "string",
            "memory_type": "string",
            "equipment_tag": "string",
            "importance": "float",
            "is_active": "string",
            "access_count": "int",
            "created_at": "string",
            "updated_at": "string",
            "source_work_order_id": "string",
        },
    )
    logger.info(f"Index '{VS_INDEX_NAME}' created")


def migrate_from_pgvector(client):
    """Copy all active memories from pgvector to Vector Search."""
    import asyncio

    from databricks_langchain import DatabricksEmbeddings

    embeddings_model = DatabricksEmbeddings(
        endpoint=os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
    )

    async def _migrate():
        from agent_server.db_schema import get_async_connection

        async with get_async_connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT id, technician_id, content, memory_type, equipment_tag,
                           importance, is_active, access_count, created_at, updated_at,
                           source_work_order_id
                    FROM maintenance_knowledge
                    WHERE is_active = TRUE
                    """
                )
            ).fetchall()

        if not rows:
            logger.info("No memories to migrate")
            return

        logger.info(f"Migrating {len(rows)} memories from pgvector to Vector Search...")
        index = client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)

        batch = []
        for row in rows:
            r = dict(row)
            embedding = embeddings_model.embed_query(r["content"])
            batch.append({
                "id": str(r["id"]),
                "technician_id": str(r["technician_id"]),
                "content": r["content"],
                "memory_type": r["memory_type"],
                "equipment_tag": r["equipment_tag"] or "",
                "importance": float(r["importance"]),
                "is_active": "true" if r["is_active"] else "false",
                "access_count": r["access_count"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else "",
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
                "source_work_order_id": str(r["source_work_order_id"]) if r["source_work_order_id"] else "",
                "embedding": embedding,
            })

            # Batch upsert every 50 items
            if len(batch) >= 50:
                index.upsert(batch)
                logger.info(f"  Upserted {len(batch)} items")
                batch = []

        if batch:
            index.upsert(batch)
            logger.info(f"  Upserted {len(batch)} items")

        logger.info(f"Migration complete: {len(rows)} memories")

    asyncio.run(_migrate())


def main():
    parser = argparse.ArgumentParser(description="Set up Databricks Vector Search for MaintBot")
    parser.add_argument("--migrate", action="store_true", help="Migrate existing pgvector memories to VS")
    args = parser.parse_args()

    from databricks.vector_search.client import VectorSearchClient
    client = VectorSearchClient()

    # Step 1: Create endpoint
    state = create_endpoint(client)

    # Step 2: Wait for endpoint
    if state != "ONLINE":
        if not wait_for_endpoint(client):
            logger.error("Cannot proceed without ONLINE endpoint")
            sys.exit(1)

    # Step 3: Create index
    create_index(client)

    # Step 4: Migrate (optional)
    if args.migrate:
        migrate_from_pgvector(client)

    logger.info("Setup complete!")
    logger.info(f"  Endpoint: {VS_ENDPOINT_NAME}")
    logger.info(f"  Index: {VS_INDEX_NAME}")
    logger.info(f"  To use: set MEMORY_BACKEND=databricks_vs in your .env")


if __name__ == "__main__":
    main()

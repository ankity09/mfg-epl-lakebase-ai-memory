"""
Long-term memory for MaintBot: Databricks Vector Search backend.

Uses a Direct Vector Access Index for real-time insert + query.
Implements the same interface as memory.py (pgvector backend) so the
router can swap between them via MEMORY_BACKEND env var.
"""

import json
import logging
import math
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from langchain_core.messages import HumanMessage

from agent_server.pii import redact_pii

logger = logging.getLogger(__name__)

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIM = 1024

VS_ENDPOINT_NAME = os.getenv("VS_ENDPOINT_NAME", "mas-3876475e-endpoint")
VS_CATALOG = os.getenv("VS_CATALOG", "ankit_yadav")
VS_SCHEMA = os.getenv("VS_SCHEMA", "maint_bot")
VS_INDEX_NAME = os.getenv("VS_INDEX_NAME", f"{VS_CATALOG}.{VS_SCHEMA}.maint_knowledge_vs")

# Lazy-init singletons
_llm = None
_embeddings = None
_vs_client = None
_vs_index = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    return _llm


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = DatabricksEmbeddings(endpoint=EMBEDDING_ENDPOINT)
    return _embeddings


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using Databricks GTE."""
    return _get_embeddings().embed_query(text)


def _get_vs_client():
    """Get or create the Vector Search client.

    In Databricks Apps, the SDK's WorkspaceClient auto-detects SP OAuth
    credentials. We get a fresh token from it and pass to VectorSearchClient.
    For local dev, falls back to default VectorSearchClient() which reads
    DATABRICKS_CONFIG_PROFILE.
    """
    global _vs_client
    if _vs_client is None:
        from databricks.vector_search.client import VectorSearchClient
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            # Get workspace URL
            host = w.config.host
            if host and not host.startswith("https://"):
                host = f"https://{host}"
            # Check if we have SP credentials (Databricks Apps injects these)
            sp_client_id = os.getenv("DATABRICKS_CLIENT_ID")
            sp_client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
            if sp_client_id and sp_client_secret:
                _vs_client = VectorSearchClient(
                    workspace_url=host,
                    service_principal_client_id=sp_client_id,
                    service_principal_client_secret=sp_client_secret,
                )
                logger.info("VS client initialized with SP credentials")
            else:
                # Local dev or PAT-based auth — get token from SDK
                token = w.config.token
                if token:
                    _vs_client = VectorSearchClient(
                        workspace_url=host,
                        personal_access_token=token,
                    )
                    logger.info("VS client initialized with PAT from WorkspaceClient")
                else:
                    _vs_client = VectorSearchClient()
                    logger.info("VS client initialized with default auth")
        except Exception as e:
            logger.warning(f"WorkspaceClient init failed ({e}), trying default VS client")
            _vs_client = VectorSearchClient()
    return _vs_client


def _get_index():
    """Get the Vector Search index reference."""
    global _vs_index
    if _vs_index is None:
        client = _get_vs_client()
        _vs_index = client.get_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_NAME,
        )
    return _vs_index


# ──────────────────────────────────────────────
# Index setup (called from setup script or on first use)
# ──────────────────────────────────────────────

def ensure_vs_index():
    """Create the VS endpoint and index if they don't exist. Idempotent."""
    client = _get_vs_client()

    # Check/create endpoint
    try:
        endpoint = client.get_endpoint(name=VS_ENDPOINT_NAME)
        logger.info(f"VS endpoint '{VS_ENDPOINT_NAME}' exists, state: {endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')}")
    except Exception:
        logger.info(f"Creating VS endpoint '{VS_ENDPOINT_NAME}'...")
        client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
        logger.info(f"VS endpoint '{VS_ENDPOINT_NAME}' creation initiated")

    # Check/create index
    try:
        client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
        logger.info(f"VS index '{VS_INDEX_NAME}' exists")
    except Exception:
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
                "is_active": "string",  # VS doesn't support boolean filters well, use "true"/"false"
                "access_count": "int",
                "created_at": "string",
                "updated_at": "string",
                "source_work_order_id": "string",
            },
        )
        logger.info(f"VS index '{VS_INDEX_NAME}' created")


# ──────────────────────────────────────────────
# Memory storage
# ──────────────────────────────────────────────

async def store_memory_vs(
    technician_id: str,
    content: str,
    memory_type: str,
    equipment_tag: Optional[str],
    importance: float,
    source_work_order_id: Optional[str],
) -> str:
    """Store a memory with PII redaction and embedding in Vector Search."""
    safe_content = redact_pii(content)
    embedding = get_embedding(safe_content)
    memory_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    index = _get_index()
    index.upsert([{
        "id": memory_id,
        "technician_id": technician_id,
        "content": safe_content,
        "memory_type": memory_type,
        "equipment_tag": equipment_tag or "",
        "importance": importance,
        "is_active": "true",
        "access_count": 0,
        "created_at": now,
        "updated_at": now,
        "source_work_order_id": source_work_order_id or "",
        "embedding": embedding,
    }])
    return memory_id


# ──────────────────────────────────────────────
# Deduplication & consolidation
# ──────────────────────────────────────────────

async def find_similar_memories_vs(
    technician_id: str,
    content: str,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Find existing memories similar to new content using Vector Search."""
    embedding = get_embedding(content)
    index = _get_index()

    results = index.similarity_search(
        query_vector=embedding,
        columns=["id", "content", "memory_type", "equipment_tag", "importance"],
        filters={"technician_id": [technician_id], "is_active": ["true"]},
        num_results=5,
        score_threshold=threshold,
    )

    rows = results.get("result", {}).get("data_array", [])
    columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]

    matches = []
    for row in rows:
        item = dict(zip(columns, row))
        # The score column is typically the last one
        if "score" in item:
            item["similarity"] = float(item["score"])
        matches.append(item)
    return matches


async def consolidate_memory_vs(
    technician_id: str,
    new_content: str,
    existing_memory_id: str,
) -> str:
    """Merge new knowledge with an existing similar memory using LLM, store in VS."""
    index = _get_index()

    # Fetch existing memory
    try:
        existing_results = index.similarity_search(
            query_vector=get_embedding(""),  # dummy, we filter by ID
            columns=["id", "content", "importance", "access_count", "created_at", "updated_at",
                      "technician_id", "memory_type", "equipment_tag", "source_work_order_id"],
            filters={"id": [existing_memory_id]},
            num_results=1,
        )
        rows = existing_results.get("result", {}).get("data_array", [])
        columns = [c["name"] for c in existing_results.get("manifest", {}).get("columns", [])]
    except Exception:
        rows = []
        columns = []

    if not rows:
        return await store_memory_vs(
            technician_id, new_content, "procedure", None, 0.5, None
        )

    existing = dict(zip(columns, rows[0]))

    # LLM merge
    llm = _get_llm()
    merge_response = await llm.ainvoke(
        [
            HumanMessage(
                content=(
                    "Merge these two related maintenance facts into one concise statement:\n"
                    f"1. {existing['content']}\n"
                    f"2. {new_content}\n\n"
                    "Provide only the merged statement:"
                )
            )
        ]
    )

    merged = merge_response.content.strip()
    safe_merged = redact_pii(merged)
    new_embedding = get_embedding(safe_merged)
    new_importance = min(1.0, float(existing.get("importance", 0.5)) + 0.1)
    now = datetime.now().isoformat()

    # Upsert the updated memory
    index.upsert([{
        "id": existing_memory_id,
        "technician_id": existing.get("technician_id", technician_id),
        "content": safe_merged,
        "memory_type": existing.get("memory_type", "procedure"),
        "equipment_tag": existing.get("equipment_tag", ""),
        "importance": new_importance,
        "is_active": "true",
        "access_count": int(existing.get("access_count", 0)) + 1,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "source_work_order_id": existing.get("source_work_order_id", ""),
        "embedding": new_embedding,
    }])
    return existing_memory_id


# ──────────────────────────────────────────────
# Hybrid retrieval
# ──────────────────────────────────────────────

async def retrieve_memories_vs(
    technician_id: str,
    query: str,
    limit: int = 5,
    w_similarity: float = 0.7,
    w_recency: float = 0.2,
    w_importance: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Retrieve memories using Databricks Vector Search + hybrid scoring.
    Same scoring formula as pgvector backend:
        final_score = 0.7 * similarity + 0.2 * recency + 0.1 * importance
    """
    embedding = get_embedding(query)
    index = _get_index()

    results = index.similarity_search(
        query_vector=embedding,
        columns=["id", "content", "memory_type", "equipment_tag", "importance",
                 "access_count", "created_at", "updated_at"],
        filters={"technician_id": [technician_id], "is_active": ["true"]},
        num_results=20,
    )

    rows = results.get("result", {}).get("data_array", [])
    columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]

    candidates = []
    for row in rows:
        item = dict(zip(columns, row))
        # VS returns a score column (cosine similarity)
        if "score" in item:
            item["similarity"] = float(item["score"])
        else:
            item["similarity"] = 0.0
        candidates.append(item)

    now = datetime.now()
    scored = []
    for mem in candidates:
        # Parse created_at string back to datetime for recency calc
        created_at = mem.get("created_at", "")
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at)
                days_old = (now - created_dt).days
            except (ValueError, TypeError):
                days_old = 0
        else:
            days_old = 0

        recency = math.exp(-days_old / 30)
        importance = float(mem.get("importance", 0.5))

        final_score = (
            w_similarity * mem["similarity"]
            + w_recency * recency
            + w_importance * importance
        )
        mem["recency_score"] = round(recency, 3)
        mem["final_score"] = round(final_score, 4)
        scored.append(mem)

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Update access counts for returned memories
    top = scored[:limit]
    if top:
        for mem in top:
            try:
                index.upsert([{
                    "id": mem["id"],
                    "access_count": int(mem.get("access_count", 0)) + 1,
                }])
            except Exception as e:
                logger.debug(f"Access count update failed for {mem['id']}: {e}")

    return top


# ──────────────────────────────────────────────
# GDPR operations
# ──────────────────────────────────────────────

async def view_memories_vs(technician_id: str) -> List[Dict[str, Any]]:
    """View all active memories for a technician from Vector Search."""
    # Use a generic embedding to get all results, relying on filter
    index = _get_index()
    try:
        results = index.similarity_search(
            query_vector=[0.0] * EMBEDDING_DIM,  # dummy vector
            columns=["id", "content", "memory_type", "equipment_tag", "importance",
                     "access_count", "created_at"],
            filters={"technician_id": [technician_id], "is_active": ["true"]},
            num_results=100,
        )
        rows = results.get("result", {}).get("data_array", [])
        columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"VS view_memories failed: {e}")
        return []


async def delete_memory_vs(technician_id: str, memory_id: str) -> bool:
    """Soft-delete a memory by setting is_active=false in Vector Search."""
    index = _get_index()
    try:
        index.upsert([{
            "id": memory_id,
            "is_active": "false",
            "updated_at": datetime.now().isoformat(),
        }])
        logger.info(f"GDPR delete (VS): memory {memory_id} for tech {technician_id}")
        return True
    except Exception as e:
        logger.error(f"VS delete_memory failed: {e}")
        return False


async def delete_all_memories_vs(technician_id: str) -> int:
    """Soft-delete all memories for a technician in Vector Search."""
    memories = await view_memories_vs(technician_id)
    count = 0
    index = _get_index()
    now = datetime.now().isoformat()
    for mem in memories:
        try:
            index.upsert([{"id": mem["id"], "is_active": "false", "updated_at": now}])
            count += 1
        except Exception as e:
            logger.error(f"VS delete failed for {mem['id']}: {e}")
    logger.info(f"GDPR delete_all (VS): {count} memories for tech {technician_id}")
    return count


async def export_memories_vs(technician_id: str) -> Dict[str, Any]:
    """Export all user data from Vector Search in portable format."""
    memories = await view_memories_vs(technician_id)
    return {
        "user_id": technician_id,
        "export_date": datetime.now().isoformat(),
        "backend": "databricks_vector_search",
        "total_memories": len(memories),
        "memories": [
            {
                "id": str(m.get("id", "")),
                "content": m.get("content", ""),
                "memory_type": m.get("memory_type", ""),
                "equipment_tag": m.get("equipment_tag", ""),
                "importance": m.get("importance", 0),
                "access_count": m.get("access_count", 0),
                "created_at": m.get("created_at", ""),
            }
            for m in memories
        ],
    }


# ──────────────────────────────────────────────
# Process conversation memories (end-of-session)
# ──────────────────────────────────────────────

# Re-use extraction from memory.py — it's backend-agnostic (LLM call only)
from agent_server.memory import extract_memories  # noqa: E402


async def process_conversation_memories_vs(
    technician_id: str,
    conversation_text: str,
    work_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract, deduplicate, and store memories using Vector Search backend."""
    memories = await extract_memories(conversation_text)
    results = {"extracted": len(memories), "stored": 0, "consolidated": 0, "skipped": 0}

    for mem in memories:
        similar = await find_similar_memories_vs(
            technician_id, mem["content"], threshold=0.85
        )

        if similar:
            await consolidate_memory_vs(
                technician_id, mem["content"], str(similar[0]["id"])
            )
            results["consolidated"] += 1
        else:
            await store_memory_vs(
                technician_id=technician_id,
                content=mem["content"],
                memory_type=mem["memory_type"],
                equipment_tag=mem.get("equipment_tag"),
                importance=mem["importance"],
                source_work_order_id=work_order_id,
            )
            results["stored"] += 1

    return results

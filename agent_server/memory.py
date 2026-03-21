"""
Long-term memory for MaintBot: pgvector-based hybrid retrieval,
LLM-driven extraction, deduplication, and consolidation.

Uses a custom `maintenance_knowledge` table (NOT AsyncDatabricksStore)
to support hybrid scoring, equipment tags, memory types, and consolidation.
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from langchain_core.messages import HumanMessage

from agent_server.db_schema import get_async_connection
from agent_server.pii import redact_pii

logger = logging.getLogger(__name__)

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIM = 1024

# Lazy-init singletons
_llm = None
_embeddings = None


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


def format_embedding_for_postgres(embedding: List[float]) -> str:
    """Format embedding as a Postgres vector literal."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


# ──────────────────────────────────────────────
# Memory extraction
# ──────────────────────────────────────────────

EXTRACTION_PROMPT = """Analyze this maintenance conversation and extract important knowledge that would be useful for future reference.

For each piece of knowledge, classify it as one of:
- failure_mode: How equipment fails or symptoms observed
- part_preference: Preferred parts, brands, or suppliers
- procedure: Diagnostic or repair procedures that worked
- machine_quirk: Known equipment-specific behaviors or calibration notes
- vendor_info: Supplier details, lead times, or availability

Return a JSON array. Each item must have:
- "content": The knowledge statement (one clear sentence)
- "memory_type": One of the types above
- "equipment_tag": Equipment ID if applicable (e.g., "HP-L4-001"), or null
- "importance": Float 0.0-1.0 (1.0 = critical safety/failure info)

Conversation:
{conversation}

Return ONLY the JSON array, no other text."""


async def extract_memories(conversation_text: str) -> List[Dict[str, Any]]:
    """Extract structured memories from a conversation using LLM."""
    llm = _get_llm()
    response = await llm.ainvoke(
        [HumanMessage(content=EXTRACTION_PROMPT.format(conversation=conversation_text))]
    )

    try:
        text = response.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        memories = json.loads(text)

        valid = []
        for m in memories:
            if all(k in m for k in ["content", "memory_type", "importance"]):
                m.setdefault("equipment_tag", None)
                m["importance"] = max(0.0, min(1.0, float(m["importance"])))
                valid.append(m)
        return valid
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Memory extraction parse error: {e}")
        return []


# ──────────────────────────────────────────────
# Memory storage
# ──────────────────────────────────────────────

async def store_memory(
    technician_id: str,
    content: str,
    memory_type: str,
    equipment_tag: Optional[str],
    importance: float,
    source_work_order_id: Optional[str],
) -> str:
    """Store a memory with PII redaction and embedding."""
    safe_content = redact_pii(content)
    embedding = get_embedding(safe_content)
    embedding_str = format_embedding_for_postgres(embedding)

    async with get_async_connection() as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO maintenance_knowledge
                    (technician_id, content, memory_type, equipment_tag,
                     importance, embedding, source_work_order_id)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                RETURNING id
                """,
                (
                    technician_id,
                    safe_content,
                    memory_type,
                    equipment_tag,
                    importance,
                    embedding_str,
                    source_work_order_id,
                ),
            )
        ).fetchone()
        return str(row["id"])


# ──────────────────────────────────────────────
# Deduplication & consolidation
# ──────────────────────────────────────────────

async def find_similar_memories(
    technician_id: str,
    content: str,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Find existing memories similar to new content using cosine distance."""
    embedding = get_embedding(content)
    embedding_str = format_embedding_for_postgres(embedding)

    async with get_async_connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT id, content, memory_type, equipment_tag, importance,
                       1 - (embedding <=> %s::vector) as similarity
                FROM maintenance_knowledge
                WHERE technician_id = %s AND is_active = TRUE
                  AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT 5
                """,
                (embedding_str, technician_id, embedding_str, threshold),
            )
        ).fetchall()
        return [dict(r) for r in rows]


async def consolidate_memory(
    technician_id: str,
    new_content: str,
    existing_memory_id: str,
) -> str:
    """Merge new knowledge with an existing similar memory using LLM."""
    async with get_async_connection() as conn:
        existing = await (
            await conn.execute(
                "SELECT content, importance FROM maintenance_knowledge WHERE id = %s",
                (existing_memory_id,),
            )
        ).fetchone()

        if not existing:
            return await store_memory(
                technician_id, new_content, "procedure", None, 0.5, None
            )

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
        embedding_str = format_embedding_for_postgres(new_embedding)
        new_importance = min(1.0, existing["importance"] + 0.1)

        await conn.execute(
            """
            UPDATE maintenance_knowledge
            SET content = %s, embedding = %s::vector, importance = %s,
                access_count = access_count + 1, updated_at = NOW()
            WHERE id = %s
            """,
            (safe_merged, embedding_str, new_importance, existing_memory_id),
        )
        return str(existing_memory_id)


# ──────────────────────────────────────────────
# Hybrid retrieval
# ──────────────────────────────────────────────

async def retrieve_memories_hybrid(
    technician_id: str,
    query: str,
    limit: int = 5,
    w_similarity: float = 0.7,
    w_recency: float = 0.2,
    w_importance: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Retrieve memories using hybrid scoring:
        final_score = 0.7 * similarity + 0.2 * recency + 0.1 * importance
    where recency = exp(-days_old / 30) (30-day half-life).
    """
    embedding = get_embedding(query)
    embedding_str = format_embedding_for_postgres(embedding)

    async with get_async_connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT id, content, memory_type, equipment_tag, importance,
                       access_count, created_at, updated_at,
                       1 - (embedding <=> %s::vector) as similarity
                FROM maintenance_knowledge
                WHERE technician_id = %s AND is_active = TRUE
                ORDER BY embedding <=> %s::vector
                LIMIT 20
                """,
                (embedding_str, technician_id, embedding_str),
            )
        ).fetchall()

        candidates = [dict(r) for r in rows]

    now = datetime.now()
    scored = []
    for mem in candidates:
        days_old = (
            (now - mem["created_at"].replace(tzinfo=None)).days
            if mem["created_at"]
            else 0
        )
        recency = math.exp(-days_old / 30)

        final_score = (
            w_similarity * mem["similarity"]
            + w_recency * recency
            + w_importance * mem["importance"]
        )
        mem["recency_score"] = round(recency, 3)
        mem["final_score"] = round(final_score, 4)
        scored.append(mem)

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Update access counts for returned memories
    top = scored[:limit]
    if top:
        async with get_async_connection() as conn:
            for mem in top:
                await conn.execute(
                    "UPDATE maintenance_knowledge SET access_count = access_count + 1 WHERE id = %s",
                    (mem["id"],),
                )

    return top


def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories for injection into the system prompt."""
    if not memories:
        return "No relevant maintenance knowledge found."

    lines = []
    for i, mem in enumerate(memories, 1):
        tag = f" [{mem['equipment_tag']}]" if mem.get("equipment_tag") else ""
        score = f" (relevance: {mem['final_score']:.2f})" if "final_score" in mem else ""
        lines.append(f"{i}. [{mem['memory_type']}]{tag} {mem['content']}{score}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Process conversation memories (end-of-session)
# ──────────────────────────────────────────────

async def process_conversation_memories(
    technician_id: str,
    conversation_text: str,
    work_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract memories from a conversation, deduplicate, and store.
    Returns summary of what was stored/consolidated.
    """
    memories = await extract_memories(conversation_text)
    results = {"extracted": len(memories), "stored": 0, "consolidated": 0, "skipped": 0}

    for mem in memories:
        # Check for similar existing memories
        similar = await find_similar_memories(
            technician_id, mem["content"], threshold=0.85
        )

        if similar:
            # Consolidate with the most similar existing memory
            await consolidate_memory(
                technician_id, mem["content"], str(similar[0]["id"])
            )
            results["consolidated"] += 1
        else:
            # Store as new memory
            await store_memory(
                technician_id=technician_id,
                content=mem["content"],
                memory_type=mem["memory_type"],
                equipment_tag=mem.get("equipment_tag"),
                importance=mem["importance"],
                source_work_order_id=work_order_id,
            )
            results["stored"] += 1

    return results

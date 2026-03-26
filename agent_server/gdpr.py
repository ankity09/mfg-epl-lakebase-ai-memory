"""
GDPR data controls for maintenance knowledge.

Implements:
- Art. 15: Right of access (view_memories)
- Art. 17: Right to erasure (delete_memory, delete_all_memories) — soft-delete via is_active=FALSE
- Art. 20: Right to data portability (export_memories)

Routes to pgvector or Databricks Vector Search based on MEMORY_BACKEND env var.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "pgvector")


# ──────────────────────────────────────────────
# pgvector implementations (original)
# ──────────────────────────────────────────────

async def _view_memories_pg(external_id: str) -> List[Dict[str, Any]]:
    from agent_server.db_schema import get_async_connection
    async with get_async_connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT m.id, m.content, m.memory_type, m.equipment_tag, m.importance,
                       m.access_count, m.created_at
                FROM maintenance_knowledge m
                JOIN technicians t ON m.technician_id = t.id
                WHERE t.external_id = %s AND m.is_active = TRUE
                ORDER BY m.created_at DESC
                """,
                (external_id,),
            )
        ).fetchall()
        return [dict(r) for r in rows]


async def _delete_memory_pg(external_id: str, memory_id: str) -> bool:
    from agent_server.db_schema import get_async_connection
    async with get_async_connection() as conn:
        result = await conn.execute(
            """
            UPDATE maintenance_knowledge m
            SET is_active = FALSE, updated_at = NOW()
            FROM technicians t
            WHERE m.technician_id = t.id
              AND t.external_id = %s
              AND m.id = %s
            """,
            (external_id, memory_id),
        )
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"GDPR delete: memory {memory_id} for user {external_id}")
        return deleted


async def _delete_all_memories_pg(external_id: str) -> int:
    from agent_server.db_schema import get_async_connection
    async with get_async_connection() as conn:
        result = await conn.execute(
            """
            UPDATE maintenance_knowledge m
            SET is_active = FALSE, updated_at = NOW()
            FROM technicians t
            WHERE m.technician_id = t.id
              AND t.external_id = %s
              AND m.is_active = TRUE
            """,
            (external_id,),
        )
        count = result.rowcount
        logger.info(f"GDPR delete_all: {count} memories for user {external_id}")
        return count


# ──────────────────────────────────────────────
# Public API (routes to correct backend)
# ──────────────────────────────────────────────

async def view_memories(external_id: str) -> List[Dict[str, Any]]:
    """View all active memories for a technician (GDPR Art. 15)."""
    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import view_memories_vs
        # VS backend uses technician_id directly (no join needed)
        # Resolve technician_id from external_id first
        tech_id = await _resolve_tech_id(external_id)
        if not tech_id:
            return []
        return await view_memories_vs(tech_id)
    return await _view_memories_pg(external_id)


async def delete_memory(external_id: str, memory_id: str) -> bool:
    """Soft-delete a specific memory (GDPR Art. 17)."""
    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import delete_memory_vs
        tech_id = await _resolve_tech_id(external_id)
        if not tech_id:
            return False
        return await delete_memory_vs(tech_id, memory_id)
    return await _delete_memory_pg(external_id, memory_id)


async def delete_all_memories(external_id: str) -> int:
    """Soft-delete all memories for a technician (GDPR Art. 17)."""
    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import delete_all_memories_vs
        tech_id = await _resolve_tech_id(external_id)
        if not tech_id:
            return 0
        return await delete_all_memories_vs(tech_id)
    return await _delete_all_memories_pg(external_id)


async def export_memories(external_id: str) -> Dict[str, Any]:
    """Export all user data in portable format (GDPR Art. 20)."""
    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import export_memories_vs
        tech_id = await _resolve_tech_id(external_id)
        if not tech_id:
            return {"user_id": external_id, "export_date": datetime.now().isoformat(), "total_memories": 0, "memories": []}
        return await export_memories_vs(tech_id)

    memories = await _view_memories_pg(external_id)
    return {
        "user_id": external_id,
        "export_date": datetime.now().isoformat(),
        "total_memories": len(memories),
        "memories": [
            {
                "id": str(m["id"]),
                "content": m["content"],
                "memory_type": m["memory_type"],
                "equipment_tag": m["equipment_tag"],
                "importance": m["importance"],
                "access_count": m["access_count"],
                "created_at": m["created_at"].isoformat() if m["created_at"] else None,
            }
            for m in memories
        ],
    }


async def _resolve_tech_id(external_id: str) -> Optional[str]:
    """Resolve technician UUID from external_id (email). Needed for VS backend."""
    try:
        from agent_server.db_schema import get_or_create_technician
        tech = await get_or_create_technician(external_id)
        return str(tech["id"])
    except Exception as e:
        logger.warning(f"Could not resolve technician for '{external_id}': {e}")
        return None

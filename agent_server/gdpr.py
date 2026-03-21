"""
GDPR data controls for maintenance knowledge.

Implements:
- Art. 15: Right of access (view_memories)
- Art. 17: Right to erasure (delete_memory, delete_all_memories) — soft-delete via is_active=FALSE
- Art. 20: Right to data portability (export_memories)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_server.db_schema import get_async_connection

logger = logging.getLogger(__name__)


async def view_memories(external_id: str) -> List[Dict[str, Any]]:
    """View all active memories for a technician (GDPR Art. 15)."""
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


async def delete_memory(external_id: str, memory_id: str) -> bool:
    """Soft-delete a specific memory (GDPR Art. 17)."""
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


async def delete_all_memories(external_id: str) -> int:
    """Soft-delete all memories for a technician (GDPR Art. 17)."""
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


async def export_memories(external_id: str) -> Dict[str, Any]:
    """Export all user data in portable format (GDPR Art. 20)."""
    memories = await view_memories(external_id)
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

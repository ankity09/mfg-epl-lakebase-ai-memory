"""
LangGraph @tool wrappers for memory operations.

Memory tools resolve technician identity from the user_id passed in custom_inputs
by looking up the technicians table directly.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


async def _resolve_technician(user_id: str = "anonymous"):
    """Resolve technician_id from user_id. Returns (tech_id, user_id) or (None, user_id)."""
    try:
        from agent_server.db_schema import get_or_create_technician
        tech = await get_or_create_technician(user_id)
        return str(tech["id"]), user_id
    except Exception as e:
        logger.warning(f"Could not resolve technician for '{user_id}': {e}")
        return None, user_id


@tool
async def get_maintenance_memory(
    query: str,
    user_id: str = "anonymous",
    limit: int = 5,
) -> str:
    """Search long-term maintenance knowledge for relevant information.

    Use this tool when you need to recall past maintenance experiences,
    known failure modes, part preferences, procedures, or equipment quirks.

    Args:
        query: What to search for (e.g., "HP-L4-001 hydraulic issues")
        user_id: The technician's user ID (default: anonymous)
        limit: Maximum number of memories to return (default 5)
    """
    tech_id, _ = await _resolve_technician(user_id)
    if not tech_id:
        return "No maintenance knowledge available yet."

    try:
        from agent_server.memory import format_memories_for_prompt, retrieve_memories_hybrid
        memories = await retrieve_memories_hybrid(
            technician_id=tech_id, query=query, limit=limit,
        )
        if not memories:
            return "No relevant maintenance knowledge found for this query."
        return format_memories_for_prompt(memories)
    except Exception as e:
        logger.error(f"Memory retrieval error for tech={tech_id} query='{query}': {e}")
        return f"Memory search failed: {str(e)[:200]}"


@tool
async def save_maintenance_memory(
    conversation_text: str,
    user_id: str = "anonymous",
) -> str:
    """Extract and save important maintenance knowledge from a conversation.

    Use this tool at the end of a diagnostic session to preserve learnings
    for future reference.

    Args:
        conversation_text: The conversation to extract knowledge from
        user_id: The technician's user ID
    """
    tech_id, _ = await _resolve_technician(user_id)
    if not tech_id:
        return "Cannot save memories: user context not available."

    from agent_server.memory import process_conversation_memories
    result = await process_conversation_memories(
        technician_id=tech_id,
        conversation_text=conversation_text,
    )
    return (
        f"Memory extraction complete: {result['extracted']} items found, "
        f"{result['stored']} new memories stored, "
        f"{result['consolidated']} consolidated with existing knowledge."
    )


@tool
async def gdpr_view_memories(user_id: str = "anonymous") -> str:
    """View all stored maintenance memories for a technician (GDPR Art. 15).

    Use this when a technician asks to see what data is stored about them.

    Args:
        user_id: The technician's user ID
    """
    from agent_server.gdpr import view_memories
    memories = await view_memories(user_id)
    if not memories:
        return f"No stored memories found for user '{user_id}'."

    lines = [f"Found {len(memories)} memories for '{user_id}':\n"]
    for m in memories:
        tag = f" [{m['equipment_tag']}]" if m.get("equipment_tag") else ""
        lines.append(
            f"- ID: {m['id']} | Type: {m['memory_type']}{tag} | "
            f"Importance: {m['importance']:.1f}\n  {m['content']}"
        )
    return "\n".join(lines)


@tool
async def gdpr_delete_memories(
    user_id: str = "anonymous",
    memory_id: Optional[str] = None,
    delete_all: bool = False,
) -> str:
    """Delete stored maintenance memories for a technician (GDPR Art. 17).

    Args:
        user_id: The technician's user ID
        memory_id: Specific memory ID to delete (optional)
        delete_all: If True, delete all memories for this user
    """
    from agent_server.gdpr import delete_all_memories, delete_memory
    if delete_all:
        count = await delete_all_memories(user_id)
        return f"Deleted all {count} memories for user '{user_id}'."
    elif memory_id:
        success = await delete_memory(user_id, memory_id)
        if success:
            return f"Memory {memory_id} deleted for user '{user_id}'."
        return f"Memory {memory_id} not found or already deleted."
    else:
        return "Please specify either a memory_id to delete or set delete_all=True."


@tool
async def gdpr_export_memories(user_id: str = "anonymous") -> str:
    """Export all stored data for a technician in portable format (GDPR Art. 20).

    Args:
        user_id: The technician's user ID
    """
    from agent_server.gdpr import export_memories
    export = await export_memories(user_id)
    return json.dumps(export, indent=2, default=str)

"""
LangGraph @tool wrappers for memory operations.

These tools are passed to create_agent(tools=...) so the agent
can decide when to recall, save, and manage memories.

The technician_id and user_id are injected via custom_inputs in the
agent state — the LLM does NOT need to provide UUIDs.
"""

import json
import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_server.gdpr import (
    delete_all_memories,
    delete_memory,
    export_memories,
    view_memories,
)
from agent_server.memory import (
    format_memories_for_prompt,
    process_conversation_memories,
    retrieve_memories_hybrid,
)

logger = logging.getLogger(__name__)


def _get_context_ids(config: RunnableConfig) -> tuple[str, str, str]:
    """Extract technician_id, user_id, work_order_id from tool config."""
    ci = config.get("configurable", {}).get("custom_inputs", {})
    tech_id = ci.get("technician_id", "")
    user_id = ci.get("user_id", "anonymous")
    wo_id = ci.get("work_order_id", "")
    return tech_id, user_id, wo_id


@tool
async def get_maintenance_memory(
    query: str,
    config: RunnableConfig,
    limit: int = 5,
) -> str:
    """Search long-term maintenance knowledge for relevant information.

    Use this tool when you need to recall past maintenance experiences,
    known failure modes, part preferences, procedures, or equipment quirks.

    Args:
        query: What to search for (e.g., "HP-L4-001 hydraulic issues")
        limit: Maximum number of memories to return (default 5)
    """
    tech_id, _, _ = _get_context_ids(config)
    if not tech_id:
        return "Cannot search memories: technician context not available."

    memories = await retrieve_memories_hybrid(
        technician_id=tech_id,
        query=query,
        limit=limit,
    )
    if not memories:
        return "No relevant maintenance knowledge found for this query."
    return format_memories_for_prompt(memories)


@tool
async def save_maintenance_memory(
    conversation_text: str,
    config: RunnableConfig,
) -> str:
    """Extract and save important maintenance knowledge from a conversation.

    Use this tool at the end of a diagnostic session to preserve learnings
    for future reference. It automatically extracts failure modes, part
    preferences, procedures, and equipment quirks.

    Args:
        conversation_text: The conversation to extract knowledge from
    """
    tech_id, _, wo_id = _get_context_ids(config)
    if not tech_id:
        return "Cannot save memories: technician context not available."

    result = await process_conversation_memories(
        technician_id=tech_id,
        conversation_text=conversation_text,
        work_order_id=wo_id or None,
    )
    return (
        f"Memory extraction complete: {result['extracted']} items found, "
        f"{result['stored']} new memories stored, "
        f"{result['consolidated']} consolidated with existing knowledge."
    )


@tool
async def gdpr_view_memories(config: RunnableConfig) -> str:
    """View all stored maintenance memories for the current technician (GDPR Art. 15).

    Use this when a technician asks to see what data is stored about them.
    """
    _, user_id, _ = _get_context_ids(config)
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
    config: RunnableConfig,
    memory_id: Optional[str] = None,
    delete_all: bool = False,
) -> str:
    """Delete stored maintenance memories for the current technician (GDPR Art. 17).

    Use this when a technician requests deletion of their stored data.
    Can delete a specific memory by ID or all memories.

    Args:
        memory_id: Specific memory ID to delete (optional)
        delete_all: If True, delete all memories for this user
    """
    _, user_id, _ = _get_context_ids(config)
    if delete_all:
        count = await delete_all_memories(user_id)
        return f"Deleted all {count} memories for user '{user_id}'."
    elif memory_id:
        success = await delete_memory(user_id, memory_id)
        if success:
            return f"Memory {memory_id} deleted for user '{user_id}'."
        return f"Memory {memory_id} not found or already deleted for user '{user_id}'."
    else:
        return "Please specify either a memory_id to delete or set delete_all=True."


@tool
async def gdpr_export_memories(config: RunnableConfig) -> str:
    """Export all stored data for the current technician in portable format (GDPR Art. 20).

    Use this when a technician requests a copy of all their stored data.
    """
    _, user_id, _ = _get_context_ids(config)
    export = await export_memories(user_id)
    return json.dumps(export, indent=2, default=str)

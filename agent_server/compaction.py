"""
Conversation compaction strategies for MaintBot.

Three strategies to manage token budget:
1. summarize  — LLM creates a concise summary of older messages (default)
2. last_n     — Keep only the last N turns, mark older as compacted
3. truncate   — Remove oldest messages until under token budget

Compaction operates on work_order_events (our custom table), NOT on
AsyncCheckpointSaver's internal checkpoint state.
"""

import logging
import os
from typing import Any, Dict, List

import tiktoken
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage

from agent_server.db_schema import get_async_connection

logger = logging.getLogger(__name__)

# Configuration
COMPACT_STRATEGY = os.getenv("COMPACT_STRATEGY", "summarize")
COMPACT_THRESHOLD = float(os.getenv("COMPACT_THRESHOLD", "0.8"))
COMPACT_KEEP_TURNS = int(os.getenv("COMPACT_KEEP_TURNS", "5"))
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")

_encoding = None


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text))


async def compact_summarize(work_order_id: str, keep_turns: int = 5) -> Dict[str, Any]:
    """
    Summarize older messages using LLM, keeping the last `keep_turns` exchanges.
    The summary replaces compacted messages as a single 'system' event.
    """
    async with get_async_connection() as conn:
        # Get all non-compacted events ordered by time
        rows = await (
            await conn.execute(
                "SELECT id, role, content, created_at FROM work_order_events "
                "WHERE work_order_id = %s AND is_compacted = FALSE "
                "ORDER BY created_at",
                (work_order_id,),
            )
        ).fetchall()
        events = [dict(r) for r in rows]

        if len(events) <= keep_turns * 2:
            return {"events_compacted": 0, "strategy": "summarize", "skipped": True}

        # Split into older (to summarize) and recent (to keep)
        cutoff = len(events) - (keep_turns * 2)
        older_events = events[:cutoff]
        older_ids = [str(e["id"]) for e in older_events]

        # Build conversation text for summarization
        conv_text = "\n".join(
            f"{e['role'].upper()}: {e['content']}" for e in older_events
        )

        # LLM summarization
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
        summary_prompt = (
            "Summarize this maintenance conversation concisely. "
            "Preserve: equipment IDs, failure modes, diagnostic findings, "
            "recommended parts, and any safety concerns.\n\n"
            f"{conv_text}\n\n"
            "Provide a concise summary:"
        )
        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()

        # Mark older events as compacted
        await conn.execute(
            "UPDATE work_order_events SET is_compacted = TRUE "
            "WHERE id = ANY(%s::uuid[])",
            (older_ids,),
        )

        # Insert summary as a system event
        summary_tokens = count_tokens(summary)
        await conn.execute(
            "INSERT INTO work_order_events "
            "(work_order_id, role, content, token_count, is_compacted) "
            "VALUES (%s, 'system', %s, %s, FALSE)",
            (work_order_id, f"[Session Summary] {summary}", summary_tokens),
        )

        return {
            "events_compacted": len(older_ids),
            "strategy": "summarize",
            "summary_tokens": summary_tokens,
        }


async def compact_last_n(work_order_id: str, keep_turns: int = 5) -> Dict[str, Any]:
    """Keep only the last `keep_turns * 2` events, mark older as compacted."""
    async with get_async_connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT id FROM work_order_events "
                "WHERE work_order_id = %s AND is_compacted = FALSE "
                "ORDER BY created_at",
                (work_order_id,),
            )
        ).fetchall()
        events = [dict(r) for r in rows]

        if len(events) <= keep_turns * 2:
            return {"events_compacted": 0, "strategy": "last_n", "skipped": True}

        cutoff = len(events) - (keep_turns * 2)
        older_ids = [str(events[i]["id"]) for i in range(cutoff)]

        await conn.execute(
            "UPDATE work_order_events SET is_compacted = TRUE "
            "WHERE id = ANY(%s::uuid[])",
            (older_ids,),
        )

        return {"events_compacted": len(older_ids), "strategy": "last_n"}


async def compact_truncate(
    work_order_id: str, token_budget: int = 4000
) -> Dict[str, Any]:
    """Remove oldest messages until total tokens <= token_budget."""
    async with get_async_connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT id, token_count FROM work_order_events "
                "WHERE work_order_id = %s AND is_compacted = FALSE "
                "ORDER BY created_at",
                (work_order_id,),
            )
        ).fetchall()
        events = [dict(r) for r in rows]

        total = sum(e["token_count"] for e in events)
        compacted_ids: List[str] = []

        for event in events:
            if total <= token_budget:
                break
            total -= event["token_count"]
            compacted_ids.append(str(event["id"]))

        if compacted_ids:
            await conn.execute(
                "UPDATE work_order_events SET is_compacted = TRUE "
                "WHERE id = ANY(%s::uuid[])",
                (compacted_ids,),
            )

        return {"events_compacted": len(compacted_ids), "strategy": "truncate"}


async def compact_conversation(
    work_order_id: str,
    max_tokens: int = 8000,
    strategy: str = None,
    keep_turns: int = None,
) -> Dict[str, Any]:
    """
    Run compaction if token count exceeds threshold.
    Returns compaction result dict.
    """
    strategy = strategy or COMPACT_STRATEGY
    keep_turns = keep_turns or COMPACT_KEEP_TURNS

    from agent_server.db_schema import get_work_order_token_count

    current_tokens = await get_work_order_token_count(work_order_id)
    threshold = max_tokens * COMPACT_THRESHOLD

    if current_tokens <= threshold:
        return {
            "compacted": False,
            "current_tokens": current_tokens,
            "threshold": threshold,
        }

    logger.info(
        f"Compacting work order {work_order_id}: "
        f"{current_tokens} tokens > {threshold} threshold, strategy={strategy}"
    )

    if strategy == "summarize":
        result = await compact_summarize(work_order_id, keep_turns=keep_turns)
    elif strategy == "last_n":
        result = await compact_last_n(work_order_id, keep_turns=keep_turns)
    elif strategy == "truncate":
        token_budget = int(max_tokens * 0.5)
        result = await compact_truncate(work_order_id, token_budget=token_budget)
    else:
        raise ValueError(f"Unknown compaction strategy: {strategy}")

    result["compacted"] = True
    new_tokens = await get_work_order_token_count(work_order_id)
    result["tokens_before"] = current_tokens
    result["tokens_after"] = new_tokens
    return result


def get_context_messages(
    events: List[Dict[str, Any]],
) -> str:
    """
    Build context string from non-compacted events for injection into system prompt.
    Includes any [Session Summary] entries.
    """
    parts = []
    for e in events:
        if e["role"] == "system" and e["content"].startswith("[Session Summary]"):
            parts.append(e["content"])
        else:
            parts.append(f"{e['role'].upper()}: {e['content']}")
    return "\n".join(parts)

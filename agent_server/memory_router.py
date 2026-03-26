"""
Memory backend router for MaintBot.

Dispatches memory operations to either pgvector (Lakebase) or Databricks
Vector Search based on the MEMORY_BACKEND env var. Adds timing
instrumentation so we can compare backends in production.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "pgvector")
logger.info(f"Memory backend: {MEMORY_BACKEND}")


def _log_timing(backend: str, operation: str, elapsed_ms: float, extra: str = ""):
    """Structured log line for memory operation timing."""
    msg = f"[MEMORY:{backend}] {operation}: {elapsed_ms:.1f}ms"
    if extra:
        msg += f", {extra}"
    logger.info(msg)


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────

async def retrieve_memories_hybrid(
    technician_id: str,
    query: str,
    limit: int = 5,
    w_similarity: float = 0.7,
    w_recency: float = 0.2,
    w_importance: float = 0.1,
) -> List[Dict[str, Any]]:
    """Retrieve memories using the configured backend."""
    t0 = time.perf_counter()

    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import retrieve_memories_vs
        results = await retrieve_memories_vs(
            technician_id, query, limit, w_similarity, w_recency, w_importance
        )
    else:
        from agent_server.memory import retrieve_memories_hybrid as _retrieve
        results = await _retrieve(
            technician_id, query, limit, w_similarity, w_recency, w_importance
        )

    elapsed = (time.perf_counter() - t0) * 1000
    _log_timing(MEMORY_BACKEND, "retrieve_memories", elapsed, f"{len(results)} results")
    return results


# ──────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────

async def store_memory(
    technician_id: str,
    content: str,
    memory_type: str,
    equipment_tag: Optional[str],
    importance: float,
    source_work_order_id: Optional[str],
) -> str:
    """Store a memory using the configured backend."""
    t0 = time.perf_counter()

    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import store_memory_vs
        result = await store_memory_vs(
            technician_id, content, memory_type, equipment_tag, importance, source_work_order_id
        )
    else:
        from agent_server.memory import store_memory as _store
        result = await _store(
            technician_id, content, memory_type, equipment_tag, importance, source_work_order_id
        )

    elapsed = (time.perf_counter() - t0) * 1000
    _log_timing(MEMORY_BACKEND, "store_memory", elapsed)
    return result


# ──────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────

async def find_similar_memories(
    technician_id: str,
    content: str,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Find similar memories using the configured backend."""
    t0 = time.perf_counter()

    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import find_similar_memories_vs
        results = await find_similar_memories_vs(technician_id, content, threshold)
    else:
        from agent_server.memory import find_similar_memories as _find
        results = await _find(technician_id, content, threshold)

    elapsed = (time.perf_counter() - t0) * 1000
    _log_timing(MEMORY_BACKEND, "find_similar", elapsed, f"{len(results)} matches")
    return results


# ──────────────────────────────────────────────
# Consolidation
# ──────────────────────────────────────────────

async def consolidate_memory(
    technician_id: str,
    new_content: str,
    existing_memory_id: str,
) -> str:
    """Consolidate memories using the configured backend."""
    t0 = time.perf_counter()

    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import consolidate_memory_vs
        result = await consolidate_memory_vs(technician_id, new_content, existing_memory_id)
    else:
        from agent_server.memory import consolidate_memory as _consolidate
        result = await _consolidate(technician_id, new_content, existing_memory_id)

    elapsed = (time.perf_counter() - t0) * 1000
    _log_timing(MEMORY_BACKEND, "consolidate_memory", elapsed)
    return result


# ──────────────────────────────────────────────
# Process conversation memories
# ──────────────────────────────────────────────

async def process_conversation_memories(
    technician_id: str,
    conversation_text: str,
    work_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract and store memories from a conversation using the configured backend."""
    t0 = time.perf_counter()

    if MEMORY_BACKEND == "databricks_vs":
        from agent_server.memory_vs import process_conversation_memories_vs
        result = await process_conversation_memories_vs(
            technician_id, conversation_text, work_order_id
        )
    else:
        from agent_server.memory import process_conversation_memories as _process
        result = await _process(technician_id, conversation_text, work_order_id)

    elapsed = (time.perf_counter() - t0) * 1000
    _log_timing(
        MEMORY_BACKEND, "process_conversation_memories", elapsed,
        f"{result.get('stored', 0)} stored, {result.get('consolidated', 0)} consolidated"
    )
    return result


# ──────────────────────────────────────────────
# Format (backend-agnostic — delegates to memory.py)
# ──────────────────────────────────────────────

def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories for injection into the system prompt."""
    from agent_server.memory import format_memories_for_prompt as _format
    return _format(memories)


# ──────────────────────────────────────────────
# Extraction (backend-agnostic — delegates to memory.py)
# ──────────────────────────────────────────────

async def extract_memories(conversation_text: str) -> List[Dict[str, Any]]:
    """Extract memories from conversation text (LLM call, backend-agnostic)."""
    from agent_server.memory import extract_memories as _extract
    return await _extract(conversation_text)

"""
Async database schema management for MaintBot custom tables.

Uses AsyncLakebasePool from databricks_ai_bridge — the same connection
mechanism that AsyncCheckpointSaver uses — for our custom tables:
technicians, work_orders, work_order_events, maintenance_knowledge.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from databricks_ai_bridge.lakebase import AsyncLakebasePool

logger = logging.getLogger(__name__)

# Custom tables live in databricks_postgres (same DB as AsyncCheckpointSaver)
# inside a dedicated schema to avoid collisions with checkpoint tables.
SCHEMA_NAME = os.getenv("MAINT_SCHEMA", "maint_bot")

# Lakebase connection config
LAKEBASE_INSTANCE_NAME = os.getenv("LAKEBASE_INSTANCE_NAME") or None
LAKEBASE_PROJECT = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
LAKEBASE_BRANCH = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None
# Direct endpoint path (skips list_endpoints API call which SP may not have scope for)
LAKEBASE_ENDPOINT = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None

_schema_initialized = False
_pool: Optional[AsyncLakebasePool] = None


async def _get_pool() -> AsyncLakebasePool:
    """Get or create the shared async Lakebase connection pool."""
    global _pool
    if _pool is None:
        _pool = AsyncLakebasePool(
            instance_name=LAKEBASE_INSTANCE_NAME,
            autoscaling_endpoint=LAKEBASE_ENDPOINT,
            project=LAKEBASE_PROJECT if not LAKEBASE_ENDPOINT else None,
            branch=LAKEBASE_BRANCH if not LAKEBASE_ENDPOINT else None,
        )
        await _pool.open()
    return _pool


@asynccontextmanager
async def get_async_connection():
    """Get an async psycopg connection with maint_bot schema on search_path."""
    pool = await _get_pool()
    async with pool.pool.connection() as conn:
        await conn.execute(f"SET search_path TO {SCHEMA_NAME}, public")
        yield conn


async def ensure_schema():
    """Create custom tables if they don't exist. Idempotent."""
    global _schema_initialized
    if _schema_initialized:
        return

    async with get_async_connection() as conn:
        # DDL statements (CREATE SCHEMA/TABLE/INDEX, ALTER TABLE) require
        # ownership or CREATE privilege that the SP typically doesn't have.
        # The admin grant script pre-creates all schemas, tables, and indexes,
        # so these are all no-ops in production.  We run each in its own
        # try/except so permission errors don't block the SP from using the
        # tables it already has access to.
        ddl_statements = [
            f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}",
            """CREATE TABLE IF NOT EXISTS technicians (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                external_id VARCHAR(255) UNIQUE NOT NULL,
                display_name VARCHAR(255),
                specializations JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS work_orders (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                technician_id UUID REFERENCES technicians(id),
                agent_id VARCHAR(100) DEFAULT 'maint-bot',
                equipment_id VARCHAR(100),
                priority VARCHAR(20) DEFAULT 'medium',
                status VARCHAR(50) DEFAULT 'active',
                summary TEXT,
                max_tokens INTEGER DEFAULT 8000,
                created_at TIMESTAMP DEFAULT NOW(),
                ended_at TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS work_order_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                work_order_id UUID REFERENCES work_orders(id),
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                user_id VARCHAR(255),
                token_count INTEGER DEFAULT 0,
                is_compacted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )""",
            "ALTER TABLE work_order_events ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)",
            "CREATE EXTENSION IF NOT EXISTS vector",
            """CREATE TABLE IF NOT EXISTS maintenance_knowledge (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                technician_id UUID REFERENCES technicians(id),
                content TEXT NOT NULL,
                memory_type VARCHAR(50) NOT NULL,
                equipment_tag VARCHAR(100),
                importance FLOAT DEFAULT 0.5,
                embedding vector(1024),
                source_work_order_id UUID,
                is_active BOOLEAN DEFAULT TRUE,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )""",
            "CREATE INDEX IF NOT EXISTS idx_events_work_order ON work_order_events(work_order_id)",
            "CREATE INDEX IF NOT EXISTS idx_events_not_compacted ON work_order_events(work_order_id) WHERE is_compacted = FALSE",
            "CREATE INDEX IF NOT EXISTS idx_work_orders_technician ON work_orders(technician_id)",
            "CREATE INDEX IF NOT EXISTS idx_work_orders_status ON work_orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_technician ON maintenance_knowledge(technician_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_type ON maintenance_knowledge(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_equipment ON maintenance_knowledge(equipment_tag)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_active ON maintenance_knowledge(is_active) WHERE is_active = TRUE",
        ]

        for sql in ddl_statements:
            try:
                await conn.execute(sql)
            except Exception as e:
                if "permission denied" in str(e).lower() or "must be owner" in str(e).lower():
                    logger.debug(f"DDL skipped (expected for SP): {e}")
                    await conn.rollback()
                else:
                    raise

    _schema_initialized = True
    logger.info("MaintBot custom schema initialized")


async def get_or_create_technician(external_id: str) -> Dict[str, Any]:
    """Get existing technician or create a new one."""
    async with get_async_connection() as conn:
        row = await (
            await conn.execute(
                "SELECT * FROM technicians WHERE external_id = %s",
                (external_id,),
            )
        ).fetchone()

        if row:
            return dict(row)

        row = await (
            await conn.execute(
                "INSERT INTO technicians (external_id, display_name) "
                "VALUES (%s, %s) RETURNING *",
                (external_id, external_id),
            )
        ).fetchone()
        return dict(row)


async def create_work_order(
    technician_id: str,
    work_order_id: Optional[str] = None,
    equipment_id: Optional[str] = None,
    priority: str = "medium",
    max_tokens: int = 8000,
) -> Dict[str, Any]:
    """Create a new work order (conversation session).

    If work_order_id is provided, use it as the primary key (e.g. frontend
    conversation UUID) instead of generating a new one with gen_random_uuid().
    """
    async with get_async_connection() as conn:
        if work_order_id:
            row = await (
                await conn.execute(
                    "INSERT INTO work_orders (id, technician_id, agent_id, equipment_id, priority, max_tokens) "
                    "VALUES (%s::uuid, %s, 'maint-bot', %s, %s, %s) RETURNING *",
                    (work_order_id, technician_id, equipment_id, priority, max_tokens),
                )
            ).fetchone()
        else:
            row = await (
                await conn.execute(
                    "INSERT INTO work_orders (technician_id, agent_id, equipment_id, priority, max_tokens) "
                    "VALUES (%s, 'maint-bot', %s, %s, %s) RETURNING *",
                    (technician_id, equipment_id, priority, max_tokens),
                )
            ).fetchone()
        return dict(row)


async def get_work_order(work_order_id: str) -> Optional[Dict[str, Any]]:
    """Get a work order by ID."""
    async with get_async_connection() as conn:
        row = await (
            await conn.execute(
                "SELECT * FROM work_orders WHERE id = %s",
                (work_order_id,),
            )
        ).fetchone()
        return dict(row) if row else None


async def log_event(
    work_order_id: str,
    role: str,
    content: str,
    token_count: int = 0,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a message event to work_order_events with token count and optional user_id."""
    async with get_async_connection() as conn:
        row = await (
            await conn.execute(
                "INSERT INTO work_order_events (work_order_id, role, content, token_count, user_id) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                (work_order_id, role, content, token_count, user_id),
            )
        ).fetchone()
        return dict(row)


async def get_work_order_token_count(work_order_id: str) -> int:
    """Sum token_count for non-compacted events in a work order."""
    async with get_async_connection() as conn:
        row = await (
            await conn.execute(
                "SELECT COALESCE(SUM(token_count), 0) as total_tokens "
                "FROM work_order_events "
                "WHERE work_order_id = %s AND is_compacted = FALSE",
                (work_order_id,),
            )
        ).fetchone()
        return row["total_tokens"]


async def get_events(work_order_id: str, include_compacted: bool = False) -> list:
    """Get events for a work order, optionally including compacted ones."""
    async with get_async_connection() as conn:
        where = "WHERE work_order_id = %s"
        if not include_compacted:
            where += " AND is_compacted = FALSE"

        rows = await (
            await conn.execute(
                f"SELECT * FROM work_order_events {where} ORDER BY created_at",
                (work_order_id,),
            )
        ).fetchall()
        return [dict(r) for r in rows]

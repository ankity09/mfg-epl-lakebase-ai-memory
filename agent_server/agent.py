"""
MaintBot — Manufacturing Maintenance Assistant (Databricks Apps deployment).

Phase 1: Short-term memory — AsyncCheckpointSaver + custom tables (technicians,
         work_orders, work_order_events) with compaction and token tracking.
Phase 2: Long-term memory — pgvector hybrid retrieval, PII redaction, GDPR controls.

Database operations gracefully degrade — the agent works without Lakebase,
falling back to stateless mode if the connection fails.
"""

import logging
import os
import re
from typing import Any, AsyncGenerator, Optional, Sequence, TypedDict

import litellm
import mlflow
from databricks_langchain import AsyncCheckpointSaver, ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from typing_extensions import Annotated

from agent_server.utils import (
    _get_or_create_thread_id,
    process_agent_astream_events,
)

# Lazy imports for custom tables and memory (avoid import-time failures in containers)
def _import_db():
    from agent_server.db_schema import (
        create_work_order, ensure_schema, get_events,
        get_or_create_technician, get_work_order, log_event,
    )
    return ensure_schema, get_or_create_technician, create_work_order, get_work_order, log_event, get_events

def _import_memory():
    from agent_server.memory import format_memories_for_prompt, retrieve_memories_hybrid
    return format_memories_for_prompt, retrieve_memories_hybrid

def _import_memory_tools():
    from agent_server.memory_tools import (
        gdpr_delete_memories, gdpr_export_memories, gdpr_view_memories,
        get_maintenance_memory, save_maintenance_memory,
    )
    return get_maintenance_memory, save_maintenance_memory, gdpr_view_memories, gdpr_delete_memories, gdpr_export_memories

def _import_compaction():
    from agent_server.compaction import compact_conversation, count_tokens
    return compact_conversation, count_tokens

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

############################################
# Configuration
############################################
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
LAKEBASE_PROJECT = os.getenv("LAKEBASE_AUTOSCALING_PROJECT") or None
LAKEBASE_BRANCH = os.getenv("LAKEBASE_AUTOSCALING_BRANCH") or None
LAKEBASE_INSTANCE_NAME = os.getenv("LAKEBASE_INSTANCE_NAME") or None
LAKEBASE_ENDPOINT = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None

_has_lakebase = bool(LAKEBASE_INSTANCE_NAME or LAKEBASE_ENDPOINT or (LAKEBASE_PROJECT and LAKEBASE_BRANCH))
if not _has_lakebase:
    logger.warning("No Lakebase config — running in stateless mode")

SYSTEM_PROMPT = """You are MaintBot, an AI maintenance assistant with access to long-term maintenance knowledge.

Your role:
- Help technicians diagnose equipment issues
- Use your maintenance knowledge to inform recommendations
- Recall past repairs, known failure modes, and part preferences
- Track diagnostic findings during the conversation

## Your Maintenance Knowledge
{user_memories}

## Guidelines
- Reference specific equipment IDs, part numbers, and measurements
- When you recall relevant knowledge, cite it explicitly: "Based on past experience..."
- Suggest systematic diagnostic approaches
- Flag safety concerns for high-severity issues
- Be concise but thorough

## Current Diagnostic Context
- Equipment: {equipment_id}
- Issue Type: {issue_type}
- Severity: {severity}
- Working Diagnosis: {diagnosis}
- Recommended Parts: {recommended_parts}
{session_context}"""


############################################
# State
############################################
class MaintenanceAssistantState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


############################################
# Tools
############################################
@tool
def detect_equipment(user_message: str) -> str:
    """Detect equipment ID from a user message. Equipment IDs follow the pattern XX-LN-NNN (e.g., HP-L4-001)."""
    match = re.search(r"[A-Z]{2}-[A-Z]\d+-\d{3}", user_message)
    if match:
        return f"Detected equipment: {match.group()}"
    return "No equipment ID detected in message."


@tool
def assess_severity(user_message: str) -> str:
    """Assess the severity level from a user's description of an equipment issue."""
    msg_lower = user_message.lower()
    for level in ["critical", "high", "medium", "low"]:
        if level in msg_lower:
            return f"Severity assessed as: {level}"
    critical_keywords = ["safety", "fire", "explosion", "injury", "shutdown", "emergency"]
    high_keywords = ["production stop", "line down", "major leak", "failure"]
    if any(kw in msg_lower for kw in critical_keywords):
        return "Severity assessed as: critical"
    if any(kw in msg_lower for kw in high_keywords):
        return "Severity assessed as: high"
    return "Severity assessed as: medium (default)"


@tool
def classify_issue_type(user_message: str) -> str:
    """Classify the type of maintenance issue from a user's description."""
    msg_lower = user_message.lower()
    for itype in ["hydraulic", "electrical", "mechanical", "software", "pneumatic"]:
        if itype in msg_lower:
            return f"Issue classified as: {itype}"
    return "Issue type: undetermined — ask technician for more details."


############################################
# Agent factory
############################################
async def init_agent(checkpointer=None, memory_context: str = ""):
    """Create the MaintBot LangGraph agent with all tools."""
    model = ChatDatabricks(endpoint=LLM_ENDPOINT)

    # Phase 1: diagnostic tools
    tools = [detect_equipment, assess_severity, classify_issue_type]

    # Phase 2: long-term memory + GDPR tools (lazy import)
    try:
        mt = _import_memory_tools()
        tools.extend(list(mt))
    except Exception as e:
        logger.warning(f"Memory tools unavailable: {e}")

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT.format(
            user_memories=memory_context or "No relevant maintenance knowledge found.",
            equipment_id="Not specified",
            issue_type="Not specified",
            severity="Not assessed",
            diagnosis="Pending investigation",
            recommended_parts="None yet",
            session_context="",
        ),
        checkpointer=checkpointer,
        state_schema=MaintenanceAssistantState,
    )


def _extract_user_message(request: ResponsesAgentRequest) -> str:
    """Extract the last user message from the Responses API input."""
    for item in reversed(request.input):
        if hasattr(item, "role") and item.role == "user":
            content = item.content
            return content if isinstance(content, str) else str(content)
    return ""


############################################
# Custom table helpers (best-effort)
############################################
async def _setup_session(request: ResponsesAgentRequest, thread_id: str):
    """
    Set up custom tables, resolve technician + work order.
    Returns (thread_id, tech_id, user_id) or (thread_id, None, user_id) if DB unavailable.
    """
    ci = dict(request.custom_inputs or {})
    user_id = ci.get("user_id", "anonymous")
    equipment_id = ci.get("equipment_id")

    try:
        ensure_schema, get_or_create_technician, create_work_order, get_work_order, _, _ = _import_db()
        await ensure_schema()
        tech = await get_or_create_technician(user_id)
        tech_id = str(tech["id"])

        wo = await get_work_order(thread_id)
        if wo is None:
            wo = await create_work_order(
                technician_id=tech_id,
                equipment_id=equipment_id,
            )
            thread_id = str(wo["id"])

        return thread_id, tech_id, user_id
    except Exception as e:
        logger.warning(f"Custom table setup failed (continuing without): {e}")
        return thread_id, None, user_id


async def _post_turn_tracking(
    thread_id: str,
    user_message: str,
    response_text: str,
    max_tokens: int = 8000,
):
    """Log events and run compaction check. Best-effort — failures are logged, not raised."""
    try:
        compact_conversation, count_tokens = _import_compaction()
        _, _, _, _, log_event, _ = _import_db()

        user_tokens = count_tokens(user_message)
        response_tokens = count_tokens(response_text)

        await log_event(thread_id, "user", user_message, token_count=user_tokens)
        await log_event(thread_id, "assistant", response_text, token_count=response_tokens)

        result = await compact_conversation(thread_id, max_tokens=max_tokens)
        if result.get("compacted"):
            logger.info(
                f"Compacted {thread_id}: {result['tokens_before']} -> {result['tokens_after']} tokens"
            )
    except Exception as e:
        logger.warning(f"Post-turn tracking failed: {e}")


############################################
# @invoke / @stream handlers
############################################
@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    thread_id = _get_or_create_thread_id(request)
    request.custom_inputs = dict(request.custom_inputs or {})
    request.custom_inputs["thread_id"] = thread_id

    outputs = []
    async for event in stream_handler(request):
        if event.type == "response.output_item.done":
            outputs.append(event.item)

    # Post-turn tracking (best-effort)
    response_text = ""
    for item in outputs:
        if hasattr(item, "content"):
            for c in item.content:
                if hasattr(c, "text"):
                    response_text += c.text

    user_message = _extract_user_message(request)
    if user_message and response_text:
        max_tokens = 8000
        try:
            _, _, _, get_work_order, _, _ = _import_db()
            wo = await get_work_order(thread_id)
            if wo:
                max_tokens = wo["max_tokens"]
        except Exception:
            pass
        await _post_turn_tracking(thread_id, user_message, response_text, max_tokens)

    return ResponsesAgentResponse(
        output=outputs,
        custom_outputs={"thread_id": thread_id},
    )


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = _get_or_create_thread_id(request)
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    user_message = _extract_user_message(request)

    # Set up custom tables + resolve technician (best-effort)
    thread_id, tech_id, user_id = await _setup_session(request, thread_id)

    # Retrieve long-term memories (best-effort)
    memory_context = ""
    if tech_id and user_message:
        try:
            format_memories_for_prompt, retrieve_memories_hybrid = _import_memory()
            memories = await retrieve_memories_hybrid(tech_id, user_message, limit=5)
            memory_context = format_memories_for_prompt(memories)
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

    # Build session context from past events (best-effort)
    session_context = ""
    if tech_id:
        try:
            _, _, _, _, _, get_events = _import_db()
            past_events = await get_events(thread_id)
            if past_events:
                from agent_server.compaction import get_context_messages
                session_context = f"\nSession history:\n{get_context_messages(past_events)}"
        except Exception as e:
            logger.warning(f"Event retrieval failed: {e}")

    # Inject tech_id into config so memory tools can access it
    config = {
        "configurable": {
            "thread_id": thread_id,
            "custom_inputs": {
                "technician_id": tech_id or "",
                "user_id": user_id,
                "work_order_id": thread_id,
            },
        }
    }
    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "custom_inputs": dict(request.custom_inputs or {}),
    }

    try:
        if _has_lakebase:
            async with AsyncCheckpointSaver(
                instance_name=LAKEBASE_INSTANCE_NAME,
                autoscaling_endpoint=LAKEBASE_ENDPOINT,
                project=LAKEBASE_PROJECT if not LAKEBASE_ENDPOINT else None,
                branch=LAKEBASE_BRANCH if not LAKEBASE_ENDPOINT else None,
            ) as checkpointer:
                await checkpointer.setup()
                agent = await init_agent(
                    checkpointer=checkpointer, memory_context=memory_context
                )
                async for event in process_agent_astream_events(
                    agent.astream(input_state, config, stream_mode=["updates", "messages"])
                ):
                    yield event
        else:
            agent = await init_agent(memory_context=memory_context)
            async for event in process_agent_astream_events(
                agent.astream(input_state, config, stream_mode=["updates", "messages"])
            ):
                yield event

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in [
            "lakebase", "pg_hba", "postgres", "database instance",
            "password authentication", "couldn't get a connection", "connection pool",
        ]):
            logger.error(f"Lakebase checkpoint error, falling back to stateless: {e}")
            agent = await init_agent(memory_context=memory_context)
            async for event in process_agent_astream_events(
                agent.astream(input_state, config, stream_mode=["updates", "messages"])
            ):
                yield event
        else:
            raise

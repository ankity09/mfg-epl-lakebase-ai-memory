# MaintBot Demo Walkthrough

A step-by-step guide to demonstrate all MaintBot features. Use this in the chat UI or via the `databricks-openai` Python client.

> **Tip:** When using the chat UI, pass `user_id` in `custom_inputs` via the API. In the chat UI, the agent uses "anonymous" by default. For the full memory demo, use the Python client with `custom_inputs: {"user_id": "tech-001"}`.

---

## Part 1: Diagnostic Tools + Short-Term Memory

### Message 1 — Equipment Detection + Severity + Classification

**Send:**
```
I'm tech-001. I have a hydraulic pressure drop on HP-L4-001, severity is high.
The secondary cylinder O-rings might be worn.
```

**What to observe:**
- `detect_equipment` tool fires → detects `HP-L4-001`
- `assess_severity` tool fires → returns `high`
- `classify_issue_type` tool fires → returns `hydraulic`
- `get_maintenance_memory` tool fires → searches for past knowledge (may be empty on first run)
- Agent provides structured diagnostic response with safety warnings

### Message 2 — Multi-Turn Context (Short-Term Memory)

**Send:**
```
The pressure gauge reads 50 PSI below normal and I can hear pump cavitation.
What should I check first?
```

**What to observe:**
- Agent remembers HP-L4-001 from the previous message (short-term memory via AsyncCheckpointSaver)
- Response references the ongoing diagnostic context
- Agent pivots from O-ring diagnosis to cavitation root cause analysis
- Recommends checking fluid level, suction strainer, inlet filter

**How to verify short-term memory works:**
The response should reference "HP-L4-001" and the hydraulic issue without you repeating it. If the agent asks "what equipment?" — short-term memory isn't working (check Lakebase checkpoint connection).

---

## Part 2: Long-Term Memory (Save + Recall)

### Message 3 — Save a Finding to Long-Term Memory

**Send:**
```
We found the root cause - the suction strainer on HP-L4-001 was completely clogged
with metal shavings from upstream component wear. We replaced the strainer (P/N: STR-HP4-200),
flushed all hydraulic lines, and replaced the fluid. Pressure is back to normal at 2200 PSI.
Please save this finding for future reference.
```

**What to observe:**
- `save_maintenance_memory` tool fires with the conversation text
- The tool extracts structured memories (failure_mode, procedure, part_preference)
- Response confirms memories were saved with count

**Example expected tool output:**
```
Memory extraction complete: 3 items found, 3 new memories stored, 0 consolidated with existing knowledge.
```

### Message 4 — Recall Memory in a New Session

Start a **new conversation** (refresh the page or use a new API call without `thread_id`).

**Send:**
```
I'm tech-001. We're seeing pressure issues on HP-L4-001 again.
Any past maintenance history I should know about?
```

**What to observe:**
- `get_maintenance_memory` tool fires with `user_id: "tech-001"` and `query: "HP-L4-001 pressure issues"`
- Tool returns the saved memory about the suction strainer + metal shavings
- Agent cites the past finding: "Based on past experience, HP-L4-001 had a clogged suction strainer..."
- Response recommends checking the strainer first based on history

**Example expected tool output:**
```
1. [failure_mode] [HP-L4-001] HP-L4-001 suction strainer was clogged with metal shavings
   causing pump cavitation and 50 PSI pressure drop (relevance: 0.73)
```

**If memory recall fails (returns "No relevant maintenance knowledge found"):**
- Check that `user_id` is being passed correctly (must match the user who saved)
- Verify memories exist: query `SELECT * FROM maint_bot.maintenance_knowledge` in SQL Editor
- Check that the `vector` extension is installed: `SELECT extname FROM pg_extension`
- Check logs at `/logz` for "Memory retrieval error" messages

---

## Part 3: GDPR Controls

### Message 5 — View All Stored Memories

**Send:**
```
What data do you have stored about me? I want to see all my maintenance memories.
```

**What to observe:**
- `gdpr_view_memories` tool fires with `user_id: "tech-001"`
- Returns a list of all stored memories with IDs, types, equipment tags, and importance scores
- Response formats the data in a readable way

### Message 6 — Delete a Specific Memory

**Send (use an actual memory ID from the previous response):**
```
Please delete memory <MEMORY-ID-FROM-ABOVE>. I want to exercise my right to erasure.
```

**What to observe:**
- `gdpr_delete_memories` tool fires with the specific memory ID
- Memory is soft-deleted (is_active=FALSE)
- Response confirms deletion

### Message 7 — Export All Data

**Send:**
```
I need a full export of all my stored data in portable format.
```

**What to observe:**
- `gdpr_export_memories` tool fires
- Returns JSON with all user data (GDPR Art. 20 data portability)

---

## Part 4: Cross-Equipment Context Switching

### Message 8 — Switch to Different Equipment

**Send:**
```
Now I need to look at CNC-L2-005. It's making a grinding noise during the Z-axis
rapid traverse. Severity is critical - we had to stop production.
```

**What to observe:**
- `detect_equipment` detects `CNC-L2-005` (new equipment)
- `assess_severity` returns `critical`
- `classify_issue_type` returns `mechanical` (grinding noise)
- Agent provides CNC-specific diagnostic guidance
- Safety warnings for critical severity

---

## Verifying Features via Database

Connect to the Lakebase endpoint (SQL Editor or psql) and run:

```sql
-- Set search path
SET search_path TO maint_bot, public;

-- Check technicians created
SELECT id, external_id, created_at FROM technicians;

-- Check work orders (one per conversation)
SELECT id, technician_id, equipment_id, status, created_at FROM work_orders ORDER BY created_at DESC;

-- Check stored memories (long-term)
SELECT id, content, memory_type, equipment_tag, importance, is_active, created_at
FROM maintenance_knowledge ORDER BY created_at DESC;

-- Check event log (token tracking)
SELECT work_order_id, role, token_count, LEFT(content, 80) as preview, created_at
FROM work_order_events ORDER BY created_at DESC LIMIT 10;

-- Check pgvector is working (embedding dimensions)
SELECT id, equipment_tag, array_length(embedding::real[], 1) as embedding_dim
FROM maintenance_knowledge LIMIT 5;
```

---

## Verifying via Python Client

```python
from databricks_openai import DatabricksOpenAI
import time

client = DatabricksOpenAI(base_url="https://<your-app-url>")

# Test 1: Basic diagnostic
r = client.responses.create(
    input=[{"role": "user", "content": "Hydraulic pressure drop on HP-L4-001, severity high"}],
    extra_body={"custom_inputs": {"user_id": "tech-001"}},
)
thread_id = r.custom_outputs["thread_id"]
print(f"Thread: {thread_id}")

# Test 2: Multi-turn (same thread)
r2 = client.responses.create(
    input=[{"role": "user", "content": "50 PSI below normal, pump cavitation detected"}],
    extra_body={"custom_inputs": {"user_id": "tech-001", "thread_id": thread_id}},
)

# Test 3: Save memory
r3 = client.responses.create(
    input=[{"role": "user", "content": "Root cause: clogged suction strainer. Please save this."}],
    extra_body={"custom_inputs": {"user_id": "tech-001", "thread_id": thread_id}},
)

# Test 4: Recall in new session
r4 = client.responses.create(
    input=[{"role": "user", "content": "Any past history for HP-L4-001?"}],
    extra_body={"custom_inputs": {"user_id": "tech-001"}},  # No thread_id = new session
)

# Print all tool outputs
for item in r4.output:
    if item.type == "function_call_output":
        print(f"Memory: {item.output[:200]}")
    elif item.type == "message" and item.role == "assistant":
        for c in item.content:
            if hasattr(c, "text"):
                print(f"Response: {c.text[:300]}")
```

---

## Expected Response Times

| Scenario | Expected Time | What It Means |
|----------|--------------|---------------|
| ~10-15 seconds | Normal | Lakebase persistence working, LLM + tools executing |
| ~30 seconds | Concerning | Custom table operations might be timing out |
| ~60+ seconds | Problem | Lakebase auth failing, falling back to stateless mode |

If responses take 60+ seconds, check the [Troubleshooting guide](../README.md#troubleshooting).

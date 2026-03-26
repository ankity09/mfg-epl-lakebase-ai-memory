# Feature Comparison: MaintBot vs. Official Databricks Agent Templates

> Last updated: 2026-03-26

---

## 1. Overview

**Short-Term Memory Template** is the official Databricks app template for building LangGraph-based chat agents with conversation persistence. It stores per-thread checkpoints in Lakebase via `AsyncCheckpointSaver`, includes a code interpreter tool (`system.ai.python_exec`), a current-time tool, and an MLflow evaluation framework with 9 scoring metrics. It ships with a quickstart wizard, preflight checks, a `discover-tools` CLI, and a React chat frontend. Memory is thread-scoped only -- there is no cross-session recall.
Repository: [agent-langgraph-short-term-memory](https://github.com/databricks/app-templates/tree/main/agent-langgraph-short-term-memory)

**Long-Term Memory Template** extends the short-term template with `AsyncDatabricksStore` for persistent, cross-session memory. It provides three memory tools (`get_user_memory`, `save_user_memory`, `delete_user_memory`), LLM-guided memory extraction via system-prompt guidelines, namespace-based user isolation (`user_memories_{user_id}`), GTE-large-en embeddings (1024-dim), and vector + keyword hybrid search. It inherits all short-term features (quickstart, preflight, evaluation, code interpreter). There is no deduplication, consolidation, token compaction, PII scrubbing, or GDPR audit trail.
Repository: [agent-langgraph-long-term-memory](https://github.com/databricks/app-templates/tree/main/agent-langgraph-long-term-memory)

**MaintBot** is a production-grade maintenance AI assistant purpose-built for manufacturing environments. It implements both short-term and long-term memory, then adds a full layer of memory intelligence (hybrid scoring, deduplication via LLM-merge, token compaction), privacy and compliance controls (PII redaction, GDPR Art. 15/17/20), domain-specific tooling (equipment detection, severity assessment, issue classification), custom Lakebase tables (technicians, work orders, work order events), and a pluggable vector backend that can swap between pgvector and Databricks Vector Search at deploy time.
Repository: [mfg-epl-lakebase-ai-memory](https://github.com/ankity09/mfg-epl-lakebase-ai-memory)

---

## 2. Feature Comparison Matrix

### Core Agent

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| LLM provider | Databricks FMAPI | Databricks FMAPI | Databricks FMAPI |
| Agent framework | LangGraph | LangGraph | LangGraph |
| API server | FastAPI | FastAPI | FastAPI |
| React frontend | Yes | Yes | Yes (e2e-chatbot-app-next) |
| Lakebase (Postgres-compatible) | Yes | Yes | Yes |

### Short-Term Memory

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| Checkpoint persistence | `AsyncCheckpointSaver` on Lakebase | `AsyncCheckpointSaver` on Lakebase | `AsyncCheckpointSaver` on Lakebase |
| Thread-scoped conversations | Yes | Yes | Yes |
| Conversation history in UI | -- | -- | Yes (sidebar, grouped by date) |
| Cross-session recall | No | Yes (via long-term store) | Yes (via long-term memory table) |

### Long-Term Memory

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| Long-term memory store | No | `AsyncDatabricksStore` | Custom pgvector table (`maintenance_knowledge`) |
| Embeddings model | -- | GTE-large-en (1024-dim) | GTE-large-en (1024-dim) |
| Storage backend | -- | Databricks-managed | pgvector on Lakebase **or** Databricks Vector Search |
| Pluggable vector backend | -- | No (Databricks only) | Yes (`MEMORY_BACKEND` env var) |
| Memory tools (save/get/delete) | No | 3 tools | Yes + GDPR-specific tools |
| Namespace-based user isolation | -- | Yes (`user_memories_{user_id}`) | Yes (user_id column + query filter) |
| Retrieval method | -- | Vector + keyword hybrid search | Vector cosine similarity + hybrid scoring |
| Top-K retrieval | -- | Top 5 | Configurable |
| HNSW index for ANN | -- | No | Yes |

### Memory Intelligence

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| LLM-driven memory extraction | No | Yes (system prompt guidelines) | Yes (structured JSON output) |
| Typed memory categories | -- | No (freeform) | 5 types: failure_mode, part_preference, procedure, machine_quirk, vendor_info |
| Equipment tagging on memories | No | No | Yes (equipment_tag field) |
| Importance scoring | No | No | Yes (0.0--1.0 per memory) |
| Deduplication | No | No | Yes (cosine > 0.85 threshold) |
| LLM-merge consolidation | No | No | Yes (merges duplicates into consolidated statements) |
| Importance escalation on merge | No | No | Yes (+0.1 on consolidation) |
| Access count tracking | No | No | Yes (access_count incremented) |
| Hybrid relevance scoring | No | No | Yes (0.7 cosine + 0.2 recency + 0.1 importance) |
| Recency decay | No | No | Yes (30-day half-life) |

### Context Management

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| Token counting | No | No | Yes (per-message token_count) |
| Token compaction | No | No | Yes (3 strategies: summarize, last_n, truncate) |
| Compaction threshold | -- | -- | 0.8 * max_tokens |
| Protected recent turns | -- | -- | Last 5 turns uncompacted |
| LLM-based summarization | -- | -- | Yes (older messages summarized, marked is_compacted) |
| Per-work-order max_tokens | -- | -- | Yes (configurable on work_orders table) |

### Privacy & Compliance

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| PII redaction before storage | No | No | Yes (5 regex patterns) |
| Phone number scrubbing | No | No | Yes |
| Email scrubbing | No | No | Yes |
| SSN scrubbing | No | No | Yes |
| Badge number scrubbing | No | No | Yes |
| Phone extension scrubbing | No | No | Yes |
| GDPR Art. 15 (right of access) | No | No | Yes (`view_memories`) |
| GDPR Art. 17 (right to erasure) | No | Partial (`delete_user_memory`) | Yes (soft delete via `is_active=FALSE`) |
| GDPR Art. 20 (data portability) | No | No | Yes (`export_memories` -- JSON) |
| Audit trail for deletions | No | No | Yes (soft delete preserves records) |

### Domain Features

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| Custom database tables | No | No | Yes (technicians, work_orders, work_order_events) |
| Technician profiles | No | No | Yes (UUID, external_id, specializations JSONB) |
| Work order management | No | No | Yes (equipment_id, priority, status, max_tokens) |
| Work order event log | No | No | Yes (role, content, token_count, is_compacted, user_id) |
| Equipment detection | No | No | Yes (XX-LN-NNN regex) |
| Severity assessment | No | No | Yes (keyword-based: critical/high/medium/low) |
| Issue type classification | No | No | Yes (hydraulic/electrical/mechanical/software/pneumatic) |
| Code interpreter tool | Yes (`system.ai.python_exec`) | Yes (`system.ai.python_exec`) | No |
| Current time tool | Yes | Yes | No |

### Vector Backend

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| pgvector on Lakebase | Checkpoints only | Checkpoints only | Yes (primary) |
| Databricks Vector Search | No | Yes (via AsyncDatabricksStore) | Yes (alternative backend) |
| Pluggable backend routing | No | No | Yes (`memory_router.py`) |
| Timing instrumentation | No | No | Yes (router adds latency metrics) |
| HNSW approximate NN index | No | No | Yes |

### Deployment & Developer Experience

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| Quickstart wizard | Yes | Yes | No |
| Preflight checks | Yes | Yes | No |
| `discover-tools` CLI | Yes | Yes | No |
| MLflow evaluation framework | Yes (9 scoring metrics) | Yes (9 scoring metrics) | No |
| Databricks Apps deployment | Yes | Yes | Yes |
| DABs bundle support | Yes | Yes | Yes |
| Graceful degradation | No | No | Yes (stateless fallback if Lakebase unavailable) |
| User identity resolution | Basic (Databricks OAuth) | Basic (Databricks OAuth) | Multi-layer (OAuth > x-forwarded-email > custom_inputs > anonymous) |

### Frontend

| Feature | Short-Term Template | Long-Term Template | MaintBot |
|---|:---:|:---:|:---:|
| Chat interface | React chat UI | React chat UI | e2e-chatbot-app-next |
| Conversation history sidebar | No | No | Yes (grouped by date) |
| User-isolated chat history | No | No | Yes |

---

## 3. What MaintBot Does Better

1. **Hybrid relevance scoring.** Memories are ranked by a weighted formula (`0.7 * cosine_similarity + 0.2 * recency_decay + 0.1 * importance_weight`) instead of raw vector distance alone. This means recent, high-importance memories surface above stale ones even when cosine similarity is close.

2. **Memory deduplication and LLM-merge consolidation.** When a new memory has cosine similarity > 0.85 with an existing one, the LLM merges them into a single consolidated statement rather than storing duplicates. This keeps the memory store clean and prevents contradictory recall.

3. **Token compaction with three strategies.** Instead of letting conversation context grow unbounded, MaintBot monitors token usage per work order and compacts older turns via summarization, last-N windowing, or truncation when the context hits 80% of the configured limit. The last 5 turns are always preserved uncompacted to maintain conversational coherence.

4. **PII redaction at the storage boundary.** Five regex patterns strip phone numbers, emails, SSNs, badge numbers, and phone extensions from all text before it reaches the database. Neither template has any PII handling -- whatever the user types goes straight into storage.

5. **GDPR compliance (Art. 15, 17, 20).** MaintBot implements right-of-access (view all stored memories), right-to-erasure (soft delete with `is_active=FALSE` to preserve audit trails), and data portability (JSON export). The long-term template only has a basic delete tool with no audit trail or export.

6. **Pluggable vector backend.** A single environment variable (`MEMORY_BACKEND`) switches between pgvector on Lakebase and Databricks Vector Search. Both expose an identical async interface via `memory_router.py`, with built-in timing instrumentation. The templates are locked to a single backend.

7. **Domain-specific diagnostic tools.** Equipment detection (XX-LN-NNN regex parsing), severity assessment (keyword-based classification into critical/high/medium/low), and issue-type classification (hydraulic, electrical, mechanical, software, pneumatic) give the agent structured understanding of maintenance context. The templates have no domain tools.

8. **Custom relational schema.** Technician profiles (with specializations JSONB), work orders (with priority, status, equipment linkage, configurable max_tokens), and work order events (with per-message token counts and compaction flags) model the maintenance domain properly. The templates rely entirely on LangGraph's checkpoint and store abstractions with no custom tables.

9. **Graceful degradation.** All database operations are wrapped in try/except blocks so the agent continues to function in a stateless mode if Lakebase is unavailable. The templates will crash if their database connection fails.

10. **Multi-layer user identity resolution.** MaintBot resolves the user from Databricks Apps OAuth (`request.context.user_id`), then falls back to `x-forwarded-email` header, then `custom_inputs`, then `"anonymous"`. The templates use only the OAuth user context with no fallback chain.

---

## 4. What the Templates Have That MaintBot Doesn't

- **Quickstart wizard.** The templates ship an interactive setup flow that walks new developers through configuration, secret provisioning, and first-run validation. MaintBot requires manual setup.

- **Preflight checks.** Before the agent starts, the templates validate that secrets are set, endpoints are reachable, and the database is provisioned. MaintBot relies on graceful degradation at runtime instead of upfront validation.

- **`discover-tools` CLI.** A command-line utility that introspects Unity Catalog and lists available tools the agent can bind to. MaintBot has its tools hardcoded.

- **MLflow evaluation framework.** The templates include a built-in evaluation harness with 9 scoring metrics (relevance, groundedness, safety, etc.) integrated with MLflow Experiments. MaintBot has no automated evaluation pipeline.

- **Code interpreter tool (`system.ai.python_exec`).** The templates can execute arbitrary Python code during a conversation, useful for data analysis, plotting, and computation. MaintBot does not include this capability.

- **Current time tool.** A simple utility tool that returns the current timestamp. MaintBot does not expose this as a standalone tool.

- **Broader generality.** The templates are domain-agnostic and designed to be extended for any use case. MaintBot is purpose-built for manufacturing maintenance and would require refactoring to serve a different domain.

---

## 5. Architecture Comparison

### Short-Term Memory Template

```
┌─────────────────────────────────────────────────┐
│                  React Chat UI                  │
└──────────────────────┬──────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────┐
│                   FastAPI                       │
│  ┌────────────────────────────────────────────┐ │
│  │              LangGraph Agent               │ │
│  │                                            │ │
│  │  Tools:                                    │ │
│  │    - system.ai.python_exec                 │ │
│  │    - current_time                          │ │
│  └──────────────┬─────────────────────────────┘ │
│                 │                                │
│  ┌──────────────▼─────────────────────────────┐ │
│  │       AsyncCheckpointSaver                 │ │
│  │       (thread-scoped only)                 │ │
│  └──────────────┬─────────────────────────────┘ │
└─────────────────┼────────────────────────────────┘
                  │ SQL
┌─────────────────▼────────────────────────────────┐
│              Lakebase (Postgres)                  │
│  ┌────────────────────────────────────────────┐  │
│  │         checkpoint tables                  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Long-Term Memory Template

```
┌─────────────────────────────────────────────────┐
│                  React Chat UI                  │
└──────────────────────┬──────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────┐
│                   FastAPI                       │
│  ┌────────────────────────────────────────────┐ │
│  │              LangGraph Agent               │ │
│  │                                            │ │
│  │  Tools:                                    │ │
│  │    - system.ai.python_exec                 │ │
│  │    - current_time                          │ │
│  │    - get_user_memory                       │ │
│  │    - save_user_memory                      │ │
│  │    - delete_user_memory                    │ │
│  └─────┬────────────────────────┬─────────────┘ │
│        │                        │                │
│  ┌─────▼──────────┐  ┌─────────▼─────────────┐ │
│  │ Async           │  │ AsyncDatabricksStore  │ │
│  │ Checkpoint      │  │ (long-term memory)    │ │
│  │ Saver           │  │                       │ │
│  │ (short-term)    │  │ - GTE-large-en embeds │ │
│  └─────┬───────────┘  │ - vector+keyword      │ │
│        │              │ - namespace isolation  │ │
│        │              └─────────┬──────────────┘ │
└────────┼────────────────────────┼────────────────┘
         │ SQL                    │ API
┌────────▼──────────┐  ┌─────────▼──────────────┐
│    Lakebase       │  │  Databricks Vector     │
│    (Postgres)     │  │  Search (managed)      │
│  ┌─────────────┐  │  └────────────────────────┘
│  │ checkpoint  │  │
│  │ tables      │  │
│  └─────────────┘  │
└───────────────────┘
```

### MaintBot

```
┌─────────────────────────────────────────────────────┐
│            e2e-chatbot-app-next Frontend            │
│  ┌───────────────────────────────────────────────┐  │
│  │  Chat UI + Conversation History Sidebar       │  │
│  │  (grouped by date, user-isolated)             │  │
│  └──────────────────────┬────────────────────────┘  │
└─────────────────────────┼───────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────┐
│                      FastAPI                        │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │               LangGraph Agent                  │ │
│  │                                                │ │
│  │  Domain Tools:           Memory Tools:         │ │
│  │    - detect_equipment      - view_memories     │ │
│  │    - assess_severity       - delete_memory     │ │
│  │    - classify_issue_type   - delete_all        │ │
│  │                            - export_memories   │ │
│  └──┬───────────┬──────────────────┬──────────────┘ │
│     │           │                  │                 │
│  ┌──▼────────┐  │  ┌──────────────▼──────────────┐  │
│  │ PII       │  │  │   Memory Intelligence       │  │
│  │ Redaction │  │  │                              │  │
│  │ (5 regex) │  │  │  - LLM extraction (5 types) │  │
│  └──┬────────┘  │  │  - Dedup (cosine > 0.85)    │  │
│     │           │  │  - LLM-merge consolidation   │  │
│     │           │  │  - Hybrid scoring            │  │
│     │           │  │    (0.7cos+0.2rec+0.1imp)    │  │
│     │           │  └──────────────┬───────────────┘  │
│     │           │                 │                   │
│  ┌──▼───────────▼─────────────────▼───────────────┐  │
│  │          memory_router.py                      │  │
│  │   (dispatches by MEMORY_BACKEND env var)       │  │
│  │   (adds timing instrumentation)                │  │
│  └──────┬──────────────────────────┬──────────────┘  │
│         │                          │                  │
│  ┌──────▼───────────┐  ┌──────────▼───────────────┐  │
│  │ AsyncCheckpoint  │  │  Vector Backend          │  │
│  │ Saver            │  │  ┌─────────────────────┐ │  │
│  │ (short-term)     │  │  │ pgvector (default)  │ │  │
│  └──────┬───────────┘  │  │ + HNSW index        │ │  │
│         │              │  ├─────────────────────┤ │  │
│         │              │  │ Databricks VS       │ │  │
│         │              │  │ (alternative)       │ │  │
│         │              │  └────────┬────────────┘ │  │
│  ┌──────▼──────────────┼──────────▼────────────┐  │  │
│  │  Token Compaction   │  (graceful degrade)   │  │  │
│  │  - summarize        │                       │  │  │
│  │  - last_n           │                       │  │  │
│  │  - truncate         │                       │  │  │
│  │  (0.8 * max_tokens) │                       │  │  │
│  └──────┬──────────────┘───────────────────────┘  │  │
└─────────┼─────────────────────────────────────────┘  │
          │ SQL                                        │
┌─────────▼────────────────────────────────────────────┘
│                  Lakebase (Postgres + pgvector)
│
│  ┌──────────────────┐  ┌──────────────────────────┐
│  │ checkpoint       │  │ maintenance_knowledge    │
│  │ tables           │  │ (1024-dim, HNSW index)   │
│  └──────────────────┘  └──────────────────────────┘
│
│  ┌──────────────────┐  ┌──────────────────────────┐
│  │ technicians      │  │ work_orders              │
│  │ (UUID, specs     │  │ (equipment_id, priority, │
│  │  JSONB)          │  │  status, max_tokens)     │
│  └──────────────────┘  └──────────────────────────┘
│
│  ┌──────────────────────────────────────────────────┐
│  │ work_order_events                                │
│  │ (role, content, token_count, is_compacted,       │
│  │  user_id)                                        │
│  └──────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────┘
```

---

## 6. Recommendation: When to Use Each

### Use the **Short-Term Memory Template** when:
- You need a quick proof-of-concept for a chat agent on Databricks Apps.
- Conversation persistence within a single session is sufficient.
- You want built-in MLflow evaluation and the quickstart wizard to get running fast.
- Your use case is general-purpose (no domain-specific memory or compliance requirements).
- You want the code interpreter tool for data analysis conversations.

### Use the **Long-Term Memory Template** when:
- You need cross-session memory so the agent remembers users over time.
- The Databricks-managed vector store (`AsyncDatabricksStore`) fits your infrastructure.
- Simple save/get/delete memory operations are sufficient.
- You do not need deduplication, consolidation, token management, or compliance controls.
- You want to get started with long-term memory without building custom infrastructure.

### Use **MaintBot** when:
- You are building for a **regulated or compliance-sensitive environment** (PII, GDPR).
- You need **production-grade memory management**: deduplication, consolidation, hybrid scoring, and token compaction to control costs and context quality at scale.
- Your domain requires **custom relational tables** alongside vector memory (technicians, work orders, event logs).
- You need **pluggable infrastructure**: the ability to switch vector backends between pgvector and Databricks Vector Search without code changes.
- **Graceful degradation** matters -- the agent must stay functional even if the database is temporarily unavailable.
- You want a **reference implementation** showing how to extend the official templates into a production-ready, domain-specific AI application.

### Migration path
MaintBot was designed as a superset of the official templates. If you start with a template and outgrow it, the architecture patterns in MaintBot (memory router, PII pipeline, token compaction, GDPR tools) can be adopted incrementally. The shared foundation -- LangGraph + FastAPI + Lakebase + `AsyncCheckpointSaver` -- means checkpoint-level compatibility is maintained.

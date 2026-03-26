# Memory Backend Testing Guide

This guide covers how to test, compare, and benchmark the two memory backends available in MaintBot: **pgvector** (Lakebase Postgres) and **Databricks Vector Search**.

---

## 1. Memory Backend Overview

MaintBot supports two pluggable backends for long-term memory (the `maintenance_knowledge` table/index):

| Backend | Storage | Retrieval | Best For |
|---------|---------|-----------|----------|
| **pgvector** (default) | Lakebase Postgres with `vector(1024)` column | SQL `ORDER BY embedding <=> vector` + HNSW index | Small-medium scale (<5M vectors), ACID consistency, low-latency local queries |
| **databricks_vs** | Databricks Vector Search Direct Access Index | `index.similarity_search()` API | Large scale (>10M vectors), managed infrastructure, production SLA |

### Architecture

```
User Request
    │
    ▼
agent.py  ──→  memory_router.py  ──→  MEMORY_BACKEND env var
                    │                       │
                    ├── "pgvector"  ────→  memory.py (Lakebase pgvector)
                    │                       └── SQL: INSERT, SELECT ORDER BY <=>
                    │
                    └── "databricks_vs" ──→  memory_vs.py (Vector Search SDK)
                                             └── API: index.upsert(), index.similarity_search()

Both backends:
  ├── Use same GTE-large-en embeddings (1024-dim)
  ├── Apply same hybrid scoring (0.7×similarity + 0.2×recency + 0.1×importance)
  ├── Share same PII redaction (pii.py)
  ├── Share same LLM extraction logic (memory.py:extract_memories)
  └── Expose identical async interface to memory_tools.py
```

### When to Use Each

- **pgvector**: Default for demos and development. Zero additional infrastructure. Transactions ensure consistency. Works offline with local Lakebase.
- **databricks_vs**: Use when testing Vector Search as a product feature, benchmarking at scale, or building demos that showcase Databricks Vector Search capabilities.

---

## 2. How to Switch Backends

### Local Development

Set the `MEMORY_BACKEND` environment variable in your `.env` file:

```bash
# Use pgvector (default)
MEMORY_BACKEND=pgvector

# Use Databricks Vector Search
MEMORY_BACKEND=databricks_vs
```

Then restart the server:

```bash
uv run start-server --port 8000
```

### Deployed App (Databricks Apps)

1. Edit `databricks.yml` and `app.yaml`:

```yaml
# In both files, change the MEMORY_BACKEND value:
- name: MEMORY_BACKEND
  value: "databricks_vs"   # or "pgvector"
```

2. Deploy and restart:

```bash
databricks bundle deploy --profile <your-profile>
databricks bundle run maint_bot_app --profile <your-profile>
```

### How to Verify Which Backend is Active

Check the app logs for the `[MEMORY:xxx]` log lines emitted by `memory_router.py`:

```
# pgvector backend:
[MEMORY:pgvector] retrieve_memories: 45.2ms, 5 results

# Vector Search backend:
[MEMORY:databricks_vs] retrieve_memories: 28.1ms, 5 results
```

On startup, the router also logs:
```
Memory backend: pgvector    # or databricks_vs
```

---

## 3. Vector Search Setup (for `databricks_vs` backend)

### Prerequisites

- Databricks workspace with **Unity Catalog** and **Serverless** enabled
- A UC catalog and schema (e.g., `ankit_yadav.maint_bot`)
- The app's service principal needs UC permissions

### Step 1: Provision Infrastructure

Run the setup script (requires `databricks-vectorsearch` installed):

```bash
uv run python scripts/setup_vector_search.py
```

Or provision manually via CLI:

```bash
# Create endpoint (if needed)
databricks vector-search-endpoints create-endpoint <endpoint-name> STANDARD \
  --profile <your-profile>

# Create Direct Vector Access index
databricks vector-search-indexes create-index \
  --json '{
    "name": "<catalog>.<schema>.maint_knowledge_vs",
    "endpoint_name": "<endpoint-name>",
    "primary_key": "id",
    "index_type": "DIRECT_ACCESS",
    "direct_access_index_spec": {
      "embedding_vector_columns": [{"name": "embedding", "embedding_dimension": 1024}],
      "schema_json": "{\"id\": \"string\", \"technician_id\": \"string\", \"content\": \"string\", \"memory_type\": \"string\", \"equipment_tag\": \"string\", \"importance\": \"float\", \"is_active\": \"string\", \"access_count\": \"int\", \"created_at\": \"string\", \"updated_at\": \"string\", \"source_work_order_id\": \"string\", \"embedding\": \"array<float>\"}"
    }
  }' --profile <your-profile>
```

### Step 2: Grant UC Permissions to App SP

```bash
# Get app SP ID
SP_APP_ID=$(databricks apps get <app-name> --profile <profile> -o json | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['service_principal_client_id'])")

# Grant catalog access
databricks grants update catalog <catalog> \
  --json "{\"changes\": [{\"principal\": \"$SP_APP_ID\", \"add\": [\"USE_CATALOG\"]}]}" \
  --profile <profile>

# Grant schema access
databricks grants update schema <catalog>.<schema> \
  --json "{\"changes\": [{\"principal\": \"$SP_APP_ID\", \"add\": [\"USE_SCHEMA\", \"SELECT\", \"MODIFY\", \"CREATE_TABLE\"]}]}" \
  --profile <profile>
```

### Step 3: Verify Index is Ready

```bash
databricks vector-search-indexes get-index <catalog>.<schema>.maint_knowledge_vs \
  --profile <profile> -o json | python3 -c \
  "import sys,json; r=json.load(sys.stdin); print(f'Ready: {r[\"status\"][\"ready\"]}')"
```

Wait until `Ready: True` before testing.

### Step 4: Optional — Migrate Existing pgvector Data

```bash
uv run python scripts/setup_vector_search.py --migrate
```

---

## 4. Test Scenarios

All tests use the same 6 questions against the deployed app. Replace `<APP_URL>` and `<TOKEN>` with your values.

### Getting a Token

```bash
TOKEN=$(databricks auth token --profile <your-profile> 2>&1 | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
APP="https://<your-app-url>"
```

### Test 1: Equipment Diagnostic (New Thread)

**Purpose**: Test equipment detection, severity assessment, and memory retrieval.

```bash
curl -s -X POST "$APP/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "The CNC-500X spindle is making a grinding noise during heavy cuts and we measured the motor temperature at 185F. What should I check first?"}], "custom_inputs": {"thread_id": "test-001"}}'
```

**What to look for**:
- `detect_equipment` fires (may not detect "CNC-500X" since it doesn't match XX-LN-NNN pattern)
- `assess_severity` returns `medium` (no explicit severity keywords)
- `classify_issue_type` returns `undetermined` (no exact type keyword match)
- `get_maintenance_memory` fires with query about CNC-500X
- Response provides systematic diagnostic steps

### Test 2: Multi-Turn Follow-Up (Same Thread)

**Purpose**: Test short-term memory (AsyncCheckpointSaver).

```bash
curl -s -X POST "$APP/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Yes there is definite radial play. About 0.003 inches. The bearing preload seems off. What should I do?"}], "custom_inputs": {"thread_id": "test-001"}}'
```

**What to look for**:
- Agent references CNC-500X from previous message (short-term memory working)
- Response discusses bearing replacement specific to that machine
- No need to repeat equipment ID

**Pass criteria**: Agent says "CNC-500X" or references the spindle without being told again.

### Test 3: Save to Long-Term Memory

**Purpose**: Test memory extraction, PII redaction, and storage.

```bash
curl -s -X POST "$APP/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "We ended up replacing the bearings in-house. Took about 5 hours. Key learning: always check bearing preload first before replacing the whole spindle unit. Also the firmware v3.2.1 was causing false temperature alarms, so we updated to v3.2.3. Please save all of this to memory."}], "custom_inputs": {"thread_id": "test-001"}}'
```

**What to look for**:
- `save_maintenance_memory` tool fires
- Result shows extracted items (expect 3-7 depending on LLM interpretation)
- `stored` count > 0, `consolidated` may be > 0 if similar memories exist

**Pass criteria**: `Memory extraction complete: N items found, M new memories stored`

### Test 4: Long-Term Memory Recall (New Thread)

**Purpose**: Test cross-session memory retrieval.

```bash
curl -s -X POST "$APP/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "My CNC-500X spindle has a temperature alarm going off. The display shows 190F. Is this a real problem or a false alarm?"}], "custom_inputs": {"thread_id": "test-002"}}'
```

**What to look for**:
- `get_maintenance_memory` returns results about CNC-500X firmware v3.2.1 false alarms
- Agent cites past experience: "Based on past experience..." or "known firmware issue"
- Response recommends checking firmware version first

**Pass criteria**: Agent mentions firmware v3.2.1 causing false alarms (recalled from Test 3).

### Test 5: GDPR View All Memories

**Purpose**: Test GDPR Art. 15 compliance.

```bash
curl -s -X POST "$APP/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Show me all the maintenance knowledge you have stored about me."}], "custom_inputs": {"thread_id": "test-003"}}'
```

**What to look for**:
- `gdpr_view_memories` tool fires
- Returns list of all memories with IDs, types, equipment tags, importance scores
- Response organizes memories by equipment

**Pass criteria**: Returns non-empty list of memories with structured metadata.

### Test 6: Cross-Equipment Overview

**Purpose**: Test broad memory recall across multiple equipment types.

```bash
curl -s -X POST "$APP/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Give me a quick overview of all the equipment issues and tips you remember from our past conversations."}], "custom_inputs": {"thread_id": "test-004"}}'
```

**What to look for**:
- Agent uses GDPR view or memory search to retrieve all known equipment
- Response covers multiple equipment types
- Information is organized by equipment ID

---

## 5. Side-by-Side Results (March 2026 Test Run)

### Test Environment

| Setting | Value |
|---------|-------|
| Workspace | `fevm-serverless-stable-trvrmk` |
| App URL | `https://maint-bot-app-7474645545773789.aws.databricksapps.com` |
| LLM | `databricks-claude-sonnet-4-5` |
| Embedding | `databricks-gte-large-en` (1024-dim) |
| pgvector index | HNSW (`vector_cosine_ops`, m=16, ef_construction=64) |
| VS endpoint | `mas-3876475e-endpoint` (STANDARD) |
| VS index | `ankit_yadav.maint_bot.maint_knowledge_vs` (Direct Access) |

### Results Comparison

| Test | pgvector | Databricks Vector Search |
|------|----------|--------------------------|
| **Q1: Diagnostic** | Tools fired correctly. Memory returned results from prior sessions (PMP-L5-012 with 0.68 relevance). | Tools fired correctly. Memory returned HP-L4-001 results from prior VS session (0.29 relevance). |
| **Q2: Multi-turn** | Short-term memory working. Referenced CNC-500X without being told. Memory retrieved prior session data (0.72). | Short-term memory working. Referenced CNC-500X without being told. Memory retrieved HP-L4-001 data (0.29). |
| **Q3: Save memory** | **7 items extracted, 5 stored, 2 consolidated** with existing knowledge. Dedup working. | **6 items extracted, 6 stored, 0 consolidated**. All new (fresh index, no prior data to consolidate). |
| **Q4: Long-term recall** | Recalled firmware v3.2.1 bug with **0.92 relevance**. Agent immediately suggested firmware check. | Recalled firmware v3.2.1 bug with **0.30 relevance**. Agent still correctly identified the false alarm possibility. |
| **Q5: GDPR view** | **18 memories** across 3+ equipment types (includes history from prior demo sessions). | **9 memories** across 2 equipment types (only from current test sessions). |
| **Q6: Overview** | Organized by CNC-500X (5), PMP-L5-012 (8), HX-L3-007 (5+). Comprehensive history. | Organized by CNC-500X (6), HP-L4-001 (3). Only current session data. |

### Key Observations

#### Relevance Scores

| Metric | pgvector | Vector Search |
|--------|----------|---------------|
| Top result relevance (Q4) | **0.92** | **0.30** |
| Typical range | 0.60 - 0.95 | 0.25 - 0.35 |
| Score differentiation | High (easy to rank) | Low (compressed range) |

**Why the difference**: pgvector returns `1 - cosine_distance` which maps naturally to 0-1 similarity. Vector Search returns a raw score that uses a different scale. Both find the correct results, but pgvector's scores are more interpretable for the hybrid scoring formula.

#### Functional Parity

Both backends successfully:
- Stored memories via `save_maintenance_memory`
- Retrieved relevant memories via `get_maintenance_memory`
- Recalled cross-session knowledge (firmware bug)
- Supported GDPR view operations
- Applied PII redaction before storage
- Used the same hybrid scoring formula

#### Differences

| Aspect | pgvector | Vector Search |
|--------|----------|---------------|
| **Deduplication** | Working — consolidated 2 memories with existing ones | Not triggered — fresh index had no prior data |
| **Score range** | 0.60 - 0.95 (wide, interpretable) | 0.25 - 0.35 (narrow, compressed) |
| **Memory count** | 18 (accumulated over multiple demo sessions) | 9 (only from these test sessions) |
| **Auth complexity** | Zero (uses same Lakebase connection) | Requires explicit SP credentials + UC grants |
| **Setup time** | Zero (pgvector extension auto-created) | ~10 min (endpoint + index provisioning) |
| **Transactions** | ACID (INSERT + SELECT in same transaction) | Eventually consistent (upsert then query) |

---

## 6. Benchmark Script

For automated latency and quality comparison, use the benchmark script:

```bash
uv run python scripts/benchmark_memory.py
```

### What It Does

1. Creates a benchmark technician (`benchmark-tech-001`)
2. Seeds both backends with 12 identical test memories (manufacturing equipment scenarios)
3. Runs 8 test queries against each backend (3 runs per query = 24 total)
4. Measures latency (p50, p95, p99, avg, min, max)
5. Compares result overlap (Jaccard similarity)
6. Prints a formatted comparison report

### How to Read the Output

```
================================================================================
MEMORY BACKEND BENCHMARK: pgvector vs Databricks Vector Search
================================================================================

Metric                       pgvector    Vector Search      Difference
----------------------------------------------------------------------
avg_ms                        45.2ms         28.1ms          -17.1ms
p50_ms                        42.0ms         25.5ms          -16.5ms
p95_ms                        68.0ms         45.2ms          -22.8ms
...

--- Result Overlap (Jaccard similarity) ---
Query                                                    Overlap  Jaccard
----------------------------------------------------------------------
HP-L4-001 is leaking hydraulic fluid...                    4/5      80%
CNC spindle making noise during operation                  3/5      60%
...

================================================================================
Latency winner: Vector Search (17.1ms faster on average)
Average result overlap (Jaccard): 72%
  > 80% = highly similar results, backends are interchangeable
  < 50% = significant differences, investigate ranking
```

### Interpreting Results

| Metric | Good | Investigate |
|--------|------|-------------|
| Latency difference | <20ms | >50ms |
| Jaccard overlap | >70% | <50% |
| p95 latency | <100ms | >200ms |
| Result count match | Both return 5 | One returns <3 |

### Common Issues

- **VS returns 0 results**: Index not ready, or UC permissions not granted to the running identity
- **pgvector is slow**: HNSW index may not be created. Check `\di` in psql.
- **Latency spikes**: First query after cold start will be slower (embedding model warmup)

---

## 7. Known Issues and Trade-offs

### Vector Search Relevance Score Compression

VS returns scores in a compressed range (~0.25-0.35) while pgvector returns scores in a wide range (~0.60-0.95). This affects the hybrid scoring formula because the similarity component has less weight differentiation in VS mode. The correct results still appear at the top, but the score gap between relevant and irrelevant results is smaller.

**Workaround**: Consider normalizing VS scores before applying the hybrid formula. Not yet implemented.

### VS Authentication in Databricks Apps

The `VectorSearchClient()` auto-detect does not work in Databricks Apps. The `memory_vs.py` implementation explicitly checks for `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` environment variables (injected by Apps) and passes them to the client.

### VS Requires UC Permissions

The app's service principal needs `USE_CATALOG`, `USE_SCHEMA`, `SELECT`, and `MODIFY` on the catalog/schema containing the VS index. These must be granted separately from the Lakebase permissions.

### pgvector ACID vs VS Eventual Consistency

pgvector operations are transactional — a memory stored in one statement is immediately visible to the next SELECT. VS uses eventual consistency — after `index.upsert()`, there may be a brief delay before `similarity_search()` finds the new vector. In practice, this delay is <1 second for Direct Access indexes.

### VS Boolean Filter Limitation

Databricks Vector Search does not support boolean column types in filters. The `is_active` field is stored as a string (`"true"`/`"false"`) instead of a boolean. Filter syntax: `is_active = "true"`.

### Index Provisioning Time

- New VS endpoints: 5-15 minutes to reach ONLINE
- New indexes on existing endpoints: 30 seconds - 2 minutes
- New indexes on new endpoints: 10+ minutes (endpoint must provision first)

**Tip**: Reuse an existing ONLINE endpoint to avoid waiting.

### Cost Considerations

| Resource | pgvector Cost | Vector Search Cost |
|----------|--------------|-------------------|
| Storage | Included in Lakebase | Endpoint unit ($) + storage |
| Compute | Lakebase connection pool | Per-query compute |
| Minimum | $0 (shared Lakebase) | 1 endpoint unit (~$0.30-0.50/day) |
| At scale | Postgres IOPS limits | Auto-scales |

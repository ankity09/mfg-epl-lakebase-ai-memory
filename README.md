# MaintBot — Manufacturing Maintenance Assistant

A stateful AI maintenance agent deployed as a **Databricks App** with conversation persistence via **Lakebase** (Databricks Postgres). Built on the official `agent-langgraph-short-term-memory` app template pattern.

## What It Does

MaintBot helps manufacturing technicians diagnose equipment issues through multi-turn conversations. It includes:

- **3 diagnostic tools**: equipment detection (`XX-LN-NNN` pattern), severity assessment, issue type classification
- **5 memory tools**: hybrid retrieval (pgvector), memory extraction, GDPR view/delete/export
- **Conversation persistence**: `AsyncCheckpointSaver` backed by Lakebase
- **Custom session tracking**: technicians, work orders, event logging with token counting
- **Compaction**: 3 strategies (summarize, last_n, truncate) to manage context windows
- **PII redaction**: 5 patterns (phone, email, SSN, badge, extension) applied before storage
- **Chat UI with history**: frontend sidebar shows past conversations, grouped by date
- **User identity resolution**: `request.context.user_id` from Databricks Apps auth, with header/custom_input fallbacks
- **Graceful degradation**: works in stateless mode if Lakebase is unavailable

## Architecture

```
Databricks App (uv + FastAPI/uvicorn)
├── MLflow AgentServer (@invoke/@stream decorators)
│   └── LangGraph agent (create_agent + tools)
│       ├── AsyncCheckpointSaver → Lakebase (databricks_postgres)
│       └── Custom tables → Lakebase (databricks_postgres.maint_bot schema)
│           ├── technicians
│           ├── work_orders
│           ├── work_order_events (token tracking + compaction)
│           └── maintenance_knowledge (pgvector, 1024-dim embeddings)
└── databricks-openai client for API access
```

## Prerequisites

- Databricks workspace with **Lakebase Autoscaling** enabled
- A Lakebase **project** and **branch** (e.g., `lakebase-mfg-epl` / `production`)
- Databricks CLI (`databricks`) configured with a profile for your workspace
- `uv` package manager installed
- Access to `databricks-claude-sonnet-4-5` and `databricks-gte-large-en` serving endpoints

## Project Structure

```
lakebase-mfg-epl-apps/
├── agent_server/
│   ├── agent.py              # @invoke/@stream handlers, MaintBot logic, tools
│   ├── start_server.py       # MLflow AgentServer bootstrap
│   ├── utils.py              # Thread ID, streaming, error helpers (from template)
│   ├── db_schema.py          # Custom tables via AsyncLakebasePool
│   ├── compaction.py         # 3 compaction strategies + token tracking
│   ├── memory.py             # pgvector hybrid retrieval, extraction, consolidation
│   ├── memory_tools.py       # LangGraph @tool wrappers for memory/GDPR
│   ├── pii.py                # PII redaction patterns
│   └── gdpr.py               # GDPR Art. 15/17/20 controls
├── scripts/
│   ├── start_app.py          # Process manager for backend+frontend, Lakebase PG env resolution
│   └── grant_lakebase_permissions.py  # 3-layer permission grants + schema/table pre-creation
├── app.yaml                  # Databricks Apps config
├── databricks.yml            # DABs bundle definition
├── pyproject.toml            # Dependencies
├── requirements.txt          # Just "uv"
└── .env.example              # Local dev environment template
```

## Deployment Guide

### Step 1: Clone and Install

```bash
git clone https://github.com/ankity09/lakebase-mfg-epl-apps.git
cd lakebase-mfg-epl-apps
uv sync
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your workspace details:

```bash
DATABRICKS_CONFIG_PROFILE=<your-profile>
MLFLOW_EXPERIMENT_ID=<your-experiment-id>
LAKEBASE_AUTOSCALING_ENDPOINT=projects/<project>/branches/<branch>/endpoints/<endpoint>
LLM_ENDPOINT=databricks-claude-sonnet-4-5
EMBEDDING_ENDPOINT=databricks-gte-large-en
```

To find your endpoint path:
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient(profile="<your-profile>")
eps = list(w.postgres.list_endpoints(parent="projects/<project>/branches/<branch>"))
print(eps[0].name)  # e.g., projects/my-project/branches/production/endpoints/primary
```

### Step 3: Update `app.yaml`

Edit `app.yaml` and replace the placeholder values:

```yaml
- name: LAKEBASE_AUTOSCALING_ENDPOINT
  value: "projects/<your-project>/branches/<your-branch>/endpoints/<your-endpoint>"
- name: LLM_ENDPOINT
  value: "databricks-claude-sonnet-4-5"   # or your preferred model
```

> **MLflow Experiment**: The app auto-creates an experiment at `/Users/<your-email>/maint-bot-app` on first startup. No manual step needed. To override, set `MLFLOW_EXPERIMENT_ID` in `app.yaml`.

### Step 4: Update `databricks.yml`

Edit `databricks.yml`:
- Set `workspace.host` under your target to your workspace URL
- Update the env vars to match your `app.yaml`

### Step 5: Deploy the Bundle

```bash
databricks bundle validate --profile <your-profile>
databricks bundle deploy --profile <your-profile>
databricks bundle run maint_bot_app --profile <your-profile>
```

Wait for the app to start. Note the app URL from the output.

### Step 6: Add Lakebase Postgres Resource (CRITICAL)

> **Autoscaling Lakebase instances require this manual step after every `bundle deploy`** because `databricks bundle deploy` resets app resources.

**Option A — Via the Databricks UI (recommended):**

1. Go to **Compute** -> **Apps** -> **maint-bot-app**
2. Click **Edit** -> Step through to **Configure resources**
3. Click **Add resource** -> **Postgres (Lakebase)**
4. Select your project, branch, `databricks_postgres` database
5. Permission: **Can connect and create**
6. Save

**Option B — Via the SDK:**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppResource, AppResourcePostgres, AppResourcePostgresPostgresPermission

w = WorkspaceClient(profile="<your-profile>")
w.apps.update(name="maint-bot-app", app=App(name="maint-bot-app", resources=[
    AppResource(name="lakebase-default", postgres=AppResourcePostgres(
        branch="projects/<project>/branches/<branch>",
        database="projects/<project>/branches/<branch>/databases/<db-id>",
        permission=AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE)),
]))
```

To find your database ID:
```python
dbs = list(w.postgres.list_databases(parent="projects/<project>/branches/<branch>"))
for db in dbs:
    detail = w.postgres.get_database(name=db.name)
    print(db.name, "->", detail.status.postgres_database)
# Use the one where postgres_database = "databricks_postgres"
```

### Step 7: Grant Lakebase Permissions (3 Layers)

Get the app's service principal client ID:
```bash
databricks apps get maint-bot-app --profile <your-profile> -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['service_principal_client_id'])"
```

Then apply all three permission layers:

#### Layer 1: Project-level ACL

In the Databricks UI: **Lakebase** -> your project -> **Settings** -> **Grant permission** -> Add the SP with **Can use**.

Or via SDK:
```python
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
w.permissions.update(
    request_object_type="database-projects",
    request_object_id="<your-project-name>",
    access_control_list=[
        AccessControlRequest(service_principal_name="<SP_CLIENT_ID>", permission_level=PermissionLevel.CAN_USE)
    ],
)
```

#### Layer 2: Postgres Role (OAuth mapping)

```bash
uv run python scripts/grant_lakebase_permissions.py <SP_CLIENT_ID> \
    --memory-type langgraph-short-term \
    --project <your-project> \
    --branch <your-branch>
```

This calls `databricks_create_role(sp_id, 'SERVICE_PRINCIPAL')` which creates a Postgres role with proper OAuth token mapping.

> **CRITICAL**: Never create the SP's Postgres role manually with `CREATE ROLE`. Always use `databricks_create_role()` via the `databricks_auth` extension. A manually-created role blocks the OAuth token mapping and causes `password authentication failed` errors.

#### Layer 3: SQL Grants

The `grant_lakebase_permissions.py` script handles this automatically. If you need to add custom schema grants:

```sql
GRANT ALL ON SCHEMA maint_bot TO "<SP_CLIENT_ID>";
ALTER DEFAULT PRIVILEGES IN SCHEMA maint_bot GRANT ALL ON TABLES TO "<SP_CLIENT_ID>";
```

### Step 8: Restart the App

After adding the resource and granting permissions, redeploy:

```bash
databricks apps deploy maint-bot-app \
    --source-code-path /Workspace/Users/<your-email>/.bundle/maint_bot_app/<target>/files \
    --profile <your-profile>
```

### Step 9: Test

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(base_url="https://<your-app-url>")

# Single turn
r = client.responses.create(
    input=[{"role": "user", "content": "I have a hydraulic pressure drop on HP-L4-001, severity is high"}],
    extra_body={"custom_inputs": {"user_id": "tech-001"}},
)
thread_id = r.custom_outputs["thread_id"]

# Multi-turn (same thread)
r2 = client.responses.create(
    input=[{"role": "user", "content": "50 PSI below normal, pump cavitation detected"}],
    extra_body={"custom_inputs": {"user_id": "tech-001", "thread_id": thread_id}},
)
```

## Local Development

```bash
cp .env.example .env
# Edit .env with your workspace profile and Lakebase config

# Backend only:
uv run start-server --port 8000

# Backend + frontend chat UI:
uv run start-app

# Test:
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Hello"}], "custom_inputs": {"user_id": "test"}}'
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `password authentication failed for user '<SP-UUID>'` | SP role created manually via `CREATE ROLE` instead of `databricks_create_role` | Drop the role (`DROP ROLE "<SP-UUID>"`) and recreate via `grant_lakebase_permissions.py` |
| `couldn't get a connection after 30.00 sec` | Postgres resource missing from app | Re-add via UI or SDK (Step 6) |
| `start-app` crashes | Frontend can't clone from GitHub in app container | Use `start-server` instead (backend-only, API at `/invocations`) |
| `databricks bundle deploy` resets resources | Known limitation for autoscaling | Re-add resource after each `bundle deploy` (Step 6) |
| Chat history sidebar says "disabled" | Frontend needs `PGDATABASE` env var to enable chat history | `start_app.py` auto-resolves this from Lakebase config. Verify `[start_app] PGHOST=...` appears in logs |
| `Custom table setup failed: permission denied for database` | SP lacks CREATE SCHEMA/TABLE privilege (normal after fresh deploy) | Run `grant_lakebase_permissions.py` to pre-create schemas and grant access |
| `PGUSER environment variable must be set` | Frontend OAuth auth requires PGUSER during npm build migration | `start_app.py` sets `PGUSER` from `DATABRICKS_CLIENT_ID` automatically; PG vars are set after build to avoid migration issues |
| Events not logged (0 rows in `work_order_events`) | Output items are dicts, not objects — `hasattr(item, "content")` fails | Fixed: use `item.get("content")` for dict-style access |
| Response takes 60+ seconds | Lakebase auth failing, agent falling back to stateless | Check logs at `/logz`, verify all 3 permission layers |
| Response takes ~15 seconds | Normal — LLM inference + tool calls | Expected behavior |

## Related Resources

MaintBot extends the official Databricks LangGraph app templates with domain-specific features, pluggable vector backends, and enterprise-grade memory management:

| Template | Description | What MaintBot Adds |
|----------|-------------|-------------------|
| [agent-langgraph-short-term-memory](https://github.com/databricks/app-templates/tree/main/agent-langgraph-short-term-memory) | Conversation persistence via AsyncCheckpointSaver | Token-aware compaction (3 strategies), custom session tracking tables, event logging |
| [agent-langgraph-long-term-memory](https://github.com/databricks/app-templates/tree/main/agent-langgraph-long-term-memory) | Long-term knowledge via AsyncDatabricksStore | Hybrid scoring (similarity + recency + importance), LLM-driven deduplication/consolidation, PII redaction, full GDPR controls, pluggable vector backend (pgvector + Databricks Vector Search) |

For a detailed feature-by-feature comparison, see [docs/FEATURE_COMPARISON.md](docs/FEATURE_COMPARISON.md).

For testing and benchmarking the memory backends, see [tests/TESTING_GUIDE.md](tests/TESTING_GUIDE.md).

## Key Design Decisions

1. **`AsyncLakebasePool` for custom tables**: Same connection mechanism as `AsyncCheckpointSaver`. Custom tables live in `databricks_postgres.maint_bot` schema (not a separate database) because `AsyncLakebasePool` hardcodes `dbname=databricks_postgres`.

2. **Lazy imports**: Custom table and memory modules are imported inside functions, not at module level. This prevents import-time failures in the app container.

3. **Best-effort DB operations**: All custom table operations (session tracking, event logging, memory retrieval) are wrapped in try/except. If Lakebase is unavailable, the agent works in stateless mode.

4. **`databricks_create_role` only**: Never use raw `CREATE ROLE` SQL for service principals. The `databricks_auth` extension's `databricks_create_role()` function sets up the OAuth token mapping required for JWT authentication.

5. **Event logging via `try/finally` + `asyncio.ensure_future()`**: The `@stream()` decorator closes async generators via `aclose()`, so code after the last `yield` never runs. Post-turn tracking is scheduled as a fire-and-forget background task in the `finally` block, which executes reliably for both the streaming (UI) and invoke (API) paths.

6. **Permission-resilient DDL**: `ensure_schema()` wraps each DDL statement (CREATE SCHEMA, CREATE TABLE, ALTER TABLE, CREATE INDEX) in individual try/except blocks. The SP typically lacks CREATE privilege on the database, but the grant script pre-creates everything, so permission errors on DDL are safe to ignore.

7. **Frontend PG env resolution**: `start_app.py` resolves `PGHOST`/`PGDATABASE`/`PGSSLMODE` from Lakebase config **after** `npm run build` (to avoid triggering migration errors) but **before** `npm run start` (so the runtime can connect). `PGUSER` is set from `DATABRICKS_CLIENT_ID` for SP OAuth auth.

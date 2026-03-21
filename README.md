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
│   ├── start_app.py          # Process manager for backend+frontend
│   └── grant_lakebase_permissions.py  # 3-layer permission grants
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
- name: MLFLOW_EXPERIMENT_ID
  value: "<your-experiment-id>"
- name: LLM_ENDPOINT
  value: "databricks-claude-sonnet-4-5"   # or your preferred model
```

### Step 4: Update `databricks.yml`

Edit `databricks.yml`:
- Set `workspace.host` under your target to your workspace URL
- Update the env vars to match your `app.yaml`

### Step 5: Create an MLflow Experiment

```bash
databricks experiments create-experiment /Users/<your-email>/maint-bot-app --profile <your-profile>
```

Note the experiment ID and update `.env`, `app.yaml`, and `databricks.yml`.

### Step 6: Deploy the Bundle

```bash
databricks bundle validate --profile <your-profile>
databricks bundle deploy --profile <your-profile>
databricks bundle run maint_bot_app --profile <your-profile>
```

Wait for the app to start. Note the app URL from the output.

### Step 7: Add Lakebase Postgres Resource (CRITICAL)

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

### Step 8: Grant Lakebase Permissions (3 Layers)

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

### Step 9: Restart the App

After adding the resource and granting permissions, redeploy:

```bash
databricks apps deploy maint-bot-app \
    --source-code-path /Workspace/Users/<your-email>/.bundle/maint_bot_app/<target>/files \
    --profile <your-profile>
```

### Step 10: Test

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
| `couldn't get a connection after 30.00 sec` | Postgres resource missing from app | Re-add via UI or SDK (Step 7) |
| `start-app` crashes | Frontend can't clone from GitHub in app container | Use `start-server` instead (backend-only, API at `/invocations`) |
| `databricks bundle deploy` resets resources | Known limitation for autoscaling | Re-add resource after each `bundle deploy` (Step 7) |
| Response takes 60+ seconds | Lakebase auth failing, agent falling back to stateless | Check logs at `/logz`, verify all 3 permission layers |
| Response takes ~15 seconds | Normal — LLM inference + tool calls | Expected behavior |

## Key Design Decisions

1. **`AsyncLakebasePool` for custom tables**: Same connection mechanism as `AsyncCheckpointSaver`. Custom tables live in `databricks_postgres.maint_bot` schema (not a separate database) because `AsyncLakebasePool` hardcodes `dbname=databricks_postgres`.

2. **Lazy imports**: Custom table and memory modules are imported inside functions, not at module level. This prevents import-time failures in the app container.

3. **Best-effort DB operations**: All custom table operations (session tracking, event logging, memory retrieval) are wrapped in try/except. If Lakebase is unavailable, the agent works in stateless mode.

4. **`databricks_create_role` only**: Never use raw `CREATE ROLE` SQL for service principals. The `databricks_auth` extension's `databricks_create_role()` function sets up the OAuth token mapping required for JWT authentication.

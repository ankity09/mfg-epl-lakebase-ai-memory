#!/usr/bin/env python3
"""
One-shot setup script for MaintBot Lakebase permissions.

Handles ALL three permission layers + pre-creates schemas and tables.
Run this ONCE after deploying the app. Requires a workspace admin or
the Lakebase project owner.

Usage:
    # Get the SP client ID from your deployed app:
    databricks apps get <app-name> --output json | jq -r '.service_principal_client_id'

    # Run the setup:
    uv run python scripts/grant_lakebase_permissions.py <SP_CLIENT_ID> \
        --project <project-name> --branch <branch-name>
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# Checkpoint tables (LangGraph)
CHECKPOINT_TABLES = [
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
    "checkpoint_blobs",
]

# Chat UI persistence schemas
CHAT_SCHEMAS = {
    "ai_chatbot": [
        ('CREATE TABLE IF NOT EXISTS ai_chatbot."User" ('
         'id TEXT PRIMARY KEY, email TEXT, name TEXT, image TEXT, '
         '"createdAt" TIMESTAMP DEFAULT NOW(), "updatedAt" TIMESTAMP DEFAULT NOW())'),
        ('CREATE TABLE IF NOT EXISTS ai_chatbot."Chat" ('
         'id TEXT PRIMARY KEY, title TEXT, "userId" TEXT, visibility TEXT DEFAULT \'private\', '
         '"createdAt" TIMESTAMP DEFAULT NOW(), "updatedAt" TIMESTAMP DEFAULT NOW(), "lastContext" TEXT)'),
        ('CREATE TABLE IF NOT EXISTS ai_chatbot."Message" ('
         'id TEXT PRIMARY KEY, "chatId" TEXT, role TEXT, parts JSONB, '
         'attachments JSONB, "createdAt" TIMESTAMP DEFAULT NOW())'),
        ('CREATE TABLE IF NOT EXISTS ai_chatbot."Vote" ('
         '"chatId" TEXT, "messageId" TEXT, "isUpvoted" BOOLEAN, '
         'PRIMARY KEY ("chatId", "messageId"))'),
    ],
    "drizzle": [
        ('CREATE TABLE IF NOT EXISTS drizzle."__drizzle_migrations" ('
         'id SERIAL PRIMARY KEY, hash text NOT NULL, created_at bigint)'),
    ],
}

# MaintBot custom tables
MAINT_BOT_TABLES = [
    ('CREATE TABLE IF NOT EXISTS maint_bot.technicians ('
     'id UUID PRIMARY KEY DEFAULT gen_random_uuid(), '
     'external_id VARCHAR(255) UNIQUE NOT NULL, '
     'display_name VARCHAR(255), '
     'specializations JSONB DEFAULT \'[]\'::jsonb, '
     'created_at TIMESTAMP DEFAULT NOW(), '
     'updated_at TIMESTAMP DEFAULT NOW())'),
    ('CREATE TABLE IF NOT EXISTS maint_bot.work_orders ('
     'id UUID PRIMARY KEY DEFAULT gen_random_uuid(), '
     'technician_id UUID REFERENCES maint_bot.technicians(id), '
     'agent_id VARCHAR(100) DEFAULT \'maint-bot\', '
     'equipment_id VARCHAR(100), '
     'priority VARCHAR(20) DEFAULT \'medium\', '
     'status VARCHAR(50) DEFAULT \'active\', '
     'summary TEXT, '
     'max_tokens INTEGER DEFAULT 8000, '
     'created_at TIMESTAMP DEFAULT NOW(), '
     'ended_at TIMESTAMP)'),
    ('CREATE TABLE IF NOT EXISTS maint_bot.work_order_events ('
     'id UUID PRIMARY KEY DEFAULT gen_random_uuid(), '
     'work_order_id UUID REFERENCES maint_bot.work_orders(id), '
     'role VARCHAR(20) NOT NULL, '
     'content TEXT NOT NULL, '
     'token_count INTEGER DEFAULT 0, '
     'is_compacted BOOLEAN DEFAULT FALSE, '
     'created_at TIMESTAMP DEFAULT NOW())'),
    ('CREATE TABLE IF NOT EXISTS maint_bot.maintenance_knowledge ('
     'id UUID PRIMARY KEY DEFAULT gen_random_uuid(), '
     'technician_id UUID REFERENCES maint_bot.technicians(id), '
     'content TEXT NOT NULL, '
     'memory_type VARCHAR(50) NOT NULL, '
     'equipment_tag VARCHAR(100), '
     'importance FLOAT DEFAULT 0.5, '
     'embedding vector(1024), '
     'source_work_order_id UUID, '
     'is_active BOOLEAN DEFAULT TRUE, '
     'access_count INTEGER DEFAULT 0, '
     'created_at TIMESTAMP DEFAULT NOW(), '
     'updated_at TIMESTAMP DEFAULT NOW())'),
]


def main():
    parser = argparse.ArgumentParser(
        description="One-shot Lakebase setup for MaintBot app"
    )
    parser.add_argument("sp_client_id", help="Service principal client ID (UUID)")
    parser.add_argument("--project", default=os.getenv("LAKEBASE_AUTOSCALING_PROJECT"),
                        help="Lakebase project name")
    parser.add_argument("--branch", default=os.getenv("LAKEBASE_AUTOSCALING_BRANCH", "production"),
                        help="Lakebase branch name")
    args = parser.parse_args()

    if not args.project:
        print("Error: --project is required", file=sys.stderr)
        sys.exit(1)

    sp_id = args.sp_client_id
    print(f"Setting up Lakebase for SP: {sp_id}")
    print(f"Project: {args.project}, Branch: {args.branch}")

    # ─── Layer 1: Project-level ACL ───
    print("\n=== Layer 1: Project-level CAN_USE ===")
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
        w = WorkspaceClient()
        w.permissions.update(
            request_object_type="database-projects",
            request_object_id=args.project,
            access_control_list=[
                AccessControlRequest(
                    service_principal_name=sp_id,
                    permission_level=PermissionLevel.CAN_USE,
                )
            ],
        )
        print(f"  Granted CAN_USE on project '{args.project}'")
    except Exception as e:
        print(f"  Warning: {e}")
        print("  You may need to grant this via the UI: Lakebase → project → Settings → Grant permission")

    # ─── Layer 2: Postgres role via databricks_create_role ───
    print("\n=== Layer 2: Postgres role (OAuth mapping) ===")
    from databricks_ai_bridge.lakebase import LakebaseClient, SchemaPrivilege, TablePrivilege
    client = LakebaseClient(project=args.project, branch=args.branch)

    try:
        result = client.create_role(sp_id, "SERVICE_PRINCIPAL")
        if result:
            print(f"  Role created: {result}")
        else:
            print("  Role already exists")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Role already exists")
        else:
            print(f"  Error: {e}")
            print("  Ask the DB owner to run: SELECT databricks_create_role('<SP_ID>', 'SERVICE_PRINCIPAL');")

    # ─── Layer 3: Schema creation + SQL grants ───
    print("\n=== Layer 3: Schemas, tables, and grants ===")

    all_schemas = ["public", "maint_bot", "drizzle", "ai_chatbot"]

    # Grant schemas via LakebaseClient
    try:
        client.grant_schema(grantee=sp_id, schemas=all_schemas, privileges=[SchemaPrivilege.ALL])
        print(f"  Schema grants via LakebaseClient: OK")
    except Exception as e:
        print(f"  Schema grants warning: {e}")

    # Grant checkpoint tables
    for t in CHECKPOINT_TABLES:
        try:
            client.grant_table(grantee=sp_id, tables=[f"public.{t}"], privileges=[TablePrivilege.ALL])
        except Exception:
            pass
    print("  Checkpoint table grants: OK")

    # Pre-create schemas and tables via raw SQL
    print("\n  Pre-creating schemas and tables...")
    try:
        import psycopg
        from psycopg.rows import dict_row
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        parent = f"projects/{args.project}/branches/{args.branch}"
        eps = list(w.postgres.list_endpoints(parent=parent))
        host = eps[0].status.hosts.host
        cred = w.postgres.generate_database_credential(endpoint=eps[0].name)
        user = w.current_user.me().user_name

        conn = psycopg.connect(host=host, port=5432, dbname="databricks_postgres",
            user=user, password=cred.token, sslmode="require", autocommit=True, row_factory=dict_row)

        # Create schemas
        for schema in ["maint_bot", "drizzle", "ai_chatbot"]:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        # pgvector extension
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create chat UI tables
        for schema, ddl_list in CHAT_SCHEMAS.items():
            for ddl in ddl_list:
                conn.execute(ddl)
        print("  Chat UI tables (drizzle, ai_chatbot): created")

        # Create MaintBot tables
        for ddl in MAINT_BOT_TABLES:
            conn.execute(ddl)
        print("  MaintBot tables (maint_bot schema): created")

        # Grant ALL on everything to the SP
        for schema in all_schemas:
            conn.execute(f'GRANT ALL ON SCHEMA {schema} TO "{sp_id}"')
            conn.execute(f'GRANT ALL ON ALL TABLES IN SCHEMA {schema} TO "{sp_id}"')
            conn.execute(f'GRANT ALL ON ALL SEQUENCES IN SCHEMA {schema} TO "{sp_id}"')
            conn.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT ALL ON TABLES TO "{sp_id}"')
            conn.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT ALL ON SEQUENCES TO "{sp_id}"')
        conn.execute(f'GRANT CONNECT ON DATABASE databricks_postgres TO "{sp_id}"')
        print("  SQL grants: applied")

        conn.close()
    except Exception as e:
        print(f"  Warning: {e}")
        print("  Tables will be created on first agent request if the SP has CREATE privilege")

    print("\n=== Setup complete! ===")
    print("Restart the app to pick up the new permissions.")


if __name__ == "__main__":
    main()

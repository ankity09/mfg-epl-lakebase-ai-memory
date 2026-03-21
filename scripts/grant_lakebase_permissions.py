#!/usr/bin/env python3
"""Grant Lakebase Postgres permissions to the MaintBot app's service principal.

Uses LakebaseClient from databricks_ai_bridge to properly create roles
and grant permissions (not raw SQL).

Usage:
    # Get the SP client ID:
    databricks apps get maint-bot-app --output json | jq -r '.service_principal_client_id'

    # Autoscaling:
    uv run python scripts/grant_lakebase_permissions.py <sp-client-id> \
        --project lakebase-mfg-epl --branch production
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# LangGraph checkpoint tables + MaintBot custom tables
PUBLIC_TABLES = [
    # LangGraph checkpoint tables
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
    "checkpoint_blobs",
]

# Chat UI persistence schemas
SHARED_SCHEMAS = {
    "ai_chatbot": ["Chat", "Message", "User", "Vote"],
    "drizzle": ["__drizzle_migrations"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Grant Lakebase permissions to MaintBot app service principal"
    )
    parser.add_argument(
        "sp_client_id",
        help="Service principal client ID (UUID)",
    )
    parser.add_argument(
        "--instance-name",
        default=os.getenv("LAKEBASE_INSTANCE_NAME"),
        help="Lakebase provisioned instance name",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("LAKEBASE_AUTOSCALING_PROJECT", "lakebase-mfg-epl"),
        help="Lakebase autoscaling project",
    )
    parser.add_argument(
        "--branch",
        default=os.getenv("LAKEBASE_AUTOSCALING_BRANCH", "production"),
        help="Lakebase autoscaling branch",
    )
    args = parser.parse_args()

    has_provisioned = bool(args.instance_name)
    has_autoscaling = bool(args.project and args.branch)

    if not has_provisioned and not has_autoscaling:
        print("Error: provide --instance-name or --project + --branch", file=sys.stderr)
        sys.exit(1)

    from databricks_ai_bridge.lakebase import (
        LakebaseClient,
        SchemaPrivilege,
        TablePrivilege,
    )

    client = LakebaseClient(
        instance_name=args.instance_name or None,
        project=args.project or None,
        branch=args.branch or None,
    )
    sp_id = args.sp_client_id

    if has_provisioned:
        print(f"Using provisioned instance: {args.instance_name}")
    else:
        print(f"Using autoscaling: {args.project}/{args.branch}")

    # 1. Create role for SP (Lakebase-specific identity mapping)
    print(f"\nCreating role for SP {sp_id}...")
    try:
        client.create_role(sp_id, "SERVICE_PRINCIPAL")
        print("  Role created.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Role already exists, skipping.")
        else:
            print(f"  Warning: {e}")

    # 2. Grant schema + table privileges
    schema_privileges = [SchemaPrivilege.USAGE, SchemaPrivilege.CREATE]
    table_privileges = [
        TablePrivilege.SELECT,
        TablePrivilege.INSERT,
        TablePrivilege.UPDATE,
        TablePrivilege.DELETE,
    ]

    # Public schema (checkpoint + custom tables)
    schema_tables = {"public": PUBLIC_TABLES, **SHARED_SCHEMAS}

    for schema, tables in schema_tables.items():
        print(f"\nGranting schema privileges on '{schema}'...")
        try:
            client.grant_schema(
                grantee=sp_id, schemas=[schema], privileges=schema_privileges
            )
            print("  Schema grants OK.")
        except Exception as e:
            print(f"  Warning: {e}")

        qualified_tables = [f"{schema}.{t}" for t in tables]
        print(f"  Granting table privileges on {qualified_tables}...")
        try:
            client.grant_table(
                grantee=sp_id, tables=qualified_tables, privileges=table_privileges
            )
            print("  Table grants OK.")
        except Exception as e:
            print(f"  Warning (tables may not exist yet): {e}")

    print(
        "\nDone. If some grants failed because tables don't exist yet, "
        "re-run after the first agent request creates them."
    )


if __name__ == "__main__":
    main()

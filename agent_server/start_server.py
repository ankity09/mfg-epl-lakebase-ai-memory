import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

logger = logging.getLogger(__name__)

# Load env vars from .env before importing the agent for proper auth
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


def _ensure_mlflow_experiment():
    """Auto-create an MLflow experiment if MLFLOW_EXPERIMENT_ID is not set."""
    if os.getenv("MLFLOW_EXPERIMENT_ID"):
        return

    mlflow.set_tracking_uri("databricks")
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        user = w.current_user.me().user_name
        exp_name = f"/Users/{user}/maint-bot-app"

        exp = mlflow.get_experiment_by_name(exp_name)
        if exp:
            exp_id = exp.experiment_id
        else:
            exp_id = mlflow.create_experiment(exp_name)
            logger.info(f"Created MLflow experiment: {exp_name} (ID: {exp_id})")

        os.environ["MLFLOW_EXPERIMENT_ID"] = str(exp_id)
    except Exception as e:
        logger.warning(f"Could not auto-create MLflow experiment: {e}")


_ensure_mlflow_experiment()

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402, F401

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841
try:
    setup_mlflow_git_based_version_tracking()
except Exception as e:
    logger.warning(f"Git-based version tracking setup failed (expected in app containers): {e}")


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")

"""
FastAPI app for the integration test environment (used inside Docker).

Serves the FileCreatorEnvironment with MCP tools on :8000.
"""

import os
import sys

# Ensure the app directory is on sys.path for file_creator_env import
sys.path.insert(0, "/app")

from file_creator_env import FileCreatorEnvironment

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

WORKSPACE = os.environ.get("WORKSPACE_DIR", "/workspace")


def _factory():
    return FileCreatorEnvironment(workspace_dir=WORKSPACE)


app = create_app(
    _factory,
    CallToolAction,
    CallToolObservation,
    env_name="file_creator_env",
    max_concurrent_envs=1,
)

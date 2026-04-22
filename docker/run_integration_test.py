#!/usr/bin/env python3
"""
In-container integration test.

This runs INSIDE the Docker container. It:
1. Starts the OpenEnv server in a background thread
2. Launches pi via PiHarnessAdapter
3. Has pi complete the file-creation task
4. Prints results and exits 0/1

Usage:
  docker run --rm \
      -e OPENENV_TEST_API_KEY=... \
      -e OPENENV_TEST_API_URL=... \
      -e OPENENV_TEST_MODEL=... \
      openenv-integration:latest \
      python /app/run_integration_test.py
"""

import json
import logging
import os
import sys
import threading
import time
import socket

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("integration-test")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENENV_TEST_API_KEY", "")
API_URL = os.environ.get("OPENENV_TEST_API_URL", "")
MODEL = os.environ.get("OPENENV_TEST_MODEL", "")
PROVIDER_NAME = "openenv-test"
WORKSPACE = "/workspace"
PORT = 8000


def check_config():
    missing = []
    if not API_KEY:
        missing.append("OPENENV_TEST_API_KEY")
    if not API_URL:
        missing.append("OPENENV_TEST_API_URL")
    if not MODEL:
        missing.append("OPENENV_TEST_MODEL")
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def write_models_json():
    """Write pi models.json so it knows about the custom LLM endpoint."""
    agent_dir = os.path.expanduser("~/.pi/agent")
    os.makedirs(agent_dir, exist_ok=True)

    config = {
        "providers": {
            PROVIDER_NAME: {
                "baseUrl": API_URL,
                "api": "openai-completions",
                "apiKey": API_KEY,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [
                    {
                        "id": MODEL,
                        "name": MODEL,
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": 32768,
                        "maxTokens": 8192,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }

    path = os.path.join(agent_dir, "models.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote models.json → %s", path)


def start_server():
    """Start the OpenEnv FastAPI server in a background thread."""
    # Import here so sys.path is ready
    sys.path.insert(0, "/app")
    from file_creator_env import FileCreatorEnvironment
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    def _factory():
        return FileCreatorEnvironment(workspace_dir=WORKSPACE)

    app = create_app(
        _factory,
        CallToolAction,
        CallToolObservation,
        env_name="file_creator_env",
        max_concurrent_envs=1,
    )

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for ready
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                logger.info("Server ready on :%d", PORT)
                return server
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("Server didn't start")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def run_test():
    from openenv.core.harnesses import (
        HarnessAction,
        HarnessConfig,
        HarnessEnvironment,
        HarnessEventType,
        HarnessTransport,
    )
    from openenv.core.harnesses.adapters.pi import PiHarnessAdapter

    mcp_url = f"http://127.0.0.1:{PORT}/mcp"

    config = HarnessConfig(
        name="pi",
        command=["pi"],
        transport=HarnessTransport.STDIO,
        working_directory=WORKSPACE,
        model=f"{PROVIDER_NAME}/{MODEL}",
        session_timeout_s=180.0,
        startup_timeout_s=30.0,
        env_vars={
            "OPENAI_API_KEY": API_KEY,
        },
    )

    adapter = PiHarnessAdapter(
        config=config,
        mcp_server_url=mcp_url,
        provider=PROVIDER_NAME,
        thinking_level="off",
    )

    env = HarnessEnvironment(adapter=adapter)

    try:
        logger.info("Resetting environment...")
        obs = env.reset()
        logger.info("Reset OK. Episode: %s", env.state.episode_id)

        logger.info("Sending task instruction to pi...")
        obs = env.step(
            HarnessAction(
                message=(
                    "You have access to MCP tools from the 'openenv' server. "
                    "First, call the 'get_task' tool to see what you need to do. "
                    "Then complete the task using your built-in tools. "
                    "Finally, call the 'check_file' tool to verify your work. "
                    "Do NOT ask for clarification — just do it."
                )
            )
        )

        logger.info("Step completed. done=%s", obs.done)
        logger.info("Response: %s", obs.metadata.get("response", "")[:500])

        # Log trajectory
        events = env.trajectory
        tool_calls = [e for e in events if e.type == HarnessEventType.TOOL_CALL]
        tool_results = [e for e in events if e.type == HarnessEventType.TOOL_RESULT]
        logger.info(
            "Trajectory: %d events, %d tool calls, %d tool results",
            len(events), len(tool_calls), len(tool_results),
        )

        for tc in tool_calls:
            logger.info(
                "  Tool call: %s (phase=%s)",
                tc.data.get("tool_name"), tc.data.get("phase"),
            )

        # Verify result
        hello_path = os.path.join(WORKSPACE, "hello.txt")
        if os.path.exists(hello_path):
            with open(hello_path) as f:
                content = f.read()
            logger.info("hello.txt content: %r", content)
            if "Hello, OpenEnv!" in content:
                logger.info("✅ SUCCESS: File created with correct content!")
                return True
            else:
                logger.error("❌ FAIL: File exists but content is wrong")
                return False
        else:
            logger.error(
                "❌ FAIL: hello.txt not found. Files: %s",
                os.listdir(WORKSPACE),
            )
            return False

    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    check_config()

    logger.info("=== Pi Harness Integration Test (in-container) ===")
    logger.info("Model:   %s/%s", PROVIDER_NAME, MODEL)
    logger.info("API URL: %s", API_URL)
    logger.info("Workspace: %s", WORKSPACE)

    write_models_json()
    start_server()

    success = run_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

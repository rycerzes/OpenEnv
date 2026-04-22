# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration test: PiHarnessAdapter + FileCreatorEnvironment

Starts a real OpenEnv HTTP server with MCP tools, launches a real pi process
via PiHarnessAdapter, and has pi complete a simple file-creation task.

Requirements:
  - pi installed and on PATH
  - pi-mcp-adapter extension installed (pi install npm:pi-mcp-adapter)
  - Environment variables for LLM access:
      OPENENV_TEST_API_KEY   - API key for the OpenAI-compatible endpoint
      OPENENV_TEST_API_URL   - Base URL (e.g. https://my-vllm.example.com/v1)
      OPENENV_TEST_MODEL     - Model name (e.g. Qwen/Qwen3-32B)

Usage:
  # Set the required env vars, then:
  PYTHONPATH=src:envs uv run pytest tests/integration/test_pi_harness_live.py -v -s

  # Or run directly:
  PYTHONPATH=src:envs uv run python tests/integration/test_pi_harness_live.py
"""

import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import time

import pytest
import uvicorn

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.harnesses import (
    HarnessAction,
    HarnessConfig,
    HarnessEnvironment,
    HarnessEventType,
    HarnessTransport,
)
from openenv.core.harnesses.adapters.pi import PiHarnessAdapter

from .file_creator_env import FileCreatorEnvironment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENENV_TEST_API_KEY", "")
API_URL = os.environ.get("OPENENV_TEST_API_URL", "")
MODEL = os.environ.get("OPENENV_TEST_MODEL", "")
# Provider name used in models.json — "openenv-test" to avoid colliding with
# pi's built-in "openai" provider registration.
PROVIDER_NAME = "openenv-test"


def _missing_config() -> bool:
    return not all([API_KEY, API_URL, MODEL])


SKIP_REASON = (
    "Set OPENENV_TEST_API_KEY, OPENENV_TEST_API_URL, and OPENENV_TEST_MODEL "
    "to run live pi integration tests"
)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Server helper: run the OpenEnv HTTP server in a background thread
# ---------------------------------------------------------------------------
class _EnvServer:
    """Runs the OpenEnv FastAPI app in a background thread."""

    def __init__(self, workspace: str, port: int) -> None:
        self.workspace = workspace
        self.port = port
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    def start(self) -> None:
        ws = self.workspace

        def _factory():
            return FileCreatorEnvironment(workspace_dir=ws)

        app = create_app(
            _factory,
            CallToolAction,
            CallToolObservation,
            env_name="file_creator_env",
            max_concurrent_envs=1,
        )

        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with socket.create_connection(
                    ("127.0.0.1", self.port), timeout=1
                ):
                    return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError(f"Server didn't start on port {self.port}")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Pi models.json helper — registers custom OpenAI-compat endpoint
# ---------------------------------------------------------------------------
def _write_models_json(pi_home: str) -> None:
    """Write a models.json for pi so it knows about our custom endpoint."""
    agent_dir = os.path.join(pi_home, ".pi", "agent")
    os.makedirs(agent_dir, exist_ok=True)

    models_config = {
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
        json.dump(models_config, f, indent=2)
    logger.info("Wrote models.json to %s", path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Create a temporary workspace directory."""
    ws = tmp_path_factory.mktemp("pi_workspace")
    yield str(ws)


@pytest.fixture(scope="module")
def server_port():
    return _find_free_port()


@pytest.fixture(scope="module")
def env_server(workspace, server_port):
    """Start the OpenEnv HTTP server."""
    srv = _EnvServer(workspace, server_port)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def pi_home(tmp_path_factory):
    """Create a temporary HOME for pi so models.json doesn't pollute real home."""
    home = tmp_path_factory.mktemp("pi_home")
    _write_models_json(str(home))
    yield str(home)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_missing_config(), reason=SKIP_REASON)
class TestPiHarnessLive:
    """Live integration tests — requires LLM endpoint."""

    def test_pi_completes_file_creation_task(
        self, workspace, server_port, env_server, pi_home
    ):
        """End-to-end: pi creates a file and verifies it via MCP tool."""

        mcp_url = f"http://127.0.0.1:{server_port}/mcp"

        config = HarnessConfig(
            name="pi",
            command=["pi"],
            transport=HarnessTransport.STDIO,
            working_directory=workspace,
            model=f"{PROVIDER_NAME}/{MODEL}",
            session_timeout_s=120.0,
            startup_timeout_s=30.0,
            env_vars={
                # Override HOME so pi reads our models.json
                "HOME": pi_home,
                # Propagate the API key
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
            # Reset starts pi and injects MCP tools via .mcp.json
            obs = env.reset()
            assert obs.done is False
            logger.info("Environment reset OK, episode=%s", env.state.episode_id)

            # Send the task instruction — pi does its ReAct loop
            instruction = (
                "You have access to MCP tools from the 'openenv' server. "
                "First, call the 'get_task' tool to see what you need to do. "
                "Then complete the task using your built-in tools. "
                "Finally, call the 'check_file' tool to verify your work. "
                "Do NOT ask for clarification — just do it."
            )

            obs = env.step(HarnessAction(message=instruction))

            # Log what happened
            logger.info("Step done=%s, response length=%d", obs.done, len(obs.metadata.get("response", "")))
            logger.info("Response preview: %s", obs.metadata.get("response", "")[:500])

            events = env.trajectory
            tool_calls = [
                e for e in events
                if e.type == HarnessEventType.TOOL_CALL
            ]
            tool_results = [
                e for e in events
                if e.type == HarnessEventType.TOOL_RESULT
            ]

            logger.info(
                "Trajectory: %d events, %d tool calls, %d tool results",
                len(events),
                len(tool_calls),
                len(tool_results),
            )

            for tc in tool_calls:
                logger.info("  Tool call: %s (phase=%s)", tc.data.get("tool_name"), tc.data.get("phase"))
            for tr in tool_results:
                logger.info("  Tool result: %s error=%s", tr.data.get("tool_name"), tr.data.get("is_error"))

            # Verify the file was actually created
            hello_path = os.path.join(workspace, "hello.txt")
            assert os.path.exists(hello_path), (
                f"hello.txt not created in {workspace}. "
                f"Files present: {os.listdir(workspace)}"
            )

            with open(hello_path) as f:
                content = f.read()
            assert "Hello, OpenEnv!" in content, (
                f"File content wrong: {content!r}"
            )

            # Verify pi called at least some tools
            assert len(tool_calls) > 0, "Pi should have made tool calls"

            logger.info("SUCCESS: Pi created the file and it has correct content")

        finally:
            env.close()

    def test_harness_environment_multi_turn(
        self, workspace, server_port, env_server, pi_home
    ):
        """Test multi-turn conversation: ask pi to do something, then ask about it."""

        mcp_url = f"http://127.0.0.1:{server_port}/mcp"

        # Use a sub-directory so it doesn't conflict with test above
        turn_workspace = os.path.join(workspace, "multi_turn")
        os.makedirs(turn_workspace, exist_ok=True)

        config = HarnessConfig(
            name="pi",
            command=["pi"],
            transport=HarnessTransport.STDIO,
            working_directory=turn_workspace,
            model=f"{PROVIDER_NAME}/{MODEL}",
            session_timeout_s=120.0,
            startup_timeout_s=30.0,
            env_vars={
                "HOME": pi_home,
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
            env.reset()

            # Turn 1: ask pi to create a file
            obs1 = env.step(
                HarnessAction(
                    message=(
                        "Create a file called 'greeting.txt' in the current directory "
                        "with the content 'Hi from turn 1'. Use your write tool."
                    )
                )
            )
            assert env.state.step_count == 1
            logger.info("Turn 1 done, response: %s", obs1.metadata.get("response", "")[:200])

            # Turn 2: ask pi to read it back
            obs2 = env.step(
                HarnessAction(
                    message="Read the file greeting.txt and tell me what it contains."
                )
            )
            assert env.state.step_count == 2
            response2 = obs2.metadata.get("response", "")
            logger.info("Turn 2 done, response: %s", response2[:200])

            # The response should mention the content
            assert "Hi from turn 1" in response2 or os.path.exists(
                os.path.join(turn_workspace, "greeting.txt")
            ), "Multi-turn conversation didn't work as expected"

            # Trajectory should have events from both turns
            assert len(env.trajectory) > 0

            logger.info("SUCCESS: Multi-turn conversation worked")

        finally:
            env.close()


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
def main():
    """Run the test directly (outside pytest) for quick iteration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if _missing_config():
        print(f"SKIP: {SKIP_REASON}")
        return

    workspace = tempfile.mkdtemp(prefix="pi_workspace_")
    pi_home = tempfile.mkdtemp(prefix="pi_home_")
    port = _find_free_port()

    print(f"Workspace: {workspace}")
    print(f"Pi home:   {pi_home}")
    print(f"Server:    http://127.0.0.1:{port}")
    print(f"Model:     {PROVIDER_NAME}/{MODEL}")
    print(f"API URL:   {API_URL}")
    print()

    _write_models_json(pi_home)

    srv = _EnvServer(workspace, port)
    srv.start()
    print("Server started")

    mcp_url = f"http://127.0.0.1:{port}/mcp"

    config = HarnessConfig(
        name="pi",
        command=["pi"],
        transport=HarnessTransport.STDIO,
        working_directory=workspace,
        model=f"{PROVIDER_NAME}/{MODEL}",
        session_timeout_s=180.0,
        startup_timeout_s=30.0,
        env_vars={
            "HOME": pi_home,
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
        print("\n--- RESET ---")
        obs = env.reset()
        print(f"Reset OK. Episode: {env.state.episode_id}")

        print("\n--- STEP (task instruction) ---")
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

        print(f"\nDone: {obs.done}")
        print(f"Response: {obs.metadata.get('response', '')[:500]}")
        print(f"\nTrajectory: {len(env.trajectory)} events")

        for e in env.trajectory:
            if e.type in (HarnessEventType.TOOL_CALL, HarnessEventType.TOOL_RESULT):
                print(f"  {e.type.value}: {e.data.get('tool_name', '?')} "
                      f"phase={e.data.get('phase', '')} "
                      f"error={e.data.get('is_error', '')}")

        # Check result
        hello_path = os.path.join(workspace, "hello.txt")
        if os.path.exists(hello_path):
            with open(hello_path) as f:
                content = f.read()
            print(f"\nhello.txt content: {content!r}")
            if "Hello, OpenEnv!" in content:
                print("\n✅ SUCCESS: File created with correct content!")
            else:
                print("\n❌ FAIL: File exists but content is wrong")
        else:
            print(f"\n❌ FAIL: hello.txt not found. Files: {os.listdir(workspace)}")

    finally:
        env.close()
        srv.stop()
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(pi_home, ignore_errors=True)


if __name__ == "__main__":
    main()

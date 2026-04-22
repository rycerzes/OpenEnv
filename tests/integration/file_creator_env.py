# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FileCreatorEnvironment — minimal custom environment for integration testing.

Exposes two MCP tools:
  - get_task(): returns the task instruction
  - check_file(path, expected_content): verifies a file was created correctly

The agent's job: call get_task, create the file with pi's built-in write tool,
then call check_file to verify. This exercises the full harness → pi → MCP loop.
"""

import os
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State


class FileCreatorEnvironment(MCPEnvironment):
    """Minimal environment: create a file and verify it."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, workspace_dir: str) -> None:
        self._workspace = workspace_dir
        mcp = FastMCP("file_creator_env")

        ws = self._workspace  # capture for closures

        @mcp.tool
        def get_task() -> dict:
            """Get the current task description.

            Returns the task that the agent must complete, including
            the target file path and expected content.
            """
            return {
                "task": "Create a file and verify it",
                "target_file": "hello.txt",
                "expected_content": "Hello, OpenEnv!",
                "instructions": (
                    "1. Create a file called 'hello.txt' in the current "
                    "working directory with the exact content: Hello, OpenEnv!\n"
                    "2. Then call check_file to verify your work."
                ),
            }

        @mcp.tool
        def check_file(path: str, expected_content: str) -> dict:
            """Check if a file exists and has the expected content.

            Args:
                path: Path to the file to check (relative to workspace).
                expected_content: The content the file should contain.

            Returns:
                Dictionary with success status and details.
            """
            full_path = os.path.join(ws, path)
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"File {path} does not exist",
                }
            with open(full_path) as f:
                actual = f.read()
            if actual.strip() == expected_content.strip():
                return {
                    "success": True,
                    "message": "File content matches!",
                    "actual_content": actual.strip(),
                }
            return {
                "success": False,
                "error": "Content mismatch",
                "expected": expected_content,
                "actual": actual,
            }

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return Observation(
            done=False,
            reward=0.0,
            metadata={"status": "ready", "workspace": self._workspace},
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}"},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state

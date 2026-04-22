# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pi coding agent harness adapter

Pi is an open-source coding agent by @mariozechner that supports RPC mode
(stdin/stdout JSONL protocol), extensions, and MCP tool integration via
the pi-mcp-adapter extension.

This adapter manages the pi process lifecycle, injects OpenEnv MCP tools
via a .mcp.json configuration file, and communicates via the RPC protocol.

Requires:
- pi (``@mariozechner/pi-coding-agent``) installed and on PATH
- pi-mcp-adapter extension installed (``pi install npm:pi-mcp-adapter``)
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from openenv.core.harnesses.adapter import HarnessAdapter
from openenv.core.harnesses.tools import resolve_tool_conflicts
from openenv.core.harnesses.types import (
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
)

logger = logging.getLogger(__name__)

# Pi built-in tools that may conflict with environment tools.
PI_BUILTIN_TOOLS = frozenset(
    {
        "read",
        "write",
        "edit",
        "multi_edit",
        "bash",
        "glob",
        "grep",
        "ls",
    }
)


class PiHarnessAdapter(HarnessAdapter):
    """Adapter for the pi coding agent via RPC mode.

    Manages a pi process, injects environment MCP tools via .mcp.json
    (consumed by pi-mcp-adapter extension), and communicates via the
    stdin/stdout JSONL RPC protocol.

    The adapter writes a ``.mcp.json`` file in the working directory before
    starting pi. Pi's pi-mcp-adapter extension reads this file and connects
    to the OpenEnv MCP server, making environment tools available as
    first-class pi tools.

    Args:
        config: HarnessConfig with pi-specific settings.
        mcp_server_url: URL of the OpenEnv MCP server endpoint
            (e.g. ``http://localhost:8000/mcp``).
        provider: LLM provider name for pi (e.g. ``anthropic``, ``openai``).
        thinking_level: Thinking/reasoning level (``off``, ``low``,
            ``medium``, ``high``). Default ``medium``.
    """

    def __init__(
        self,
        config: HarnessConfig,
        mcp_server_url: str,
        provider: Optional[str] = None,
        thinking_level: str = "medium",
    ) -> None:
        super().__init__(config)
        self.mcp_server_url = mcp_server_url
        self.provider = provider
        self.thinking_level = thinking_level
        self._process: Optional[asyncio.subprocess.Process] = None
        self._injected_tools: List[Dict[str, Any]] = []

    async def start(self, working_directory: str) -> None:
        """Start the pi process in RPC mode.

        Launches ``pi --mode rpc --no-session`` with the configured
        provider and model. Waits for the process to be ready by
        sending a ``get_state`` command.

        Args:
            working_directory: Path where pi should operate.
        """
        env: Optional[Dict[str, str]] = None
        if self.config.env_vars or self.config.api_key_env_var:
            env = dict(os.environ)
            env.update(self.config.env_vars)
            if self.config.api_key_env_var:
                key = os.environ.get(self.config.api_key_env_var, "")
                if key:
                    env[self.config.api_key_env_var] = key

        cmd = list(self.config.command)
        if "--mode" not in cmd:
            cmd.extend(["--mode", "rpc"])
        if "--no-session" not in cmd:
            cmd.append("--no-session")
        if self.provider and "--provider" not in cmd:
            cmd.extend(["--provider", self.provider])
        if self.config.model and "--model" not in cmd:
            cmd.extend(["--model", self.config.model])

        logger.info("Starting pi: %s (cwd=%s)", " ".join(cmd), working_directory)

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory,
            env=env,
        )

        # Wait for pi to be ready
        await self._wait_for_ready()

        # Set thinking level if specified
        if self.thinking_level != "medium":
            await self._send_command(
                {"type": "set_thinking_level", "level": self.thinking_level}
            )

    async def stop(self) -> None:
        """Stop the pi process.

        Sends an abort command first, then terminates. Falls back to
        kill if terminate doesn't work within 5 seconds.
        """
        if self._process is None:
            return

        try:
            # Try graceful abort first
            if self._process.stdin and self._process.returncode is None:
                try:
                    abort_cmd = json.dumps({"type": "abort"}) + "\n"
                    self._process.stdin.write(abort_cmd.encode())
                    await self._process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            self._process.kill()
        finally:
            self._process = None

    async def inject_tools(self, tools: List) -> None:
        """Write .mcp.json for pi-mcp-adapter to discover OpenEnv tools.

        Creates a ``.mcp.json`` file in the working directory that
        configures pi-mcp-adapter to connect to the OpenEnv MCP server.
        Tools are resolved for conflicts with pi's built-in tools
        (``read``, ``write``, ``edit``, ``bash``, etc.).

        Must be called BEFORE ``start()``.

        Args:
            tools: List of MCP tool definitions from the environment.
        """
        resolved = resolve_tool_conflicts(tools, list(PI_BUILTIN_TOOLS))
        self._injected_tools = [{"name": getattr(t, "name", str(t))} for t in resolved]

        if not tools:
            return

        config_path = self.config.mcp_config_path or str(
            Path(self.config.working_directory) / ".mcp.json"
        )

        mcp_config: Dict[str, Any] = {
            "mcpServers": {
                "openenv": {
                    "url": self.mcp_server_url,
                }
            }
        }

        path = Path(config_path)

        # Merge with existing config if present
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        if "mcpServers" not in existing:
            existing["mcpServers"] = {}
        existing["mcpServers"]["openenv"] = mcp_config["mcpServers"]["openenv"]

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2))
        logger.info("Wrote MCP config to %s", config_path)

    async def send_message(self, message: str) -> HarnessResponse:
        """Send a prompt to pi and collect the full response.

        Sends a ``prompt`` command via stdin and reads JSONL events from
        stdout until an ``agent_end`` event is received.

        Args:
            message: The user message / instruction for this turn.

        Returns:
            HarnessResponse with the aggregated text response and events.

        Raises:
            RuntimeError: If the pi process is not running.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Pi process is not running")

        events: List[HarnessEvent] = []
        ts = time.time()

        # Send prompt command
        prompt_cmd = json.dumps({"type": "prompt", "message": message}) + "\n"
        self._process.stdin.write(prompt_cmd.encode())
        await self._process.stdin.drain()

        events.append(
            HarnessEvent(
                type=HarnessEventType.LLM_REQUEST,
                timestamp=ts,
                data={"message": message},
            )
        )

        # Read events until agent_end
        response_text = ""
        done = False

        try:
            response_text, done = await asyncio.wait_for(
                self._read_until_agent_end(events),
                timeout=self.config.session_timeout_s,
            )
        except asyncio.TimeoutError:
            events.append(
                HarnessEvent(
                    type=HarnessEventType.ERROR,
                    timestamp=time.time(),
                    data={"message": "Session timeout", "recoverable": False},
                )
            )
            return HarnessResponse(
                response="Error: session timeout",
                events=events,
                done=True,
            )

        events.append(
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE,
                timestamp=time.time(),
                data={"response": response_text},
            )
        )

        return HarnessResponse(
            response=response_text,
            events=events,
            done=done,
        )

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        """Send a prompt and stream events as they arrive.

        Yields HarnessEvent instances mapped from pi's RPC event stream.
        The final event has type TURN_COMPLETE.

        Args:
            message: The user message / instruction for this turn.

        Yields:
            HarnessEvent instances as pi processes the turn.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Pi process is not running")

        # Send prompt command
        prompt_cmd = json.dumps({"type": "prompt", "message": message}) + "\n"
        self._process.stdin.write(prompt_cmd.encode())
        await self._process.stdin.drain()

        yield HarnessEvent(
            type=HarnessEventType.LLM_REQUEST,
            timestamp=time.time(),
            data={"message": message},
        )

        if self._process.stdout is None:
            return

        # Stream events until agent_end
        while True:
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=self.config.session_timeout_s,
                )
            except asyncio.TimeoutError:
                yield HarnessEvent(
                    type=HarnessEventType.ERROR,
                    timestamp=time.time(),
                    data={"message": "Session timeout", "recoverable": False},
                )
                return

            if not line:
                return

            decoded = line.decode().strip()
            if not decoded:
                continue

            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            # Skip command responses (have "command" field)
            if "command" in data:
                continue

            harness_event = self._map_pi_event(event_type, data)
            if harness_event is not None:
                yield harness_event

            if event_type == "agent_end":
                yield HarnessEvent(
                    type=HarnessEventType.TURN_COMPLETE,
                    timestamp=time.time(),
                    data={"response": self._extract_response_from_agent_end(data)},
                )
                return

    async def is_alive(self) -> bool:
        """Check if the pi process is still running."""
        if self._process is None:
            return False
        return self._process.returncode is None

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _wait_for_ready(self) -> None:
        """Wait for pi to be ready by sending a get_state command."""
        try:
            response = await asyncio.wait_for(
                self._send_command({"type": "get_state"}),
                timeout=self.config.startup_timeout_s,
            )
            logger.info("Pi ready: %s", response.get("data", {}).get("model"))
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Pi failed to start within {self.config.startup_timeout_s}s"
            )

    async def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command and wait for the corresponding response.

        Reads lines from stdout, skipping events (no ``command`` field),
        until a response matching the command type is found.

        Args:
            command: The command dict to send.

        Returns:
            The parsed response dict.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Pi process is not running")
        if self._process.stdout is None:
            raise RuntimeError("Pi stdout is not available")

        payload = json.dumps(command) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

        cmd_type = command["type"]

        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise RuntimeError("Pi process closed stdout unexpectedly")

            decoded = line.decode().strip()
            if not decoded:
                continue

            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                continue

            # Response to our command
            if data.get("type") == "response" and data.get("command") == cmd_type:
                return data

            # Skip events and responses to other commands

    async def _read_until_agent_end(self, events: List[HarnessEvent]) -> tuple:
        """Read pi events until agent_end, accumulating HarnessEvents.

        Args:
            events: List to append mapped HarnessEvents to.

        Returns:
            Tuple of (response_text, done).
        """
        if self._process is None or self._process.stdout is None:
            return ("", False)

        response_text = ""
        text_chunks: List[str] = []

        while True:
            line = await self._process.stdout.readline()
            if not line:
                break

            decoded = line.decode().strip()
            if not decoded:
                continue

            try:
                data = json.loads(decoded)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            # Skip command responses
            if "command" in data:
                continue

            # Accumulate text deltas
            if event_type == "message_update":
                ame = data.get("assistantMessageEvent", {})
                if ame.get("type") == "text_delta":
                    delta = ame.get("delta", "")
                    text_chunks.append(delta)

            # Map to harness events
            harness_event = self._map_pi_event(event_type, data)
            if harness_event is not None:
                events.append(harness_event)

            # agent_end signals completion
            if event_type == "agent_end":
                if not text_chunks:
                    response_text = self._extract_response_from_agent_end(data)
                else:
                    response_text = "".join(text_chunks)
                return (response_text, True)

        # Process exited without agent_end
        response_text = "".join(text_chunks)
        return (response_text, False)

    def _map_pi_event(
        self, event_type: str, data: Dict[str, Any]
    ) -> Optional[HarnessEvent]:
        """Map a pi RPC event to a HarnessEvent.

        Args:
            event_type: The pi event type string.
            data: The full event data dict.

        Returns:
            A HarnessEvent, or None if the event should be skipped.
        """
        ts = time.time()

        if event_type == "message_update":
            ame = data.get("assistantMessageEvent", {})
            ame_type = ame.get("type", "")

            if ame_type == "text_delta":
                return HarnessEvent(
                    type=HarnessEventType.LLM_CHUNK,
                    timestamp=ts,
                    data={
                        "delta": ame.get("delta", ""),
                        "content_index": ame.get("contentIndex"),
                    },
                )
            elif ame_type == "toolcall_start":
                return HarnessEvent(
                    type=HarnessEventType.TOOL_CALL,
                    timestamp=ts,
                    data={
                        "tool_call_id": ame.get("toolCall", {}).get("id"),
                        "tool_name": ame.get("toolCall", {}).get("name"),
                        "phase": "start",
                    },
                )
            elif ame_type == "toolcall_end":
                tool_call = ame.get("toolCall", {})
                return HarnessEvent(
                    type=HarnessEventType.TOOL_CALL,
                    timestamp=ts,
                    data={
                        "tool_call_id": tool_call.get("id"),
                        "tool_name": tool_call.get("name"),
                        "arguments": tool_call.get("arguments"),
                        "phase": "end",
                    },
                )
            elif ame_type in ("error",):
                return HarnessEvent(
                    type=HarnessEventType.ERROR,
                    timestamp=ts,
                    data={
                        "reason": ame.get("reason", "unknown"),
                        "message": str(ame),
                    },
                )

        elif event_type == "tool_execution_start":
            return HarnessEvent(
                type=HarnessEventType.TOOL_CALL,
                timestamp=ts,
                data={
                    "tool_call_id": data.get("toolCallId"),
                    "tool_name": data.get("toolName"),
                    "arguments": data.get("args"),
                    "phase": "execution_start",
                },
            )

        elif event_type == "tool_execution_end":
            return HarnessEvent(
                type=HarnessEventType.TOOL_RESULT,
                timestamp=ts,
                data={
                    "tool_call_id": data.get("toolCallId"),
                    "tool_name": data.get("toolName"),
                    "result": data.get("result"),
                    "is_error": data.get("isError", False),
                },
            )

        elif event_type == "message_start":
            return HarnessEvent(
                type=HarnessEventType.LLM_RESPONSE,
                timestamp=ts,
                data={"phase": "start", "message": data.get("message")},
            )

        elif event_type == "message_end":
            return HarnessEvent(
                type=HarnessEventType.LLM_RESPONSE,
                timestamp=ts,
                data={"phase": "end", "message": data.get("message")},
            )

        elif event_type == "agent_end":
            # Handled by caller
            return None

        elif event_type == "extension_error":
            return HarnessEvent(
                type=HarnessEventType.ERROR,
                timestamp=ts,
                data={
                    "extension": data.get("extensionPath"),
                    "event": data.get("event"),
                    "message": data.get("error"),
                },
            )

        return None

    def _extract_response_from_agent_end(self, data: Dict[str, Any]) -> str:
        """Extract the final text response from an agent_end event.

        agent_end includes a ``messages`` array with all messages from
        the run. We extract text content from the last assistant message.

        Args:
            data: The agent_end event data.

        Returns:
            The concatenated text content from the last assistant message.
        """
        messages = data.get("messages", [])
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role == "assistant":
                content = msg.get("content", [])
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        texts.append(block)
                if texts:
                    return "\n".join(texts)
        return ""

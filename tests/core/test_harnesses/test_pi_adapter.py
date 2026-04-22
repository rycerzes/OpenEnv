# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for PiHarnessAdapter"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openenv.core.harnesses import HarnessConfig, HarnessEventType, HarnessTransport
from openenv.core.harnesses.adapters.pi import PI_BUILTIN_TOOLS, PiHarnessAdapter  # noqa: F401


@pytest.fixture
def pi_config():
    return HarnessConfig(
        name="pi",
        command=["pi"],
        transport=HarnessTransport.STDIO,
        model="claude-sonnet-4-20250514",
        api_key_env_var="ANTHROPIC_API_KEY",
        session_timeout_s=10.0,
        startup_timeout_s=5.0,
    )


@pytest.fixture
def adapter(pi_config):
    return PiHarnessAdapter(
        config=pi_config,
        mcp_server_url="http://localhost:8000/mcp",
        provider="anthropic",
    )


class TestPiHarnessAdapterImport:
    """Test imports."""

    def test_import_from_adapters(self):
        from openenv.core.harnesses.adapters import PiHarnessAdapter

        assert PiHarnessAdapter is not None

    def test_inherits_from_harness_adapter(self):
        from openenv.core.harnesses import HarnessAdapter

        assert issubclass(PiHarnessAdapter, HarnessAdapter)


class TestPiHarnessAdapterInit:
    """Test initialization."""

    def test_stores_config(self, adapter, pi_config):
        assert adapter.config is pi_config

    def test_stores_mcp_server_url(self, adapter):
        assert adapter.mcp_server_url == "http://localhost:8000/mcp"

    def test_stores_provider(self, adapter):
        assert adapter.provider == "anthropic"

    def test_default_thinking_level(self, adapter):
        assert adapter.thinking_level == "medium"

    def test_custom_thinking_level(self, pi_config):
        adapter = PiHarnessAdapter(
            config=pi_config,
            mcp_server_url="http://localhost:8000/mcp",
            thinking_level="high",
        )
        assert adapter.thinking_level == "high"

    def test_process_starts_none(self, adapter):
        assert adapter._process is None


class TestPiHarnessAdapterInjectTools:
    """Test MCP tool injection via .mcp.json config file."""

    @pytest.mark.asyncio
    async def test_inject_creates_mcp_json(self, adapter, tmp_path):
        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(tmp_path / ".mcp.json"),
                "working_directory": str(tmp_path),
            }
        )

        class FakeTool:
            name = "submit_plan"

        await adapter.inject_tools([FakeTool()])

        config_path = tmp_path / ".mcp.json"
        assert config_path.exists()

        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert "openenv" in data["mcpServers"]
        assert data["mcpServers"]["openenv"]["url"] == "http://localhost:8000/mcp"

    @pytest.mark.asyncio
    async def test_inject_merges_with_existing_config(self, adapter, tmp_path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {"existing": {"url": "http://other:9000/mcp"}},
                    "otherSetting": True,
                }
            )
        )

        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(config_path),
                "working_directory": str(tmp_path),
            }
        )

        class FakeTool:
            name = "submit_plan"

        await adapter.inject_tools([FakeTool()])

        data = json.loads(config_path.read_text())
        assert "existing" in data["mcpServers"]
        assert "openenv" in data["mcpServers"]
        assert data["otherSetting"] is True

    @pytest.mark.asyncio
    async def test_inject_handles_corrupted_config(self, adapter, tmp_path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text("not valid json{{{")

        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(config_path),
                "working_directory": str(tmp_path),
            }
        )

        class FakeTool:
            name = "submit_plan"

        await adapter.inject_tools([FakeTool()])

        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert "openenv" in data["mcpServers"]

    @pytest.mark.asyncio
    async def test_inject_no_tools_skips_file(self, adapter, tmp_path):
        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(tmp_path / ".mcp.json"),
                "working_directory": str(tmp_path),
            }
        )
        await adapter.inject_tools([])
        config_path = tmp_path / ".mcp.json"
        assert not config_path.exists()

    @pytest.mark.asyncio
    async def test_inject_resolves_conflicts_with_builtins(self, adapter, tmp_path):
        """Tools named 'read', 'write', etc. get env_ prefix."""
        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(tmp_path / ".mcp.json"),
                "working_directory": str(tmp_path),
            }
        )

        from openenv.core.env_server.mcp_types import Tool

        read_tool = Tool(
            name="read",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
        )
        custom_tool = Tool(
            name="submit_plan",
            description="Submit a plan",
            input_schema={"type": "object", "properties": {}},
        )

        await adapter.inject_tools([read_tool, custom_tool])

        # Should have resolved the "read" conflict
        assert any(t["name"] == "env_read" for t in adapter._injected_tools)
        assert any(t["name"] == "submit_plan" for t in adapter._injected_tools)

    @pytest.mark.asyncio
    async def test_inject_defaults_to_working_directory(self, adapter, tmp_path):
        """Without mcp_config_path, uses working_directory/.mcp.json."""
        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": None,
                "working_directory": str(tmp_path),
            }
        )

        class FakeTool:
            name = "submit_plan"

        await adapter.inject_tools([FakeTool()])

        config_path = tmp_path / ".mcp.json"
        assert config_path.exists()


class TestPiHarnessAdapterLifecycle:
    """Test start/stop with mocked subprocess."""

    def _make_ready_response(self):
        return (
            json.dumps(
                {
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {
                        "model": {
                            "provider": "anthropic",
                            "id": "claude-sonnet-4-20250514",
                        }
                    },
                }
            )
            + "\n"
        )

    @pytest.mark.asyncio
    async def test_start_launches_rpc_mode(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            return_value=self._make_ready_response().encode()
        )

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await adapter.start("/workspace")

            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            cmd = call_args[0]
            assert "pi" in cmd
            assert "--mode" in cmd
            assert "rpc" in cmd
            assert "--no-session" in cmd
            assert "--provider" in cmd
            assert "anthropic" in cmd
            assert "--model" in cmd
            assert "claude-sonnet-4-20250514" in cmd
            assert call_args[1]["stdin"] == asyncio.subprocess.PIPE
            assert call_args[1]["stdout"] == asyncio.subprocess.PIPE
            assert call_args[1]["cwd"] == "/workspace"

    @pytest.mark.asyncio
    async def test_start_does_not_duplicate_flags(self, pi_config):
        """If command already has --mode rpc, don't add it again."""
        pi_config = pi_config.model_copy(
            update={"command": ["pi", "--mode", "rpc", "--no-session"]}
        )
        adapter = PiHarnessAdapter(
            config=pi_config,
            mcp_server_url="http://localhost:8000/mcp",
            provider="anthropic",
        )

        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            return_value=self._make_ready_response().encode()
        )

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await adapter.start("/workspace")

            cmd = mock_exec.call_args[0]
            # Should only have one --mode, not two
            assert cmd.count("--mode") == 1
            assert cmd.count("--no-session") == 1

    @pytest.mark.asyncio
    async def test_start_sends_get_state_for_readiness(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            return_value=self._make_ready_response().encode()
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await adapter.start("/workspace")

            # Should have sent get_state command
            written = mock_process.stdin.write.call_args_list[0][0][0]
            cmd = json.loads(written.decode().strip())
            assert cmd["type"] == "get_state"

    @pytest.mark.asyncio
    async def test_start_sets_thinking_level(self, pi_config):
        adapter = PiHarnessAdapter(
            config=pi_config,
            mcp_server_url="http://localhost:8000/mcp",
            thinking_level="high",
        )

        ready_resp = self._make_ready_response().encode()
        thinking_resp = (
            json.dumps(
                {"type": "response", "command": "set_thinking_level", "success": True}
            ).encode()
            + b"\n"
        )

        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[ready_resp, thinking_resp]
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await adapter.start("/workspace")

            # Should have sent both get_state and set_thinking_level
            calls = mock_process.stdin.write.call_args_list
            assert len(calls) == 2
            thinking_cmd = json.loads(calls[1][0][0].decode().strip())
            assert thinking_cmd["type"] == "set_thinking_level"
            assert thinking_cmd["level"] == "high"

    @pytest.mark.asyncio
    async def test_start_raises_on_timeout(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        # Never returns a response line
        mock_process.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="failed to start"):
                await adapter.start("/workspace")

    @pytest.mark.asyncio
    async def test_start_inherits_parent_env_when_env_vars_set(self, adapter):
        adapter.config = adapter.config.model_copy(
            update={"env_vars": {"CUSTOM_VAR": "custom_value"}}
        )

        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            return_value=self._make_ready_response().encode()
        )

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await adapter.start("/workspace")

            env = mock_exec.call_args[1]["env"]
            assert "PATH" in env
            assert env["CUSTOM_VAR"] == "custom_value"

    @pytest.mark.asyncio
    async def test_start_passes_none_env_when_no_overrides(self):
        config = HarnessConfig(
            name="pi",
            command=["pi"],
        )
        adapter = PiHarnessAdapter(
            config=config,
            mcp_server_url="http://localhost:8000/mcp",
        )

        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        ready_resp = (
            json.dumps(
                {
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"model": None},
                }
            ).encode()
            + b"\n"
        )
        mock_process.stdout.readline = AsyncMock(return_value=ready_resp)

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await adapter.start("/workspace")

            assert mock_exec.call_args[1]["env"] is None

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock(return_value=0)

        adapter._process = mock_process
        await adapter.stop()

        mock_process.terminate.assert_called_once()
        assert adapter._process is None

    @pytest.mark.asyncio
    async def test_stop_sends_abort_first(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock(return_value=0)

        adapter._process = mock_process
        await adapter.stop()

        # Should have sent abort before terminating
        written = mock_process.stdin.write.call_args[0][0]
        cmd = json.loads(written.decode().strip())
        assert cmd["type"] == "abort"

    @pytest.mark.asyncio
    async def test_stop_kills_on_timeout(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_process.kill = MagicMock()

        adapter._process = mock_process
        await adapter.stop()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert adapter._process is None

    @pytest.mark.asyncio
    async def test_stop_handles_broken_pipe(self, adapter):
        """Stop should handle broken pipe when sending abort."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock(side_effect=BrokenPipeError)
        mock_process.stdin.drain = AsyncMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock(return_value=0)

        adapter._process = mock_process
        await adapter.stop()

        mock_process.terminate.assert_called_once()
        assert adapter._process is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_no_process(self, adapter):
        await adapter.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_is_alive_false_when_no_process(self, adapter):
        assert await adapter.is_alive() is False

    @pytest.mark.asyncio
    async def test_is_alive_true_when_running(self, adapter):
        mock_process = MagicMock()
        mock_process.returncode = None
        adapter._process = mock_process
        assert await adapter.is_alive() is True

    @pytest.mark.asyncio
    async def test_is_alive_false_when_exited(self, adapter):
        mock_process = MagicMock()
        mock_process.returncode = 0
        adapter._process = mock_process
        assert await adapter.is_alive() is False


class TestPiHarnessAdapterSendMessage:
    """Test message sending with mocked process I/O."""

    def _make_agent_end(self, text="Done.", messages=None):
        """Build an agent_end event with optional messages."""
        if messages is None:
            messages = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                }
            ]
        return json.dumps({"type": "agent_end", "messages": messages}).encode() + b"\n"

    def _make_prompt_response(self):
        return (
            json.dumps(
                {"type": "response", "command": "prompt", "success": True}
            ).encode()
            + b"\n"
        )

    def _make_events(self, *events):
        """Convert event dicts into encoded lines."""
        return [json.dumps(e).encode() + b"\n" for e in events]

    @pytest.mark.asyncio
    async def test_send_message_basic(self, adapter):
        prompt_resp = self._make_prompt_response()
        agent_end = self._make_agent_end("Fixed the bug.")

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[prompt_resp, agent_end])

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("Fix the bug")

        assert resp.response == "Fixed the bug."
        assert resp.done is True
        # Should have LLM_REQUEST and TURN_COMPLETE at minimum
        assert any(e.type == HarnessEventType.LLM_REQUEST for e in resp.events)
        assert any(e.type == HarnessEventType.TURN_COMPLETE for e in resp.events)

    @pytest.mark.asyncio
    async def test_send_message_with_text_streaming(self, adapter):
        """Text deltas are accumulated into the response."""
        prompt_resp = self._make_prompt_response()

        delta1 = (
            json.dumps(
                {
                    "type": "message_update",
                    "message": {},
                    "assistantMessageEvent": {
                        "type": "text_delta",
                        "contentIndex": 0,
                        "delta": "Hello ",
                    },
                }
            ).encode()
            + b"\n"
        )

        delta2 = (
            json.dumps(
                {
                    "type": "message_update",
                    "message": {},
                    "assistantMessageEvent": {
                        "type": "text_delta",
                        "contentIndex": 0,
                        "delta": "world!",
                    },
                }
            ).encode()
            + b"\n"
        )

        agent_end = self._make_agent_end("Hello world!")

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[prompt_resp, delta1, delta2, agent_end]
        )

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("Say hello")

        # Text should be accumulated from deltas
        assert resp.response == "Hello world!"

        # Should have LLM_CHUNK events
        chunks = [e for e in resp.events if e.type == HarnessEventType.LLM_CHUNK]
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_send_message_with_tool_events(self, adapter):
        prompt_resp = self._make_prompt_response()

        tool_start = (
            json.dumps(
                {
                    "type": "tool_execution_start",
                    "toolCallId": "call_123",
                    "toolName": "bash",
                    "args": {"command": "ls"},
                }
            ).encode()
            + b"\n"
        )

        tool_end = (
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "toolCallId": "call_123",
                    "toolName": "bash",
                    "result": {"content": [{"type": "text", "text": "file1.py"}]},
                    "isError": False,
                }
            ).encode()
            + b"\n"
        )

        agent_end = self._make_agent_end("Listed files.")

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[prompt_resp, tool_start, tool_end, agent_end]
        )

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("List files")

        tool_calls = [e for e in resp.events if e.type == HarnessEventType.TOOL_CALL]
        tool_results = [
            e for e in resp.events if e.type == HarnessEventType.TOOL_RESULT
        ]

        assert len(tool_calls) == 1
        assert tool_calls[0].data["tool_name"] == "bash"
        assert len(tool_results) == 1
        assert tool_results[0].data["tool_name"] == "bash"
        assert tool_results[0].data["is_error"] is False

    @pytest.mark.asyncio
    async def test_send_message_raises_when_not_running(self, adapter):
        with pytest.raises(RuntimeError, match="not running"):
            await adapter.send_message("test")

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, adapter):
        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("slow task")

        assert "timeout" in resp.response.lower()
        assert resp.done is True
        error_events = [e for e in resp.events if e.type == HarnessEventType.ERROR]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_send_message_extension_error_captured(self, adapter):
        prompt_resp = self._make_prompt_response()

        ext_error = (
            json.dumps(
                {
                    "type": "extension_error",
                    "extensionPath": "/ext/bridge.ts",
                    "event": "tool_call",
                    "error": "Connection refused",
                }
            ).encode()
            + b"\n"
        )

        agent_end = self._make_agent_end("Error occurred.")

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[prompt_resp, ext_error, agent_end]
        )

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("do something")

        error_events = [e for e in resp.events if e.type == HarnessEventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].data["message"] == "Connection refused"


class TestPiHarnessAdapterStreaming:
    """Test streaming interface."""

    @pytest.mark.asyncio
    async def test_streaming_yields_events(self, adapter):
        prompt_resp = (
            json.dumps(
                {"type": "response", "command": "prompt", "success": True}
            ).encode()
            + b"\n"
        )

        agent_end = (
            json.dumps(
                {
                    "type": "agent_end",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Done."}],
                        }
                    ],
                }
            ).encode()
            + b"\n"
        )

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[prompt_resp, agent_end])

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        events = []
        async for event in adapter.send_message_streaming("test"):
            events.append(event)

        assert len(events) >= 2  # LLM_REQUEST + TURN_COMPLETE
        assert events[0].type == HarnessEventType.LLM_REQUEST
        assert events[-1].type == HarnessEventType.TURN_COMPLETE

    @pytest.mark.asyncio
    async def test_streaming_timeout(self, adapter):
        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        events = []
        async for event in adapter.send_message_streaming("slow task"):
            events.append(event)

        assert any(e.type == HarnessEventType.LLM_REQUEST for e in events)
        assert any(e.type == HarnessEventType.ERROR for e in events)


class TestPiHarnessAdapterEventMapping:
    """Test the pi event → HarnessEvent mapping."""

    def test_text_delta_maps_to_llm_chunk(self, adapter):
        event = adapter._map_pi_event(
            "message_update",
            {
                "type": "message_update",
                "message": {},
                "assistantMessageEvent": {
                    "type": "text_delta",
                    "contentIndex": 0,
                    "delta": "hello",
                },
            },
        )
        assert event is not None
        assert event.type == HarnessEventType.LLM_CHUNK
        assert event.data["delta"] == "hello"

    def test_toolcall_start_maps_to_tool_call(self, adapter):
        event = adapter._map_pi_event(
            "message_update",
            {
                "type": "message_update",
                "message": {},
                "assistantMessageEvent": {
                    "type": "toolcall_start",
                    "toolCall": {"id": "call_1", "name": "bash"},
                },
            },
        )
        assert event is not None
        assert event.type == HarnessEventType.TOOL_CALL
        assert event.data["tool_name"] == "bash"
        assert event.data["phase"] == "start"

    def test_toolcall_end_maps_to_tool_call(self, adapter):
        event = adapter._map_pi_event(
            "message_update",
            {
                "type": "message_update",
                "message": {},
                "assistantMessageEvent": {
                    "type": "toolcall_end",
                    "toolCall": {
                        "id": "call_1",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    },
                },
            },
        )
        assert event is not None
        assert event.type == HarnessEventType.TOOL_CALL
        assert event.data["phase"] == "end"
        assert event.data["arguments"] == {"command": "ls"}

    def test_tool_execution_start_maps(self, adapter):
        event = adapter._map_pi_event(
            "tool_execution_start",
            {
                "type": "tool_execution_start",
                "toolCallId": "call_1",
                "toolName": "bash",
                "args": {"command": "ls"},
            },
        )
        assert event is not None
        assert event.type == HarnessEventType.TOOL_CALL
        assert event.data["phase"] == "execution_start"

    def test_tool_execution_end_maps_to_tool_result(self, adapter):
        event = adapter._map_pi_event(
            "tool_execution_end",
            {
                "type": "tool_execution_end",
                "toolCallId": "call_1",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "output"}]},
                "isError": False,
            },
        )
        assert event is not None
        assert event.type == HarnessEventType.TOOL_RESULT
        assert event.data["is_error"] is False

    def test_message_start_maps_to_llm_response(self, adapter):
        event = adapter._map_pi_event(
            "message_start",
            {"type": "message_start", "message": {"role": "assistant"}},
        )
        assert event is not None
        assert event.type == HarnessEventType.LLM_RESPONSE
        assert event.data["phase"] == "start"

    def test_message_end_maps_to_llm_response(self, adapter):
        event = adapter._map_pi_event(
            "message_end",
            {"type": "message_end", "message": {"role": "assistant"}},
        )
        assert event is not None
        assert event.type == HarnessEventType.LLM_RESPONSE
        assert event.data["phase"] == "end"

    def test_extension_error_maps(self, adapter):
        event = adapter._map_pi_event(
            "extension_error",
            {
                "type": "extension_error",
                "extensionPath": "/ext/bridge.ts",
                "event": "tool_call",
                "error": "Connection refused",
            },
        )
        assert event is not None
        assert event.type == HarnessEventType.ERROR

    def test_agent_end_returns_none(self, adapter):
        """agent_end is handled by caller, not mapped."""
        event = adapter._map_pi_event(
            "agent_end",
            {"type": "agent_end", "messages": []},
        )
        assert event is None

    def test_unknown_event_returns_none(self, adapter):
        event = adapter._map_pi_event(
            "queue_update",
            {"type": "queue_update", "steering": [], "followUp": []},
        )
        assert event is None


class TestPiHarnessAdapterResponseExtraction:
    """Test response text extraction from agent_end."""

    def test_extracts_text_from_assistant_message(self, adapter):
        text = adapter._extract_response_from_agent_end(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello world"}],
                    }
                ]
            }
        )
        assert text == "Hello world"

    def test_extracts_from_last_assistant_message(self, adapter):
        text = adapter._extract_response_from_agent_end(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "First"}],
                    },
                    {"role": "user", "content": "response"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Last"}],
                    },
                ]
            }
        )
        assert text == "Last"

    def test_handles_empty_messages(self, adapter):
        text = adapter._extract_response_from_agent_end({"messages": []})
        assert text == ""

    def test_handles_string_content(self, adapter):
        text = adapter._extract_response_from_agent_end(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": ["Simple text"],
                    }
                ]
            }
        )
        assert text == "Simple text"

    def test_handles_no_messages_key(self, adapter):
        text = adapter._extract_response_from_agent_end({})
        assert text == ""


class TestPiHarnessAdapterEndToEnd:
    """Test PiHarnessAdapter with HarnessEnvironment."""

    def test_works_with_harness_environment(self, adapter):
        from openenv.core.harnesses import HarnessEnvironment

        env = HarnessEnvironment(adapter=adapter)
        assert env.adapter is adapter

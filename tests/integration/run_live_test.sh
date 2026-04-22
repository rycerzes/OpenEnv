#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Run the live pi harness integration test (Option A: local, no Docker)
#
# Usage:
#   export OPENENV_TEST_API_KEY="your-key"
#   export OPENENV_TEST_API_URL="https://your-endpoint.com/v1"
#   export OPENENV_TEST_MODEL="your-model-name"
#   bash tests/integration/run_live_test.sh
#
# Or pass inline:
#   OPENENV_TEST_API_KEY=... OPENENV_TEST_API_URL=... OPENENV_TEST_MODEL=... \
#       bash tests/integration/run_live_test.sh
# -----------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== Pi Harness Live Integration Test ==="
echo ""

# Check config
if [[ -z "${OPENENV_TEST_API_KEY:-}" ]] || [[ -z "${OPENENV_TEST_API_URL:-}" ]] || [[ -z "${OPENENV_TEST_MODEL:-}" ]]; then
    echo "ERROR: Set these environment variables:"
    echo "  OPENENV_TEST_API_KEY   - API key for OpenAI-compatible endpoint"
    echo "  OPENENV_TEST_API_URL   - Base URL (e.g. https://my-vllm.example.com/v1)"
    echo "  OPENENV_TEST_MODEL     - Model name (e.g. Qwen/Qwen3-32B)"
    echo ""
    echo "Example:"
    echo "  OPENENV_TEST_API_KEY=sk-xxx \\"
    echo "  OPENENV_TEST_API_URL=https://api.example.com/v1 \\"
    echo "  OPENENV_TEST_MODEL=Qwen/Qwen3-32B \\"
    echo "      bash tests/integration/run_live_test.sh"
    exit 1
fi

echo "API URL: ${OPENENV_TEST_API_URL}"
echo "Model:   ${OPENENV_TEST_MODEL}"
echo "Key:     ${OPENENV_TEST_API_KEY:0:8}..."
echo ""

# Check pi is available
if ! command -v pi &>/dev/null; then
    echo "ERROR: 'pi' not found on PATH. Install: npm install -g @mariozechner/pi-coding-agent"
    exit 1
fi
echo "pi version: $(pi --version)"

# Check pi-mcp-adapter is installed
if ! pi list 2>&1 | grep -q 'pi-mcp-adapter'; then
    echo "WARNING: pi-mcp-adapter not installed. Installing..."
    pi install npm:pi-mcp-adapter
fi

echo ""
echo "--- Running pytest ---"
PYTHONPATH=src:envs uv run pytest tests/integration/test_pi_harness_live.py -v -s --tb=long 2>&1

echo ""
echo "--- Or run directly (more verbose) ---"
echo "PYTHONPATH=src:envs uv run python -m tests.integration.test_pi_harness_live"

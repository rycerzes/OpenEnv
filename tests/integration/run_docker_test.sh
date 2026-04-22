#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Build and run the Docker integration test (Option B: containerized)
#
# Usage:
#   OPENENV_TEST_API_KEY=... \
#   OPENENV_TEST_API_URL=... \
#   OPENENV_TEST_MODEL=... \
#       bash tests/integration/run_docker_test.sh
#
# This validates the hackathon architecture:
#   Container starts → FastAPI/MCP server on :8000 → pi subprocess → LLM calls
# -----------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== Pi Harness Docker Integration Test ==="
echo ""

# Check config
if [[ -z "${OPENENV_TEST_API_KEY:-}" ]] || [[ -z "${OPENENV_TEST_API_URL:-}" ]] || [[ -z "${OPENENV_TEST_MODEL:-}" ]]; then
    echo "ERROR: Set these environment variables:"
    echo "  OPENENV_TEST_API_KEY   - API key for OpenAI-compatible endpoint"
    echo "  OPENENV_TEST_API_URL   - Base URL (e.g. https://my-vllm.example.com/v1)"
    echo "  OPENENV_TEST_MODEL     - Model name (e.g. Qwen/Qwen3-32B)"
    exit 1
fi

echo "API URL: ${OPENENV_TEST_API_URL}"
echo "Model:   ${OPENENV_TEST_MODEL}"
echo ""

IMAGE="openenv-integration:latest"

# Build
echo "--- Building Docker image ---"
docker build -t "$IMAGE" -f docker/Dockerfile.integration .
echo ""

# Run the in-container test
echo "--- Running integration test in container ---"
docker run --rm \
    -e OPENENV_TEST_API_KEY="${OPENENV_TEST_API_KEY}" \
    -e OPENENV_TEST_API_URL="${OPENENV_TEST_API_URL}" \
    -e OPENENV_TEST_MODEL="${OPENENV_TEST_MODEL}" \
    "$IMAGE" \
    python /app/run_integration_test.py

echo ""
echo "--- Docker test complete ---"

# Optional: also test that the server mode works (start server, connect from host)
echo ""
echo "To test server mode manually:"
echo "  docker run --rm -p 8000:8000 \\"
echo "      -e OPENENV_TEST_API_KEY=... \\"
echo "      -e OPENENV_TEST_API_URL=... \\"
echo "      -e OPENENV_TEST_MODEL=... \\"
echo "      $IMAGE"
echo ""
echo "  Then from host: curl http://localhost:8000/health"

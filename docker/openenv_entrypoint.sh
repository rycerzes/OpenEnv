#!/usr/bin/env bash
# -----------------------------------------------------------------------
# OpenEnv entrypoint — starts the FastAPI/MCP server,
# then waits forever (or for a signal) so the container stays alive.
#
# The training loop connects via HTTP/WS on :8000.
# pi is launched by the PiHarnessAdapter inside the Python process,
# NOT by this entrypoint.
# -----------------------------------------------------------------------
set -euo pipefail

echo "[openenv] Starting environment server on :8000 ..."

# Start the env server in the background
cd /app
exec python -m uvicorn server_app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info

#!/bin/bash

# ☁️ Cloud-Only Startup Script for Hugging Face Spaces
# All LLM calls are routed to Groq and HF Inference APIs via USE_HF_CLOUD=1
# Ollama is NOT used — image is kept lightweight to prevent HF storage purges.

echo "[HF-INIT] ===== Application Startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="

# 1. Start Backend Worker
echo "[HF-INIT] Starting Backend Worker on port 8700..."
python -u -m uvicorn "Agent.worker:app" --host 0.0.0.0 --port 8700 &

# 2. Start MCP Tool Server
echo "[HF-INIT] Starting MCP Tool Server on port 8701..."
python -m uvicorn "Agent.mcp_server:app" --host 0.0.0.0 --port 8701 &

# 3. Start Web UI
echo "[HF-INIT] Starting Web UI on port 7860..."
python -m uvicorn "web.app:app" --host 0.0.0.0 --port 7860 &

echo "[HF-INIT] All services started. Container is live."

# Keep the container alive
wait
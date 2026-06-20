#!/bin/bash

# Set OLLAMA_HOST based on the environment variable, stripping http://
if [ -n "$OLLAMA_BASE_URL" ]; then
    export OLLAMA_HOST=$(echo $OLLAMA_BASE_URL | sed 's|http://||' | sed 's|https://||')
else
    export OLLAMA_HOST=127.0.0.1:11434
fi

# Give the Ollama service a few seconds to start up
echo "Waiting for Ollama service at $OLLAMA_HOST to be ready..."
MAX_RETRIES=30
COUNT=0
while ! curl -s http://$OLLAMA_HOST/api/tags > /dev/null; do
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Ollama service not reachable after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Ollama is still warming up (attempt $COUNT/$MAX_RETRIES)..."
    sleep 2
done
echo "Ollama service is UP and running!"

# Pull required models (this will pull them on the dedicated ollama container)
echo "Pulling Qwen models via remote host (this may take a few minutes)..."
ollama pull qwen2.5-coder:1.5b
ollama pull qwen3.5:2b

# Start the three Microservices
echo "Starting Agentic Pipeline services..."

# 1. Start the Backend Worker
python -u -m uvicorn "Agent.worker:app" --host 0.0.0.0 --port 8700 &

# 2. Start the MCP Tool Server
python -m uvicorn "Agent.mcp_server:app" --host 0.0.0.0 --port 8701 &

# 3. Start the Web UI (This is the main process)
echo "Web UI will be available at http://0.0.0.0:8000"
python -m uvicorn "web.app:app" --host 0.0.0.0 --port 8000

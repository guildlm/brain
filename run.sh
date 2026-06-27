#!/bin/bash
# GuildLM Brain — convenience launcher.
#
# Starts the brain API server against a local Ollama instance. For ad-hoc
# routing use the CLI directly, e.g.:
#   brain classify "Build an HTTP API in Go" --offline
#   brain chat "Review my Go code for race conditions"
set -euo pipefail

echo "========================================="
echo "GuildLM Central Orchestrator (brain)"
echo "========================================="

BRAIN_BASE_URL="${BRAIN_BASE_URL:-http://localhost:11434/v1}"
BRAIN_ROUTER_MODEL="${BRAIN_ROUTER_MODEL:-qwen2.5:7b-instruct}"
OLLAMA_HOST="${BRAIN_BASE_URL%/v1}"
OLLAMA_HOST="${OLLAMA_HOST/http:\/\//}"

# 1. Ensure dependencies are importable.
if ! python3 -c "import fastapi, openai, typer" &> /dev/null; then
    echo "Installing brain dependencies..."
    python3 -m pip install -e .
fi

# 2. Check the inference backend is reachable (Ollama by default).
if ! curl -s "http://${OLLAMA_HOST}/api/tags" &> /dev/null; then
    echo "WARNING: inference backend not reachable at ${BRAIN_BASE_URL}."
    echo "Install Ollama from https://ollama.com and run 'ollama serve',"
    echo "or point BRAIN_BASE_URL at a vLLM/OpenAI-compatible endpoint."
fi

# 3. Ensure the router/base model is available (best-effort).
if curl -s "http://${OLLAMA_HOST}/api/tags" 2>/dev/null | grep -q "${BRAIN_ROUTER_MODEL%%:*}"; then
    echo "Base model '${BRAIN_ROUTER_MODEL}' is available."
else
    echo "NOTE: pull the base model with: ollama pull ${BRAIN_ROUTER_MODEL}"
fi

echo "-----------------------------------------"
echo "Starting brain API on http://0.0.0.0:8000 ..."
export PYTHONPATH="."
python3 -m src.cli serve "$@"

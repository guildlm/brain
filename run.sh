#!/bin/bash
# GuildLM Central Runner

echo "========================================="
echo "🧠 GuildLM Central Orchestrator"
echo "========================================="

# 1. Check if python environment has openai installed
if ! python3 -c "import openai" &> /dev/null; then
    echo "⚠️  OpenAI python package not found. Installing..."
    python3 -m pip install openai
fi

# 2. Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "❌ ERROR: Ollama is not running on localhost:11434."
    echo "Please install Ollama from https://ollama.com and run 'ollama serve'"
    exit 1
fi

# 3. Check if base model is downloaded
echo "Checking if base model 'qwen2.5:7b-instruct' is available..."
if ! curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:7b-instruct"; then
    echo "⚠️  Base model not found. Pulling Qwen2.5-7B (this may take a while)..."
    ollama pull qwen2.5:7b-instruct
fi

echo "✅ All systems go! Starting Brain..."
echo "-----------------------------------------"

# 4. Start Brain
export PYTHONPATH="."
python3 -m src.core.brain

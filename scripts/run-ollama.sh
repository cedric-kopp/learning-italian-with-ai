#!/usr/bin/env bash
# Run Tutore with the Ollama backend.
# Prereq: `ollama serve` running and `ollama pull qwen3:14b` done.
set -euo pipefail
cd "$(dirname "$0")/.."

export LLM_BACKEND=ollama

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "!! Ollama not reachable on :11434. Start it with:  ollama serve"
  echo "   and pull a model:  ollama pull ${OLLAMA_MODEL:-qwen3:14b}"
  exit 1
fi

echo "==> Tutore on http://localhost:${PORT:-8770}  (backend: ollama)"
exec uv run python -m tutore.app

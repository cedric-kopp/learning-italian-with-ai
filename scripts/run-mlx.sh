#!/usr/bin/env bash
# Run Tutore with the MLX backend. Starts mlx_lm.server if it isn't already up,
# then launches the app. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

export LLM_BACKEND=mlx
MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen3-14B-4bit}"
MLX_PORT="${MLX_PORT:-8080}"

started_server=0
if ! curl -sf "http://localhost:${MLX_PORT}/v1/models" >/dev/null 2>&1; then
  echo "==> Starting mlx_lm.server (${MLX_MODEL}) on :${MLX_PORT} …"
  uv run mlx_lm.server --model "$MLX_MODEL" --port "$MLX_PORT" &
  SERVER_PID=$!
  started_server=1
  trap '[ $started_server -eq 1 ] && kill $SERVER_PID 2>/dev/null || true' EXIT
  # Wait for it to come up (model load can take a bit).
  for _ in $(seq 1 60); do
    curl -sf "http://localhost:${MLX_PORT}/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
fi

echo "==> Tutore on http://localhost:${PORT:-8770}  (backend: mlx)"
exec uv run python -m tutore.app

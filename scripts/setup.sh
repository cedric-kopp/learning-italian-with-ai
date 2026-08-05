#!/usr/bin/env bash
# One-time setup for Tutore. Installs system + Python deps.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> System deps (ffmpeg for audio decode, espeak-ng for Italian TTS G2P)"
if command -v brew >/dev/null 2>&1; then
  brew list ffmpeg     >/dev/null 2>&1 || brew install ffmpeg
  brew list espeak-ng  >/dev/null 2>&1 || brew install espeak-ng
else
  echo "!! Homebrew not found — install ffmpeg and espeak-ng manually."
fi

echo "==> Python env (uv, pinned to 3.12)"
# Installs deps incl. the spaCy Italian model (pinned as a URL dep in pyproject).
uv sync

echo "==> .env"
[ -f .env ] || cp .env.example .env

cat <<'EOF'

Setup done. Next:
  • LLM engine — pick ONE:
      Ollama:  brew install ollama && ollama serve   (then: ollama pull qwen3:14b)
      MLX:     uv run mlx_lm.server --model mlx-community/Qwen3-14B-4bit --port 8080
  • Run the tutor:
      ./scripts/run-ollama.sh      (or)      ./scripts/run-mlx.sh
  • Open http://localhost:8000

First run downloads the Whisper + Kokoro models (a few GB) — give it a minute.
EOF

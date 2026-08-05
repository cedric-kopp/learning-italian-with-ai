"""Runtime configuration, loaded from environment / .env.

The one job that matters here is LLM backend swappability: `LLM_BACKEND`
selects a base URL + model id, and everything downstream is identical whether
we're talking to Ollama or an MLX server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class LLMConfig:
    backend: str  # "ollama" | "mlx"
    base_url: str
    model: str
    # Local servers ignore the key, but the OpenAI client insists on a non-empty one.
    api_key: str = "local"


def load_llm_config() -> LLMConfig:
    backend = _get("LLM_BACKEND", "ollama").strip().lower()
    if backend == "mlx":
        return LLMConfig(
            backend="mlx",
            base_url=_get("MLX_BASE_URL", "http://localhost:8080/v1"),
            model=_get("MLX_MODEL", "mlx-community/Qwen3-14B-4bit"),
        )
    # Default: ollama.
    return LLMConfig(
        backend="ollama",
        base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model=_get("OLLAMA_MODEL", "gemma3:4b"),
    )


LLM = load_llm_config()

WHISPER_MODEL = _get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

TTS_VOICE_IT = _get("TTS_VOICE_IT", "im_nicola")
TTS_VOICE_EN = _get("TTS_VOICE_EN", "am_michael")

HOST = _get("HOST", "127.0.0.1")
PORT = int(_get("PORT", "8770"))

DB_PATH = ROOT / _get("DB_PATH", "data/tutore.db")

# --- Anki flashcards (Phase 2) ---
# Bigger model used ONLY for the batch card-generation handoff at session end
# (latency irrelevant there). Chat still uses the fast LLM above.
CARD_MODEL = _get("CARD_MODEL", "qwen3:14b")
ANKI_CONNECT_URL = _get("ANKI_CONNECT_URL", "http://127.0.0.1:8765")
ANKI_DECK = _get("ANKI_DECK", "Tutore Italian")
ANKI_TAGS = [t.strip() for t in _get("ANKI_TAGS", "tutore,italian").split(",") if t.strip()]

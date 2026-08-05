# Tutore 🇮🇹 — a local, open-weight voice tutor for Italian

Speak Italian out loud, get corrected, and hear a natural spoken reply — all
running **locally** on Apple Silicon. Built for getting to A2+ speaking before an
Erasmus exchange.

**Pipeline:** your voice → `mlx-whisper` (STT) → local LLM → Kokoro (TTS) → spoken
reply. The transcript of what you said is shown too — an honest pronunciation
check (if Whisper mishears you, an Italian might too).

## What Phase 1 does

- 🎤 **Push-to-talk conversation** with *Marco*, a patient tutor scaled to your level.
- ✏️ **Live corrections** — see what you said vs the fixed version, with a short *why*.
- 🔁 **Two modes:** Italian immersion ↔ English teacher (toggle any time).
- 🎚️ **Level (A1–B1) + voice-speed** dials.
- 🔄 **Swappable brain:** run **Ollama** or **MLX** with one env var — identical UX.
- 🧠 **Memory foundation:** every turn saves new vocab + your mistakes to SQLite
  (the base the mistake ledger, scenarios, grammar lab & dashboard build on).

## What Phase 2 adds — Anki flashcards 🗂

- 🃏 **Session-end card generation** — the "🗂 Anki" panel's *"End session → create
  cards"* hands your session's new vocabulary to a bigger model (**qwen3:14b**),
  which writes proper cards: Italian front (with article for gender), **German**
  back + part of speech + an example sentence.
- 🔗 **Live push to Anki** via the **AnkiConnect** add-on — cards land straight in
  your chosen deck. Note ids are saved so re-running never duplicates; if Anki is
  closed, cards are held and pushed on your next click (no regeneration).
- 📥 **Italian-only known-vocab import** — paste words you already know; the tutor
  stops re-teaching them and won't turn them into cards.

### Anki setup (one-time)
1. Install the **AnkiConnect** add-on: Anki → *Tools → Add-ons → Get Add-ons* →
   code **`2055492159`** → restart Anki.
2. Keep **Anki desktop open** when you push cards. Default deck is *Tutore Italian*
   (change via `ANKI_DECK` in `.env` or the deck picker).

> Roadmap (next phases): Erasmus scenario role-plays → grammar lab →
> pronunciation drills → progress dashboard.

## Requirements

- macOS on Apple Silicon (built/tested on M4, 32GB).
- [`uv`](https://docs.astral.sh/uv/), Homebrew.
- One LLM engine: **Ollama** *or* **MLX** (`mlx_lm`).

## Setup

```bash
./scripts/setup.sh          # installs ffmpeg, espeak-ng, Python deps; writes .env
```

Then pick an LLM engine:

```bash
# Option A — Ollama
brew install ollama
ollama serve &
ollama pull qwen3:14b        # or gemma3:12b (edit OLLAMA_MODEL in .env)

# Option B — MLX (no separate install; runs in the project venv)
#   run-mlx.sh starts mlx_lm.server for you.
```

## Run

```bash
./scripts/run-ollama.sh      # or:  ./scripts/run-mlx.sh
```

Open **http://localhost:8770**, hold the mic button (or **Space**), and talk.

First launch downloads the Whisper + Kokoro models (a few GB), so give it a minute.

## Configuration

Everything lives in `.env` (copied from `.env.example`): `LLM_BACKEND`, model ids,
`WHISPER_MODEL`, TTS voices, port, DB path. Switch backends by setting
`LLM_BACKEND=ollama` or `mlx` — nothing else changes.

## Project layout

```
tutore/
  config.py   # env + LLM backend selection
  db.py       # SQLite/SQLModel schema = persistent memory
  stt.py      # mlx-whisper speech-to-text
  llm.py      # OpenAI-compatible client (Ollama/MLX) + JSON turn parsing
  prompts.py  # tutor system prompts + structured turn schema
  tts.py      # Kokoro text-to-speech
  memory.py   # profile block + per-turn persistence (vocab/mistakes)
  app.py      # FastAPI: /api/turn, /api/tts, serves web/
web/          # vanilla push-to-talk UI (no build step)
scripts/      # setup.sh, run-ollama.sh, run-mlx.sh
data/         # tutore.db (created on first run)
```

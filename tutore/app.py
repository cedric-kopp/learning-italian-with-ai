"""FastAPI app — wires STT -> LLM -> TTS and serves the web UI.

Phase 1 endpoints:
  POST /api/turn   audio + mode/level -> transcript, corrections, reply, audio
  POST /api/tts    text -> audio (for replays)
  GET  /api/profile / PUT /api/profile
  GET  /api/backend
  GET  /            static web app
"""

from __future__ import annotations

import base64
import os
import tempfile
import time

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, llm, memory, stt, tts
from .config import HOST, PORT, WEB_DIR

app = FastAPI(title="Tutore — Italian voice tutor")

# In-memory conversation history per session (Phase 1: single-process, fine).
_HISTORY: dict[int, list[dict[str, str]]] = {}
_MAX_HISTORY = 12  # keep the last N messages to bound context


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/api/backend")
def get_backend() -> dict:
    return llm.backend_info()


@app.get("/api/profile")
def get_profile() -> dict:
    p = memory.get_profile()
    return {
        "level": p.level,
        "goals": p.goals,
        "tts_speed": p.tts_speed,
        "voice_it": p.voice_it,
        "voice_en": p.voice_en,
    }


class ProfileUpdate(BaseModel):
    level: str | None = None
    goals: str | None = None
    tts_speed: float | None = None


@app.put("/api/profile")
def put_profile(update: ProfileUpdate) -> dict:
    with db.get_session() as s:
        p = s.get(db.Profile, 1)
        if update.level is not None:
            p.level = update.level
        if update.goals is not None:
            p.goals = update.goals
        if update.tts_speed is not None:
            p.tts_speed = max(0.5, min(1.5, update.tts_speed))
        s.add(p)
        s.commit()
    return {"ok": True}


class TTSBody(BaseModel):
    text: str
    lang: str = "it"
    speed: float | None = None


@app.post("/api/tts")
def post_tts(body: TTSBody) -> dict:
    speed = body.speed if body.speed is not None else memory.get_profile().tts_speed
    wav = tts.synthesize(body.text, lang=body.lang, speed=speed)
    return {"audio_b64": base64.b64encode(wav).decode("ascii")}


@app.post("/api/turn")
async def post_turn(
    audio: UploadFile,
    mode: str = Form("immersion"),
    level: str = Form(""),
    session_id: int = Form(0),
) -> JSONResponse:
    # 1) Persist upload to a temp file (ffmpeg/whisper read from disk).
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # 2) Speech -> text.
        t0 = time.perf_counter()
        tr = stt.transcribe(tmp_path, language="it")
        stt_ms = int((time.perf_counter() - t0) * 1000)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    profile = memory.get_profile()
    effective_level = level or profile.level

    if not tr.text.strip():
        # Nothing intelligible — ask for a repeat without hitting the LLM.
        reply = "Scusa, non ho sentito bene. Puoi ripetere, per favore?"
        wav = tts.synthesize(reply, lang="it", speed=profile.tts_speed)
        return JSONResponse(
            {
                "transcript": "",
                "stt_confidence": tr.confidence,
                "reply": reply,
                "reply_lang": "it",
                "corrections": [],
                "new_vocab": [],
                "audio_b64": base64.b64encode(wav).decode("ascii"),
            }
        )

    # 3) Text -> tutor (structured JSON).
    sid = session_id or 0
    history = _HISTORY.setdefault(sid, [])
    result = llm.run_turn(
        mode=mode,
        level=effective_level,
        profile_block=memory.profile_block(),
        history=history,
        user_text=tr.text,
    )

    # Update rolling history.
    history.append({"role": "user", "content": tr.text})
    history.append({"role": "assistant", "content": result.reply})
    del history[:-_MAX_HISTORY]

    # 4) Persist to memory (vocab + mistakes harvested here).
    memory.persist_turn(
        session_id=sid or None,
        transcript=tr.text,
        stt_confidence=tr.confidence,
        result=result,
    )

    # 5) Reply -> speech.
    t1 = time.perf_counter()
    wav = tts.synthesize(result.reply, lang=result.reply_lang, speed=profile.tts_speed)
    tts_ms = int((time.perf_counter() - t1) * 1000)

    timings = {"stt_ms": stt_ms, "llm_ms": result.llm_ms, "tts_ms": tts_ms}
    print(f"[turn] stt={stt_ms}ms  llm={result.llm_ms}ms  tts={tts_ms}ms", flush=True)

    return JSONResponse(
        {
            "transcript": tr.text,
            "stt_confidence": tr.confidence,
            "reply": result.reply,
            "reply_lang": result.reply_lang,
            "detected_level": result.detected_level,
            "corrections": [c.__dict__ for c in result.corrections],
            "new_vocab": [v.__dict__ for v in result.new_vocab],
            "audio_b64": base64.b64encode(wav).decode("ascii"),
            "timings": timings,
        }
    )


# Static web UI (mounted last so /api/* wins).
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run("tutore.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()

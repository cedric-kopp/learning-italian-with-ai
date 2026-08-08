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

from fastapi import Body, FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import select

from . import anki, cards, db, llm, memory, scenarios, stt, tts
from .config import ANKI_DECK, APP_NAME, HOST, NATIVE, PORT, TARGET, TUTOR_NAME, WEB_DIR

app = FastAPI(title=f"{APP_NAME} — {TARGET.name_en} voice tutor")

# In-memory conversation history per session (Phase 1: single-process, fine).
_HISTORY: dict[int, list[dict[str, str]]] = {}
_MAX_HISTORY = 12  # keep the last N messages to bound context


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    memory.ensure_known_csv()  # recreate the gitignored known-vocab file if missing


@app.get("/api/backend")
def get_backend() -> dict:
    return llm.backend_info()


@app.get("/api/config")
def get_config() -> dict:
    """Drives the (language-agnostic) frontend: title, flag, mode labels, tutor."""
    return {
        "app_name": APP_NAME,
        "tutor_name": TUTOR_NAME,
        "target": {"code": TARGET.code, "name": TARGET.name_en, "flag": TARGET.flag},
        "native": {"code": NATIVE.code, "name": NATIVE.name_en, "flag": NATIVE.flag},
    }


@app.get("/api/profile")
def get_profile() -> dict:
    p = memory.get_profile()
    return {
        "level": p.level,
        "goals": p.goals,
        "tts_speed": p.tts_speed,
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
    lang: str = TARGET.code
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
    length: str = Form("medium"),
    scenario_id: str = Form(""),
    session_id: int = Form(0),
) -> JSONResponse:
    # 1) Persist upload to a temp file (ffmpeg/whisper read from disk).
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # 2) Speech -> text. Immersion biases to the target language (steadier
        # transcription + honest pronunciation signal); teacher mode auto-detects so
        # a native-language question isn't force-transcribed into target gibberish.
        t0 = time.perf_counter()
        stt_lang = TARGET.whisper if mode == "immersion" else None
        tr = stt.transcribe(tmp_path, language=stt_lang)
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
        reply = TARGET.repeat_prompt
        wav = tts.synthesize(reply, lang=TARGET.code, speed=profile.tts_speed)
        return JSONResponse(
            {
                "transcript": "",
                "stt_confidence": tr.confidence,
                "reply": reply,
                "reply_lang": TARGET.code,
                "corrections": [],
                "new_vocab": [],
                "audio_b64": base64.b64encode(wav).decode("ascii"),
            }
        )

    # 3) Text -> tutor (structured JSON).
    sid = session_id or 0
    history = _HISTORY.setdefault(sid, [])
    scenario = scenarios.get(scenario_id) if mode == "scenario" else None
    result = llm.run_turn(
        mode=mode,
        level=effective_level,
        # Role-play stays in character; skip the learner-profile injection.
        profile_block="" if scenario else memory.profile_block(),
        history=history,
        user_text=tr.text,
        length=length,
        scenario=scenario,
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
        mode=mode,
    )

    # Scenario: stamp completion the first time the goal is met.
    if scenario and result.goal_met:
        _complete_scenario_run(sid or None, scenario.id)

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
            "goal_met": result.goal_met,
            "audio_b64": base64.b64encode(wav).decode("ascii"),
            "timings": timings,
        }
    )


# ----------------------------------------------------------------------------
# Phase 2: known-vocab import + Anki flashcard generation
# ----------------------------------------------------------------------------


@app.post("/api/vocab/import")
async def import_vocab(
    text: str = Form(""),
    file: UploadFile | None = None,
) -> dict:
    """Ingest a target-language-only list of words the learner already knows."""
    payload = text or ""
    if file is not None:
        payload += "\n" + (await file.read()).decode("utf-8", errors="ignore")
    result = memory.import_known_vocab(payload)
    return result


@app.get("/api/anki/status")
def anki_status() -> dict:
    available = anki.is_available()
    decks = anki.deck_names() if available else []
    return {"available": available, "decks": decks, "default_deck": ANKI_DECK}


class GenerateBody(BaseModel):
    deck: str | None = None
    session_id: int | None = None


def _card_from_row(row) -> "cards.Card":
    return cards.Card(
        term=row.term, translation=row.translation, article=row.article,
        pos=row.pos, example=row.example,
    )


@app.post("/api/cards/generate")
def generate_cards_endpoint(body: GenerateBody = Body(default=GenerateBody())) -> dict:
    """Session-end handoff. The big model (qwen3:14b) curates card-worthy vocab
    from the whole session transcript — filtered + deduped — then pushes to Anki.
    If cards were already generated but not pushed (Anki was closed), we just push
    them again (no re-curation)."""
    deck = body.deck or ANKI_DECK
    generated = 0

    pending = memory.pending_cards()  # card_ready AND not yet exported
    if pending:
        push_cards = [(r.id, _card_from_row(r)) for r in pending]
    else:
        transcript = memory.session_transcript(body.session_id)
        if not transcript.strip():
            return {"generated": 0, "pushed": 0, "skipped": 0,
                    "message": f"No conversation yet — talk with {TUTOR_NAME} first."}

        exclude = memory.card_exclude_set()
        built = cards.curate_and_generate(transcript, exclude)  # qwen3:14b, slow

        # Defensive post-filter: honor the exclude set + dedupe by normalized form.
        seen: set[str] = set()
        filtered = []
        for c in built:
            key = memory._norm(c.term)
            if not key or key in exclude or key in seen:
                continue
            seen.add(key)
            filtered.append(c)

        if not filtered:
            return {"generated": 0, "pushed": 0, "skipped": 0,
                    "message": "No new card-worthy vocabulary found this session."}

        # Store as card_ready conversation vocab (upsert by normalized term).
        ids = []
        with db.get_session() as s:
            existing = {memory._norm(v.term): v for v in s.exec(select(db.Vocab)).all()}
            for c in filtered:
                key = memory._norm(c.term)
                row = existing.get(key) or db.Vocab(term=c.term, source="conversation")
                row.translation, row.article, row.pos = c.translation, c.article, c.pos
                if c.example:
                    row.example = c.example
                row.card_ready = True
                s.add(row)
                s.flush()
                existing[key] = row
                ids.append(row.id)
            s.commit()
        generated = len(filtered)

        with db.get_session() as s:
            push_cards = []
            for vid in ids:
                row = s.get(db.Vocab, vid)
                if row is not None:
                    push_cards.append((vid, _card_from_row(row)))

    if not push_cards:
        return {"generated": generated, "pushed": 0, "skipped": 0,
                "message": "Card generation returned nothing usable."}

    if not anki.is_available():
        return {
            "generated": generated,
            "pushed": 0,
            "skipped": 0,
            "anki_available": False,
            "message": f"{len(push_cards)} card(s) ready, but Anki isn't open. "
            "Open Anki (with the AnkiConnect add-on) and click again to push them.",
        }

    push = anki.add_cards([c for _, c in push_cards], deck=deck)

    pushed_ids: list[int] = []
    with db.get_session() as s:
        for (vid, _), note_id in zip(push_cards, push.note_ids):
            if not note_id:
                continue
            row = s.get(db.Vocab, vid)
            if row is None:
                continue
            row.anki_note_id = int(note_id)
            row.exported_at = memory._now()
            s.add(row)
            pushed_ids.append(vid)
        s.commit()

    # Auto-promote carded words to "known" (stops re-teaching; syncs the CSV).
    promoted = memory.promote_to_known(pushed_ids)

    return {
        "generated": generated,
        "pushed": push.pushed,
        "skipped": push.skipped,
        "deck": deck,
        "anki_available": True,
        "promoted": promoted,
        "message": f"Added {push.pushed} card(s) to “{deck}”."
        + (f" {push.skipped} skipped (already in deck)." if push.skipped else "")
        + (f" {promoted} added to your known list." if promoted else ""),
    }


@app.get("/api/vocab")
def list_vocab(limit: int = 100) -> dict:
    with db.get_session() as s:
        rows = s.exec(select(db.Vocab).order_by(db.Vocab.created_at.desc()).limit(limit)).all()
        out = [
            {
                "term": r.term,
                "translation": r.translation,
                "source": r.source,
                "card_ready": r.card_ready,
                "exported": r.exported_at is not None,
            }
            for r in rows
        ]
    return {"vocab": out}


# ----------------------------------------------------------------------------
# Phase 3: scenario role-plays
# ----------------------------------------------------------------------------


def _completed_scenario_ids() -> set[str]:
    with db.get_session() as s:
        rows = s.exec(select(db.ScenarioRun).where(db.ScenarioRun.completed_at.is_not(None))).all()
        return {r.scenario_id for r in rows}


def _complete_scenario_run(session_id: int | None, scenario_id: str) -> None:
    with db.get_session() as s:
        q = (
            select(db.ScenarioRun)
            .where(db.ScenarioRun.scenario_id == scenario_id)
            .where(db.ScenarioRun.completed_at.is_(None))
            .order_by(db.ScenarioRun.started_at.desc())
        )
        if session_id:
            q = q.where(db.ScenarioRun.session_id == session_id)
        run = s.exec(q).first()
        if run is not None:
            run.completed_at = memory._now()
            s.add(run)
            s.commit()


@app.get("/api/scenarios")
def list_scenarios() -> dict:
    done = _completed_scenario_ids()
    return {
        "scenarios": [
            {
                "id": sc.id, "emoji": sc.emoji, "title": sc.title,
                "setting": sc.setting, "character": sc.character, "goal": sc.goal,
                "completed": sc.id in done,
            }
            for sc in scenarios.SCENARIOS
        ]
    }


class ScenarioStart(BaseModel):
    scenario_id: str
    session_id: int = 0
    level: str = ""
    length: str = "medium"


@app.post("/api/scenario/start")
def start_scenario(body: ScenarioStart) -> JSONResponse:
    scenario = scenarios.get(body.scenario_id)
    if scenario is None:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)

    profile = memory.get_profile()
    level = body.level or profile.level
    result = llm.opening_line(scenario, level, length=body.length)

    # Fresh role-play history: the opening line is the assistant's first message.
    sid = body.session_id or 0
    _HISTORY[sid] = [{"role": "assistant", "content": result.reply}]

    with db.get_session() as s:
        s.add(db.ScenarioRun(scenario_id=scenario.id, session_id=sid or None))
        s.commit()

    wav = tts.synthesize(result.reply, lang=result.reply_lang or TARGET.code, speed=profile.tts_speed)
    return JSONResponse(
        {
            "scenario_id": scenario.id,
            "reply": result.reply,
            "reply_lang": result.reply_lang,
            "audio_b64": base64.b64encode(wav).decode("ascii"),
        }
    )


# Static web UI (mounted last so /api/* wins).
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run("tutore.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()

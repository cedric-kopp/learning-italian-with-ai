"""SQLite persistence via SQLModel — this *is* the tutor's memory.

Phase 1 actively writes: Profile, Session, Turn, Vocab, Mistake.
Document / PronAttempt / GrammarAttempt are defined now so Phases 2-6 slot in
without a migration.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine
from sqlmodel import Session as DBSession

from .config import DB_PATH


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Profile(SQLModel, table=True):
    """Single-row learner profile. Injected into every tutor prompt."""

    id: Optional[int] = Field(default=1, primary_key=True)
    level: str = "A2"  # working CEFR level, A1..B1
    goals: str = "Speak confidently for everyday life."
    tts_speed: float = 0.9  # 1.0 = natural; <1 slower for beginners
    voice_target: str = ""  # Kokoro voice for the language being learned
    voice_native: str = ""  # Kokoro voice for the explanation/native language
    updated_at: dt.datetime = Field(default_factory=_now)


class ChatSession(SQLModel, table=True):
    """One conversation sitting."""

    __tablename__ = "session"

    id: Optional[int] = Field(default=None, primary_key=True)
    started_at: dt.datetime = Field(default_factory=_now)
    ended_at: Optional[dt.datetime] = None
    mode: str = "immersion"  # "immersion" | "english_teacher"
    summary: Optional[str] = None


class Turn(SQLModel, table=True):
    """One user-speaks / tutor-replies exchange."""

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: Optional[int] = Field(default=None, foreign_key="session.id", index=True)
    created_at: dt.datetime = Field(default_factory=_now)
    mode: str = ""  # "immersion" | "english_teacher" | "scenario" (for transcript review)
    transcript: str = ""  # what Whisper heard (pronunciation signal)
    stt_confidence: Optional[float] = None  # avg logprob proxy, higher = clearer
    tutor_reply: str = ""
    corrections_json: str = "[]"  # serialized list of correction dicts


class Vocab(SQLModel, table=True):
    """A vocabulary item with SM-2 spaced-repetition scheduling (Phase 2 UI)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    term: str = Field(index=True)  # the word/phrase in the TARGET language
    example: str = ""
    source: str = "conversation"  # "conversation" | "upload" | "manual"
    created_at: dt.datetime = Field(default_factory=_now)
    # SM-2 fields (retained; scheduling now handled by Anki, so unused for review)
    ease: float = 2.5
    interval_days: float = 0.0
    reps: int = 0
    lapses: int = 0
    due_at: dt.datetime = Field(default_factory=_now)
    # Anki card fields (Phase 2). Populated by the qwen3:14b card-generation pass.
    translation: str = ""  # the word in the NATIVE language (card back)
    article: str = ""  # definite article for nouns, encodes gender: il/la/lo/l'
    pos: str = ""  # part of speech: noun/verb/adjective/...
    card_ready: bool = False  # big model has enriched this into a studyable card
    anki_note_id: Optional[int] = None  # note id returned by AnkiConnect once pushed
    exported_at: Optional[dt.datetime] = None


class Mistake(SQLModel, table=True):
    """A logged error for the mistake ledger + targeted review."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: dt.datetime = Field(default_factory=_now)
    kind: str = "grammar"  # "grammar" | "vocab" | "pronunciation"
    your_text: str = ""
    correction: str = ""
    explanation: str = ""
    tag: str = ""  # e.g. "passato_prossimo_aux", "prep_di_vs_a"


class ScenarioRun(SQLModel, table=True):
    """A role-play session for a scenario (Phase 3). completed_at set on goal_met."""

    id: Optional[int] = Field(default=None, primary_key=True)
    scenario_id: str = Field(index=True)
    session_id: Optional[int] = Field(default=None, index=True)
    started_at: dt.datetime = Field(default_factory=_now)
    completed_at: Optional[dt.datetime] = None


class Document(SQLModel, table=True):
    """Uploaded material the learner has already covered (Phase 2 ingest)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: dt.datetime = Field(default_factory=_now)
    title: str = ""
    kind: str = "notes"  # "known_vocab" | "notes" | "material"
    content: str = ""


class PronAttempt(SQLModel, table=True):
    """Read-aloud drill result (Phase 5)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: dt.datetime = Field(default_factory=_now)
    target: str = ""
    recognized: str = ""
    score: float = 0.0
    focus: str = ""  # sound being drilled, e.g. "gli"


class GrammarAttempt(SQLModel, table=True):
    """Grammar-lab exercise result (Phase 4)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: dt.datetime = Field(default_factory=_now)
    topic: str = ""
    prompt: str = ""
    answer: str = ""
    correct: bool = False
    explanation: str = ""


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
        )
    return _engine


def _default_literal(col) -> str:
    """SQL literal for a NOT NULL column's default (needed to ALTER ADD COLUMN on
    a table that already has rows)."""
    default = getattr(col, "default", None)
    val = default.arg if default is not None and getattr(default, "is_scalar", False) else None
    if val is None:
        try:
            val = col.type.python_type()  # e.g. str() -> "", int() -> 0
        except Exception:
            val = ""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, float)):
        return str(val)
    return "''"


# Column renames for the i18n refactor — applied first so data is preserved
# (SQLite RENAME COLUMN needs 3.25+). Idempotent: only renames old->new if the old
# column still exists and the new one doesn't yet.
_RENAMES: dict[str, list[tuple[str, str]]] = {
    "vocab": [("italian", "term"), ("german", "translation")],
    "profile": [("voice_it", "voice_target"), ("voice_en", "voice_native")],
}


def _migrate() -> None:
    """Rename legacy columns, then add any model columns missing from existing
    tables. create_all() makes new tables but never ALTERs existing ones."""
    from sqlalchemy import inspect, text

    engine = get_engine()
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())

    # 1) Renames first (preserve existing data).
    with engine.begin() as conn:
        for tbl, pairs in _RENAMES.items():
            if tbl not in existing_tables:
                continue
            cols = {c["name"] for c in insp.get_columns(tbl)}
            for old, new in pairs:
                if old in cols and new not in cols:
                    conn.execute(text(f'ALTER TABLE "{tbl}" RENAME COLUMN "{old}" TO "{new}"'))

    # 2) Additive column pass (re-inspect after any renames).
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # freshly created by create_all with all columns
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                coltype = col.type.compile(engine.dialect)
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}'
                if not col.nullable:
                    ddl += f" DEFAULT {_default_literal(col)}"
                conn.execute(text(ddl))


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())
    _migrate()
    # Ensure the singleton profile row exists.
    with get_session() as s:
        if s.get(Profile, 1) is None:
            s.add(Profile(id=1))
            s.commit()


def get_session() -> DBSession:
    return DBSession(get_engine())

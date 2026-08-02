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
    goals: str = "Speak Italian confidently for daily life and university in Rome."
    tts_speed: float = 0.9  # 1.0 = natural; <1 slower for beginners
    voice_it: str = "im_nicola"
    voice_en: str = "am_michael"
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
    transcript: str = ""  # what Whisper heard (pronunciation signal)
    stt_confidence: Optional[float] = None  # avg logprob proxy, higher = clearer
    tutor_reply: str = ""
    corrections_json: str = "[]"  # serialized list of correction dicts


class Vocab(SQLModel, table=True):
    """A vocabulary item with SM-2 spaced-repetition scheduling (Phase 2 UI)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    italian: str = Field(index=True)
    english: str = ""
    example: str = ""
    source: str = "conversation"  # "conversation" | "upload" | "manual"
    created_at: dt.datetime = Field(default_factory=_now)
    # SM-2 fields
    ease: float = 2.5
    interval_days: float = 0.0
    reps: int = 0
    lapses: int = 0
    due_at: dt.datetime = Field(default_factory=_now)


class Mistake(SQLModel, table=True):
    """A logged error for the mistake ledger + targeted review."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: dt.datetime = Field(default_factory=_now)
    kind: str = "grammar"  # "grammar" | "vocab" | "pronunciation"
    your_text: str = ""
    correction: str = ""
    explanation: str = ""
    tag: str = ""  # e.g. "passato_prossimo_aux", "prep_di_vs_a"


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


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())
    # Ensure the singleton profile row exists.
    with get_session() as s:
        if s.get(Profile, 1) is None:
            s.add(Profile(id=1))
            s.commit()


def get_session() -> DBSession:
    return DBSession(get_engine())

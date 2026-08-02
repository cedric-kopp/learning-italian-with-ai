"""Memory: compile the learner-profile prompt block and persist each turn.

Phase 1 scope:
- profile_block(): a compact digest (level, goals, known-vocab sample, top
  recurring mistakes) injected into the tutor's system prompt.
- persist_turn(): write the exchange + auto-harvest new vocab + log mistakes.
Phases 2-6 extend this (uploads, SRS, richer retrieval).
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter

from sqlmodel import select

from . import db
from .db import Mistake, Profile, Turn, Vocab
from .llm import TurnResult


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def get_profile() -> Profile:
    with db.get_session() as s:
        p = s.get(Profile, 1)
        if p is None:
            p = Profile(id=1)
            s.add(p)
            s.commit()
            s.refresh(p)
        return p


def profile_block(max_vocab: int = 40, max_tags: int = 6) -> str:
    """Human-readable summary of what we know about the learner."""
    with db.get_session() as s:
        p = s.get(Profile, 1) or Profile(id=1)
        vocab = s.exec(select(Vocab).order_by(Vocab.created_at.desc()).limit(max_vocab)).all()
        mistakes = s.exec(select(Mistake).order_by(Mistake.created_at.desc()).limit(100)).all()

    lines = [f"- Working level: {p.level}", f"- Goals: {p.goals}"]

    if vocab:
        words = ", ".join(v.italian for v in vocab if v.italian)
        lines.append(f"- Recently learned words (avoid re-teaching): {words}")

    tags = Counter(m.tag for m in mistakes if m.tag)
    if tags:
        top = ", ".join(f"{t} (x{n})" for t, n in tags.most_common(max_tags))
        lines.append(f"- Recurring mistakes to watch for and reinforce: {top}")

    return "\n".join(lines)


def persist_turn(
    *,
    session_id: int | None,
    transcript: str,
    stt_confidence: float | None,
    result: TurnResult,
) -> None:
    """Save the turn and harvest vocab + mistakes into long-term memory."""
    with db.get_session() as s:
        s.add(
            Turn(
                session_id=session_id,
                transcript=transcript,
                stt_confidence=stt_confidence,
                tutor_reply=result.reply,
                corrections_json=json.dumps(
                    [c.__dict__ for c in result.corrections], ensure_ascii=False
                ),
            )
        )

        for c in result.corrections:
            if not (c.correction or c.your_text):
                continue
            s.add(
                Mistake(
                    kind="grammar",
                    your_text=c.your_text,
                    correction=c.correction,
                    explanation=c.explanation,
                    tag=c.tag,
                )
            )

        for v in result.new_vocab:
            italian = v.italian.strip()
            if not italian:
                continue
            exists = s.exec(select(Vocab).where(Vocab.italian == italian)).first()
            if exists:
                continue
            s.add(
                Vocab(
                    italian=italian,
                    english=v.english,
                    example=v.example,
                    source="conversation",
                    due_at=_now(),  # available immediately for Phase 2 SRS
                )
            )

        s.commit()

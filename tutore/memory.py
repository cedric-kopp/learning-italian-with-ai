"""Memory: compile the learner-profile prompt block and persist each turn.

Phase 1 scope:
- profile_block(): a compact digest (level, goals, known-vocab sample, top
  recurring mistakes) injected into the tutor's system prompt.
- persist_turn(): write the exchange + auto-harvest new vocab + log mistakes.
Phases 2-6 extend this (uploads, SRS, richer retrieval).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import Counter

from sqlmodel import select

from . import db
from .config import ROOT
from .db import Mistake, Profile, Turn, Vocab
from .llm import TurnResult

KNOWN_CSV = ROOT / "italian_known_vocab.csv"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


_ARTICLES = {"il", "lo", "la", "i", "gli", "le", "un", "una", "uno"}


def _norm(word: str) -> str:
    """Normalize an Italian word for dedup: lowercase, strip punctuation, and drop
    a leading definite/indefinite article so 'la casa' == 'casa', 'Piacere' ==
    'piacere'."""
    w = (word or "").strip().lower()
    w = re.sub(r"^[\"'“”«»\-\s]+|[\"'“”«».,;:!?]+$", "", w)
    w = re.sub(r"^(l'|un')\s*", "", w)  # attached apostrophe articles
    parts = w.split()
    if len(parts) > 1 and parts[0] in _ARTICLES:
        parts = parts[1:]
    return " ".join(parts).strip()


def get_profile() -> Profile:
    with db.get_session() as s:
        p = s.get(Profile, 1)
        if p is None:
            p = Profile(id=1)
            s.add(p)
            s.commit()
            s.refresh(p)
        return p


def profile_block(max_known: int = 60, max_recent: int = 25, max_tags: int = 6) -> str:
    """Human-readable summary of what we know about the learner."""
    with db.get_session() as s:
        p = s.get(Profile, 1) or Profile(id=1)
        known = s.exec(
            select(Vocab).where(Vocab.source == "upload").order_by(Vocab.created_at.desc()).limit(max_known)
        ).all()
        recent = s.exec(
            select(Vocab).where(Vocab.source == "conversation").order_by(Vocab.created_at.desc()).limit(max_recent)
        ).all()
        mistakes = s.exec(select(Mistake).order_by(Mistake.created_at.desc()).limit(100)).all()

    lines = [f"- Working level: {p.level}", f"- Goals: {p.goals}"]

    if known:
        words = ", ".join(v.italian for v in known if v.italian)
        lines.append(f"- Words the learner ALREADY KNOWS (do not re-teach these): {words}")

    if recent:
        words = ", ".join(v.italian for v in recent if v.italian)
        lines.append(f"- Recently introduced in conversation: {words}")

    tags = Counter(m.tag for m in mistakes if m.tag)
    if tags:
        top = ", ".join(f"{t} (x{n})" for t, n in tags.most_common(max_tags))
        lines.append(f"- Recurring mistakes to watch for and reinforce: {top}")

    return "\n".join(lines)


# Split on newlines, commas, semicolons, or bullet characters.
_VOCAB_SPLIT = re.compile(r"[\n,;••]+")


def import_known_vocab(text: str) -> dict:
    """Ingest an Italian-only list of words the learner already knows.

    Format is forgiving: one word/phrase per line, or comma/semicolon separated.
    Leading bullets/numbering are stripped. Returns {added, skipped}.
    """
    seen_new: set[str] = set()
    candidates = []
    for raw in _VOCAB_SPLIT.split(text or ""):
        word = raw.strip().strip("-*0123456789.) \t").strip()
        if not word:
            continue
        key = word.lower()
        if key in seen_new:
            continue
        seen_new.add(key)
        candidates.append(word)

    added = skipped = 0
    with db.get_session() as s:
        existing = {_norm(v.italian): v for v in s.exec(select(Vocab)).all()}
        for word in candidates:
            key = _norm(word)
            row = existing.get(key)
            if row is not None:
                # Promote a conversation word to "known" if the learner lists it.
                if row.source != "upload":
                    row.source = "upload"
                    s.add(row)
                skipped += 1
                continue
            new_row = Vocab(italian=word, source="upload", card_ready=False, due_at=_now())
            s.add(new_row)
            existing[key] = new_row
            added += 1
        s.commit()
    if added or skipped:
        export_known_vocab_csv()  # keep the committed file in sync with the DB
    return {"added": added, "skipped": skipped}


def export_known_vocab_csv() -> int:
    """Mirror the DB's known list (source='upload') to italian_known_vocab.csv so
    the committed file always reflects the source of truth. Called on every change."""
    with db.get_session() as s:
        words = sorted(
            (v.italian for v in s.exec(select(Vocab).where(Vocab.source == "upload")).all()),
            key=str.lower,
        )
    with open(KNOWN_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["italian"])
        for word in words:
            w.writerow([word])
    return len(words)


def promote_to_known(vocab_ids: list[int]) -> int:
    """Mark carded conversation words as 'known' so the tutor stops re-teaching them
    and they roll into the known list. Re-exports the CSV. Returns count promoted."""
    if not vocab_ids:
        return 0
    n = 0
    with db.get_session() as s:
        for vid in vocab_ids:
            row = s.get(Vocab, vid)
            if row is not None and row.source != "upload":
                row.source = "upload"
                s.add(row)
                n += 1
        s.commit()
    if n:
        export_known_vocab_csv()
    return n


def session_transcript(session_id: int | None, max_turns: int = 60) -> str:
    """Reconstruct a session's conversation as 'Tu: … / Marco: …' lines for the
    curator model. Falls back to the most recent turns if no session id."""
    with db.get_session() as s:
        q = select(Turn)
        if session_id:
            q = q.where(Turn.session_id == session_id)
        rows = s.exec(q.order_by(Turn.created_at.desc()).limit(max_turns)).all()
    rows = list(reversed(rows))
    lines = []
    for t in rows:
        if t.transcript:
            lines.append(f"Tu: {t.transcript}")
        if t.tutor_reply:
            lines.append(f"Marco: {t.tutor_reply}")
    return "\n".join(lines)


def card_exclude_set() -> set[str]:
    """Normalized Italian words to keep OUT of card generation: everything the
    learner already knows (uploaded) or already has a card for (exported)."""
    with db.get_session() as s:
        rows = s.exec(select(Vocab)).all()
        out: set[str] = set()
        for v in rows:
            if v.source == "upload" or v.exported_at is not None or v.anki_note_id is not None:
                out.add(_norm(v.italian))
    return {w for w in out if w}


def pending_cards() -> list[Vocab]:
    """Cards generated but not yet pushed to Anki (retry-without-regeneration)."""
    with db.get_session() as s:
        rows = s.exec(
            select(Vocab)
            .where(Vocab.card_ready == True)  # noqa: E712
            .where(Vocab.exported_at.is_(None))
            .order_by(Vocab.created_at.asc())
        ).all()
        for r in rows:
            s.expunge(r)
        return rows


def reset_conversation_vocab() -> int:
    """One-time cleanup: drop auto-harvested conversation vocab (keeps uploaded
    known-words). Returns how many rows were removed."""
    with db.get_session() as s:
        rows = s.exec(select(Vocab).where(Vocab.source == "conversation")).all()
        n = len(rows)
        for r in rows:
            s.delete(r)
        s.commit()
    return n


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

        # Note: small-model new_vocab is NOT persisted here — it's display-only
        # (live chips). Card-worthy vocab is curated from the full transcript by
        # the big model at session end (see cards.curate_and_generate).

        s.commit()

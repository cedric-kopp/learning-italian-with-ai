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
from pathlib import Path

from sqlmodel import select

from . import db
from .config import (
    LLM,
    NATIVE,
    ROOT,
    SAVE_TRANSCRIPTS,
    TARGET,
    TRANSCRIPTS_DIR,
    TUTOR_NAME,
)
from .db import Mistake, Profile, Turn, Vocab
from .llm import TurnResult

KNOWN_CSV = ROOT / "known_vocab.csv"
_LEGACY_KNOWN_CSV = ROOT / "italian_known_vocab.csv"  # migrated on first use


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# Target-language articles, split for elision handling (l' / un').
_ELISION_ARTICLES = tuple(a for a in TARGET.articles if a.endswith("'"))
_WORD_ARTICLES = {a for a in TARGET.articles if not a.endswith("'")}


def _norm(word: str) -> str:
    """Normalize a target-language word for dedup: lowercase, strip punctuation, and
    drop a leading article so 'la casa' == 'casa', 'Piacere' == 'piacere'."""
    w = (word or "").strip().lower()
    w = re.sub(r"^[\"'“”«»\-\s]+|[\"'“”«».,;:!?]+$", "", w)
    for a in _ELISION_ARTICLES:  # attached apostrophe articles
        if w.startswith(a):
            w = w[len(a):].lstrip()
            break
    parts = w.split()
    if len(parts) > 1 and parts[0] in _WORD_ARTICLES:
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
        words = ", ".join(v.term for v in known if v.term)
        lines.append(f"- Words the learner ALREADY KNOWS (do not re-teach these): {words}")

    if recent:
        words = ", ".join(v.term for v in recent if v.term)
        lines.append(f"- Recently introduced in conversation: {words}")

    tags = Counter(m.tag for m in mistakes if m.tag)
    if tags:
        top = ", ".join(f"{t} (x{n})" for t, n in tags.most_common(max_tags))
        lines.append(f"- Recurring mistakes to watch for and reinforce: {top}")

    return "\n".join(lines)


# Split on newlines, commas, semicolons, or bullet characters.
_VOCAB_SPLIT = re.compile(r"[\n,;••]+")


def import_known_vocab(text: str) -> dict:
    """Ingest a target-language-only list of words the learner already knows.

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
        existing = {_norm(v.term): v for v in s.exec(select(Vocab)).all()}
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
            new_row = Vocab(term=word, source="upload", card_ready=False, due_at=_now())
            s.add(new_row)
            existing[key] = new_row
            added += 1
        s.commit()
    if added or skipped:
        export_known_vocab_csv()  # keep the committed file in sync with the DB
    return {"added": added, "skipped": skipped}


def export_known_vocab_csv() -> int:
    """Mirror the DB's known list (source='upload') to known_vocab.csv so the
    committed file always reflects the source of truth. Called on every change."""
    with db.get_session() as s:
        words = sorted(
            (v.term for v in s.exec(select(Vocab).where(Vocab.source == "upload")).all()),
            key=str.lower,
        )
    with open(KNOWN_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term"])
        for word in words:
            w.writerow([word])
    # Retire the legacy Italian-specific filename once migrated.
    if _LEGACY_KNOWN_CSV.exists() and _LEGACY_KNOWN_CSV != KNOWN_CSV:
        try:
            _LEGACY_KNOWN_CSV.unlink()
        except OSError:
            pass
    return len(words)


def ensure_known_csv() -> None:
    """The known-vocab file is gitignored (personal, per-instance), so recreate it
    from the DB on first run of a fresh clone. Migrates the legacy filename too."""
    if KNOWN_CSV.exists():
        return
    if _LEGACY_KNOWN_CSV.exists():  # older clone had the Italian-named file
        _LEGACY_KNOWN_CSV.replace(KNOWN_CSV)
        return
    export_known_vocab_csv()  # writes header (+ any words already in the DB)


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
    """Reconstruct a session's conversation as 'User: … / <Tutor>: …' lines for the
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
            lines.append(f"User: {t.transcript}")
        if t.tutor_reply:
            lines.append(f"{TUTOR_NAME}: {t.tutor_reply}")
    return "\n".join(lines)


def card_exclude_set() -> set[str]:
    """Normalized target-language words to keep OUT of card generation: everything
    the learner already knows (uploaded) or already has a card for (exported)."""
    with db.get_session() as s:
        rows = s.exec(select(Vocab)).all()
        out: set[str] = set()
        for v in rows:
            if v.source == "upload" or v.exported_at is not None or v.anki_note_id is not None:
                out.add(_norm(v.term))
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
    mode: str = "",
) -> None:
    """Save the turn and harvest vocab + mistakes into long-term memory."""
    with db.get_session() as s:
        s.add(
            Turn(
                session_id=session_id,
                mode=mode,
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

    # Mirror the full session to a gitignored log file for later quality review.
    # Rewritten every turn so it always reflects the whole sitting, no matter how
    # the session ends (tab close, refresh, etc.). Best-effort — never break a turn.
    if session_id:
        try:
            save_session_log(session_id)
        except Exception as e:  # pragma: no cover - logging must not break a turn
            print(f"[transcript] save failed for session {session_id}: {e}", flush=True)


def _fmt_dt(value: dt.datetime | None) -> str:
    """Format a stored datetime as 'YYYY-MM-DD HH:MM:SS UTC' (SQLite may hand back
    naive datetimes; treat those as UTC)."""
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


_MODE_LABEL = {
    "immersion": "immersion",
    "english_teacher": "teacher",
    "scenario": "role-play",
}


def save_session_log(session_id: int) -> Path | None:
    """Write a session's full conversation to transcripts/ as both a readable
    Markdown log and a structured JSON dump (for future automated evaluation).

    Returns the Markdown path, or None if disabled / the session has no turns.
    """
    if not SAVE_TRANSCRIPTS or not session_id:
        return None

    with db.get_session() as s:
        turns = s.exec(
            select(Turn).where(Turn.session_id == session_id).order_by(Turn.created_at.asc())
        ).all()
    if not turns:
        return None

    started = turns[0].created_at
    if started is not None and started.tzinfo is None:
        started = started.replace(tzinfo=dt.timezone.utc)
    stamp = (started or _now()).astimezone(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    base = f"{stamp}_session-{session_id}"

    # --- structured JSON (for programmatic review / eval) ---
    turn_records = []
    for i, t in enumerate(turns, 1):
        try:
            corrections = json.loads(t.corrections_json or "[]")
        except (ValueError, TypeError):
            corrections = []
        turn_records.append(
            {
                "n": i,
                "at": _fmt_dt(t.created_at),
                "mode": t.mode or "",
                "stt_confidence": t.stt_confidence,
                "learner": t.transcript,
                "tutor": t.tutor_reply,
                "corrections": corrections,
            }
        )

    meta = {
        "session_id": session_id,
        "started_at": _fmt_dt(started),
        "saved_at": _fmt_dt(_now()),
        "target_lang": {"code": TARGET.code, "name": TARGET.name_en},
        "native_lang": {"code": NATIVE.code, "name": NATIVE.name_en},
        "tutor_name": TUTOR_NAME,
        "backend": LLM.backend,
        "model": LLM.model,
        "num_turns": len(turns),
    }

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS_DIR / f"{base}.json").write_text(
        json.dumps({**meta, "turns": turn_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- human-readable Markdown ---
    md = [
        f"# {TUTOR_NAME} session log",
        "",
        f"- **Session:** {session_id}",
        f"- **Started:** {meta['started_at']}",
        f"- **Language:** {TARGET.name_en} (explanations in {NATIVE.name_en})",
        f"- **Tutor:** {TUTOR_NAME}",
        f"- **Backend:** {LLM.backend} · {LLM.model}",
        f"- **Turns:** {len(turns)}",
        "",
        "---",
        "",
    ]
    for rec in turn_records:
        head = f"### Turn {rec['n']} · {rec['at']}"
        label = _MODE_LABEL.get(rec["mode"], rec["mode"])
        if label:
            head += f" · {label}"
        if rec["stt_confidence"] is not None:
            head += f" · clarity ~{round(rec['stt_confidence'] * 100)}%"
        md.append(head)
        md.append("")
        md.append(f"**You:** {rec['learner'] or '_(nothing heard)_'}")
        md.append("")
        md.append(f"**{TUTOR_NAME}:** {rec['tutor']}")
        if rec["corrections"]:
            md.append("")
            md.append("_Corrections:_")
            for c in rec["corrections"]:
                was = c.get("your_text") or ""
                fix = c.get("correction") or ""
                why = c.get("explanation") or ""
                tag = c.get("tag") or ""
                line = "- "
                if was:
                    line += f"~~{was}~~ → "
                line += f"**{fix}**"
                if why:
                    line += f" — {why}"
                if tag:
                    line += f" `[{tag}]`"
                md.append(line)
        md.append("")
        md.append("---")
        md.append("")

    md_path = TRANSCRIPTS_DIR / f"{base}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path

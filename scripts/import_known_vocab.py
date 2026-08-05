"""One-off: clean a raw (OCR-scraped) Italian vocab CSV into the known-vocab list.

Reads italian_vocab.csv (columns: #, Italian, German), cleans the Italian column
with qwen3:14b (fix spelling/accents, un-merge OCR-joined words, lemmatize, drop
function words / grammar terms / proper nouns, keep useful set phrases), dedups,
then (a) replaces the source="upload" rows in tutore.db and (b) writes a
versionable italian_known_vocab.csv.

Usage:  uv run python scripts/import_known_vocab.py [path/to/input.csv]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from openai import OpenAI
from sqlmodel import delete, select

from tutore import db
from tutore.cards import _STOP
from tutore.config import CARD_MODEL, LLM
from tutore.memory import _norm

ROOT = Path(__file__).resolve().parent.parent
INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "italian_vocab.csv"
OUTPUT_CSV = ROOT / "italian_known_vocab.csv"

client = OpenAI(base_url=LLM.base_url, api_key=LLM.api_key)

# Backstop: an item is junk if EVERY token is a function word / auxiliary form
# (e.g. the un-merged "e tu", "e di", "sono di" that the model keeps).
_FUNCTION_TOKENS = {
    # articles / prepositions / conjunctions
    "il", "lo", "la", "i", "gli", "le", "l'", "un", "uno", "una", "un'",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "del", "della",
    "dello", "dei", "degli", "delle", "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "nel", "nella", "sul", "e", "ed", "o", "od", "ma", "se", "che", "né",
    # pronouns
    "io", "tu", "lui", "lei", "noi", "voi", "loro", "mi", "ti", "ci", "vi", "si",
    "me", "te", "lo", "gli", "ne", "chi", "cui",
    # bare auxiliary conjugations (essere/avere)
    "sono", "sei", "è", "siamo", "siete", "ero", "era", "essere",
    "ho", "hai", "ha", "abbiamo", "avete", "hanno", "avere",
}


def _all_function(word: str) -> bool:
    toks = [t for t in word.lower().replace("'", "' ").split() if t]
    return bool(toks) and all(t in _FUNCTION_TOKENS or t.rstrip("'") in _FUNCTION_TOKENS for t in toks)

_SYSTEM = """You clean a raw Italian vocabulary list scraped (with OCR errors) from a
Duolingo screen recording. The learner is a German native speaker. For EACH input
Italian string, output the cleaned canonical vocabulary item, or null to DROP it.

CLEAN:
- Fix spelling and missing accents: caffe→caffè, te→tè, li→lì, perche→perché.
- Un-merge words joined by OCR: sonodi→"sono di", etu→"e tu", apresto→"a presto".
- Dictionary form: infinitive for verbs; singular for nouns; adjectives to
  MASCULINE singular (timida→timido, allegra→allegro). Keep a noun's own gender
  (ragazza stays ragazza).

DROP (output null):
- Pure function words: articles, prepositions, conjunctions, pronouns, and the bare
  auxiliaries essere/avere and their conjugations (un, e, o, io, tu, lui, ho,
  "e di", "e tu", "io sono").
- Interjections/fillers (ah, eh, mah), grammar meta-terms, and proper nouns/names.

KEEP: content nouns/verbs/adjectives/adverbs, AND useful fixed expressions and
greetings (per favore, come va, a presto, buongiorno, buonasera, arrivederci).

Return ONLY JSON: {"items":[{"in":"<original>","out":"<cleaned>" or null}]} — same
length and order as the input. No markdown."""


def clean_batch(words: list[str]) -> list[str | None]:
    resp = client.chat.completions.create(
        model=CARD_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(words, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=3000,
        response_format={"type": "json_object"},
        **({"reasoning_effort": "none"} if "qwen3" in CARD_MODEL.lower() else {}),
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    items = data.get("items", [])
    out: list[str | None] = []
    for i in range(len(words)):
        if i < len(items) and isinstance(items[i], dict):
            val = items[i].get("out")
            out.append(val.strip() if isinstance(val, str) and val.strip() else None)
        else:
            out.append(None)
    return out


def main() -> None:
    # 1) Read Italian column.
    raw: list[str] = []
    with open(INPUT, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = (row.get("Italian") or "").strip()
            if w:
                raw.append(w)
    print(f"read {len(raw)} raw entries from {INPUT.name}")

    # 2) Clean in batches.
    cleaned: list[str] = []
    dropped = 0
    B = 40
    for i in range(0, len(raw), B):
        batch = raw[i : i + B]
        for orig, out in zip(batch, clean_batch(batch)):
            if out is None:
                dropped += 1
                continue
            cleaned.append(out)
        print(f"  cleaned {min(i + B, len(raw))}/{len(raw)} (kept {len(cleaned)}, dropped {dropped})")

    # 3) Dedup (normalized) + stoplist backstop.
    final: dict[str, str] = {}  # norm -> display form
    for w in cleaned:
        key = _norm(w)
        if not key or key in _STOP or key in final or _all_function(w):
            continue
        final[key] = w
    words = sorted(final.values(), key=str.lower)
    print(f"final known-vocab: {len(words)} unique (from {len(raw)} raw)")

    # 4) Replace source="upload" rows in the DB.
    db.init_db()
    with db.get_session() as s:
        s.exec(delete(db.Vocab).where(db.Vocab.source == "upload"))
        for w in words:
            s.add(db.Vocab(italian=w, source="upload", card_ready=False))
        s.commit()
        n = len(s.exec(select(db.Vocab).where(db.Vocab.source == "upload")).all())
    print(f"DB: {n} known (upload) rows written to {db.DB_PATH}")

    # 5) Versionable flat file.
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["italian"])
        for w in words:
            writer.writerow([w])
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

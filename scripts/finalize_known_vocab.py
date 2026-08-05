"""Finalize the known-vocab list: collapse inflected forms to citation form.

The mappings below are hand-decided (verb conjugations -> infinitive, plurals ->
singular, one interjection dropped), keeping distinct gender pairs and lexemes.
Deterministic — no model call. Rewrites the DB (source="upload") and the CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlmodel import delete, select

from tutore import db
from tutore.memory import _norm

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "italian_known_vocab.csv"

# Inflected forms / junk to drop (their citation form is added below, or already present).
REMOVE = {
    "arance",           # -> arancia (present)
    "devi", "devo",     # -> dovere
    "ha bisogno di", "ho bisogno di",  # -> avere bisogno di
    "molte", "molti",   # -> molto (present)
    "preferiamo", "preferisco",  # -> preferire (preferito stays, distinct)
    "ragazze",          # -> ragazza (present)
    "ragazzi",          # -> ragazzo (present)
    "vada",             # -> andare (present)
    "vestiti",          # -> vestito (present)
    "vogliamo", "voglio", "vogliono", "volete", "vorrei", "vuoi", "vuole",  # -> volere
    "guarda",           # -> guardare
    "sta piovendo",     # -> piovere
    "stanno bene",      # -> stare bene
    "non trovo",        # -> trovare
    "dove sono",        # keep dov'è (present)
    "wow",              # interjection
}

# Citation forms to add.
ADD = [
    "dovere", "avere bisogno di", "preferire", "volere",
    "guardare", "piovere", "stare bene", "trovare",
]


def main() -> None:
    words = [r["italian"] for r in csv.DictReader(open(CSV_PATH, encoding="utf-8"))]
    kept = [w for w in words if _norm(w) not in {_norm(r) for r in REMOVE}]

    final: dict[str, str] = {}
    for w in kept + ADD:
        key = _norm(w)
        if key and key not in final:
            final[key] = w
    result = sorted(final.values(), key=str.lower)
    print(f"{len(words)} -> {len(result)} "
          f"(removed {len(words) - len(kept)} inflected, added {len(ADD)})")

    db.init_db()
    with db.get_session() as s:
        s.exec(delete(db.Vocab).where(db.Vocab.source == "upload"))
        for w in result:
            s.add(db.Vocab(italian=w, source="upload", card_ready=False))
        s.commit()
        n = len(s.exec(select(db.Vocab).where(db.Vocab.source == "upload")).all())
    print(f"DB: {n} known (upload) rows")

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["italian"])
        for w in result:
            writer.writerow([w])
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()

"""AnkiConnect client — live push into the user's Anki desktop.

AnkiConnect (an Anki add-on) runs a JSON HTTP server on 127.0.0.1:8765 whenever
Anki desktop is open. We POST {action, version, params} and get {result, error}.
No third-party dependency needed — just urllib.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import ANKI_CONNECT_URL, ANKI_DECK, ANKI_TAGS

_VERSION = 6


class AnkiUnavailable(RuntimeError):
    """Anki desktop / AnkiConnect is not reachable."""


def _invoke(action: str, timeout: float = 5.0, **params):
    body = json.dumps({"action": action, "version": _VERSION, "params": params}).encode("utf-8")
    req = urllib.request.Request(ANKI_CONNECT_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise AnkiUnavailable(str(e)) from e
    if data.get("error") is not None:
        raise RuntimeError(f"AnkiConnect error on {action}: {data['error']}")
    return data.get("result")


def is_available() -> bool:
    try:
        _invoke("version", timeout=2.0)
        return True
    except Exception:
        return False


def deck_names() -> list[str]:
    return list(_invoke("deckNames") or [])


def ensure_deck(name: str) -> None:
    _invoke("createDeck", deck=name)  # idempotent — no-op if it already exists


@dataclass
class PushResult:
    pushed: int
    skipped: int
    note_ids: list  # aligned with input cards; None where not added (e.g. duplicate)


def add_cards(cards: list, deck: str | None = None, tags: list[str] | None = None) -> PushResult:
    """Push Cards (objects exposing .front / .back) into a deck via addNotes.

    addNotes returns a note id per note, or null for ones it couldn't add
    (typically duplicates). Uses Basic model (Front/Back)."""
    if not cards:
        return PushResult(0, 0, [])
    deck = deck or ANKI_DECK
    tags = tags if tags is not None else ANKI_TAGS
    ensure_deck(deck)

    notes = [
        {
            "deckName": deck,
            "modelName": "Basic",
            "fields": {"Front": c.front, "Back": c.back},
            "tags": list(tags),
            "options": {"allowDuplicate": False, "duplicateScope": "deck"},
        }
        for c in cards
    ]
    ids = _invoke("addNotes", notes=notes) or [None] * len(cards)
    pushed = sum(1 for i in ids if i)
    return PushResult(pushed=pushed, skipped=len(cards) - pushed, note_ids=list(ids))

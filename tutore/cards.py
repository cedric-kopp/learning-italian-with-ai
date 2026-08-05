"""Batch flashcard generation — the session-end handoff to a larger model.

gemma3:4b keeps the conversation fast; here we spend qwen3:14b (CARD_MODEL) to
turn harvested Italian words into proper German-backed Anki cards. Latency is
irrelevant (runs once when the user ends a session), so we buy quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

from .config import CARD_MODEL, LLM
from .llm import _demarkdown, _extract_json

_ARTICLES = {"il", "lo", "la", "i", "gli", "le", "un", "una", "uno"}


def _strip_article(word: str) -> str:
    w = re.sub(r"^(l'|un')\s*", "", word.strip().lower())
    parts = w.split()
    if len(parts) > 1 and parts[0] in _ARTICLES:
        parts = parts[1:]
    return " ".join(parts).strip()

# Same local server as chat, different (bigger) model.
_client = OpenAI(base_url=LLM.base_url, api_key=LLM.api_key)


@dataclass
class Card:
    italian: str
    german: str = ""
    article: str = ""  # il/la/lo/l' for nouns, "" otherwise
    pos: str = ""  # noun/verb/adjective/adverb/phrase/other
    example_it: str = ""

    @property
    def front(self) -> str:
        """Anki front: Italian, with article for nouns so gender is learned."""
        if self.article and self.pos == "noun":
            sep = "" if self.article.endswith("'") else " "  # l'azione, not l' azione
            return f"{self.article}{sep}{self.italian}"
        return self.italian

    @property
    def back(self) -> str:
        """Anki back: German translation + POS tag + Italian example."""
        parts = [self.german]
        if self.pos:
            parts.append(f"({self.pos})")
        line = " ".join(p for p in parts if p)
        if self.example_it:
            line += f"\n\nes. {self.example_it}"
        return line


# --- Step 1: candidate extraction (deterministic, via spaCy Italian) ----------
# spaCy gives us reliable lemmatization + POS, so proper nouns (PROPN), function
# words (DET/ADP/PRON/AUX/…), and inflected forms are handled grammatically rather
# than by a fragile LLM prompt. A small stoplist still drops grammar meta-terms
# (tagged NOUN) and trivial greetings.
_UPOS_TO_POS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}
_STOP = {
    # grammar / linguistic meta-terminology (spaCy tags these as NOUN)
    "coniugazione", "verbo", "aggettivo", "sostantivo", "avverbio", "articolo",
    "preposizione", "pronome", "congiunzione", "participio", "gerundio", "infinito",
    "congiuntivo", "indicativo", "condizionale", "imperativo", "presente", "passato",
    "prossimo", "imperfetto", "futuro", "trapassato", "singolare", "plurale",
    "maschile", "femminile", "genere", "frase", "parola", "esempio", "grammatica",
    # bare auxiliaries + trivial greetings/politeness
    "essere", "avere", "ciao", "salve", "grazie", "prego", "buongiorno", "buonasera",
    "buonanotte", "arrivederci", "sì",
    # grammatical particles/adverbs spaCy tags as ADV/PART -> would leak as "cards"
    "non", "più", "meno", "ci", "ne", "già", "mai", "ancora", "appena", "anche",
    "poi", "qui", "qua", "lì", "là", "ecco", "molto", "poco", "tanto", "troppo",
}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        # Tagger + morphologizer give POS/lemma; parser + NER not needed → faster.
        _nlp = spacy.load("it_core_news_sm", disable=["parser", "ner"])
    return _nlp


def extract_candidates(transcript: str, exclude: set[str]) -> list[tuple[str, str]]:
    """Return [(surface, pos)] of card-worthy content-word tokens from the
    transcript. spaCy handles POS + proper-noun/function-word filtering; the LLM
    does the actual lemmatization (it's better on Italian clitics than sm)."""
    doc = _get_nlp()(transcript)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for tok in doc:
        pos = _UPOS_TO_POS.get(tok.pos_)
        if pos is None:  # keeps only NOUN/VERB/ADJ/ADV; drops PROPN, AUX, DET, …
            continue
        lemma = (tok.lemma_ or "").strip().lower()  # used only for dedup + stop check
        surface = tok.text.strip()
        if not surface or not lemma or len(lemma) < 2:
            continue
        if lemma in _STOP or lemma in exclude or lemma in seen:
            continue
        if not surface.replace("'", "").isalpha():
            continue
        seen.add(lemma)
        out.append((surface, pos))
    return out


# --- Step 2: lemmatize + enrich (LLM's strengths) -----------------------------
_IT_ARTICLES = {"il", "lo", "la", "l'", "i", "gli", "le"}
_MODEL_POS = {"noun", "verb", "adjective", "adverb"}

_ENRICH_SYSTEM = """You are an Italian→German lexicographer making Anki cards for a
German native speaker (level A2). You receive Italian words AS THEY APPEARED in a
conversation (possibly inflected, or a verb with an attached pronoun), each with a
rough part-of-speech HINT that may be WRONG. For EACH input, echo the word and
return its card. If the word is a conjugated verb, give the INFINITIVE even when
the hint says adjective/noun.
{
  "cards": [
    {
      "word": "<the EXACT input word, echoed verbatim so it can be matched back>",
      "italian": "<DICTIONARY form: infinitive for verbs (drop attached pronouns), MASCULINE SINGULAR for adjectives, SINGULAR for nouns>",
      "pos": "noun|verb|adjective|adverb",
      "it_article": "<the ITALIAN definite singular article for NOUNS only: il, lo, la, or l' — NOT a German article; empty '' for non-nouns>",
      "german": "<German translation; for nouns include the German article, e.g. 'das Haus'>",
      "example_it": "<one short, natural A2 Italian example sentence using the word>"
    }
  ]
}
Include EVERY input word exactly once. Return ONLY that JSON. No markdown."""


def _enrich(candidates: list[tuple[str, str]]) -> list[Card]:
    import json

    payload = [{"word": w, "pos_hint": p} for w, p in candidates]
    kwargs = dict(
        model=CARD_MODEL,
        messages=[
            {"role": "system", "content": _ENRICH_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_tokens=2500,
    )
    if "qwen3" in CARD_MODEL.lower():
        kwargs["reasoning_effort"] = "none"
    try:
        resp = _client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        kwargs.pop("reasoning_effort", None)
        resp = _client.chat.completions.create(**kwargs)

    data = _extract_json(resp.choices[0].message.content or "") or {}
    enriched = data.get("cards", []) if isinstance(data, dict) else []

    # Map results back BY ECHOED WORD (not index) — the model drops/reorders items
    # in large batches, and index-alignment would otherwise emit raw surface forms.
    by_word: dict[str, dict] = {}
    for e in enriched:
        if isinstance(e, dict):
            key = _strip_article(str(e.get("word", "")))
            if key:
                by_word.setdefault(key, e)

    cards: list[Card] = []
    for surface, spacy_pos in candidates:
        e = by_word.get(_strip_article(surface))
        if not e:
            continue  # no enrichment -> DROP (never fall back to the raw surface)
        italian = _demarkdown(e.get("italian", "")).lower()
        if not italian:
            continue
        pos = _demarkdown(e.get("pos", "")).lower()
        if pos not in _MODEL_POS:
            pos = spacy_pos  # fall back to spaCy's tag if the model's is unusable
        # Keep the article only if it's a real Italian article (guards against the
        # model putting a German article like "der" here).
        article = _demarkdown(e.get("it_article", "")).lower()
        if article not in _IT_ARTICLES or pos != "noun":
            article = ""
        cards.append(
            Card(
                italian=italian,
                german=_demarkdown(e.get("german", "")),
                article=article,
                pos=pos,
                example_it=_demarkdown(e.get("example_it", "")),
            )
        )
    return cards


def curate_and_generate(transcript: str, exclude: set[str] | None = None, cap: int = 20) -> list[Card]:
    """spaCy extracts clean content-word tokens from the transcript (dropping proper
    nouns / function words grammatically); the LLM lemmatizes + enriches them into
    German-backed cards. Final dedup honors the exclude set and the LLM's lemmas."""
    transcript = (transcript or "").strip()
    if not transcript:
        return []
    exclude = exclude or set()
    candidates = extract_candidates(transcript, exclude)[:cap]
    if not candidates:
        return []

    seen: set[str] = set()
    out: list[Card] = []
    for c in _enrich(candidates):
        key = _strip_article(c.italian)
        if not c.german or not key or key in _STOP or key in exclude or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

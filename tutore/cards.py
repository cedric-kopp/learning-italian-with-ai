"""Batch flashcard generation — the session-end handoff to a larger model.

The fast chat model keeps conversation snappy; here we spend CARD_MODEL (qwen3:14b)
to turn harvested TARGET-language words into proper NATIVE-language-backed Anki
cards. Latency is irrelevant (runs once at session end), so we buy quality. All
language-specific facts come from the registry via config.TARGET / config.NATIVE.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .config import CARD_MODEL, LLM, NATIVE, TARGET, TUTOR_NAME
from .llm import _demarkdown, _extract_json

# Articles for the target language, split into attached-elision ("l'") vs standalone.
_ELISION_ARTICLES = tuple(a for a in TARGET.articles if a.endswith("'"))
_WORD_ARTICLES = {a for a in TARGET.articles if not a.endswith("'")}
_TARGET_ARTICLES = set(TARGET.articles)
# Registry stopwords + the transcript speaker labels (so "User:" / the tutor name
# never become flashcards).
_STOP = set(TARGET.stopwords) | {"user", TUTOR_NAME.lower()}


def _strip_article(word: str) -> str:
    """Drop a leading article so 'la casa' == 'casa' for dedup/matching."""
    w = word.strip().lower()
    for a in _ELISION_ARTICLES:
        if w.startswith(a):
            w = w[len(a):].lstrip()
            break
    parts = w.split()
    if len(parts) > 1 and parts[0] in _WORD_ARTICLES:
        parts = parts[1:]
    return " ".join(parts).strip()


# Same local server as chat, different (bigger) model.
_client = OpenAI(base_url=LLM.base_url, api_key=LLM.api_key)


@dataclass
class Card:
    term: str  # dictionary form in the TARGET language
    translation: str = ""  # NATIVE-language translation (card back)
    article: str = ""  # target definite article for nouns, "" otherwise
    pos: str = ""  # noun/verb/adjective/adverb
    example: str = ""  # short target-language example sentence

    @property
    def front(self) -> str:
        """Anki front: target word, with article for nouns so gender is learned."""
        if self.article and self.pos == "noun":
            sep = "" if self.article.endswith("'") else " "  # l'azione, not l' azione
            return f"{self.article}{sep}{self.term}"
        return self.term

    @property
    def back(self) -> str:
        """Anki back: native translation + POS tag + a target-language example."""
        parts = [self.translation]
        if self.pos:
            parts.append(f"({self.pos})")
        line = " ".join(p for p in parts if p)
        if self.example:
            line += f"\n\nex: {self.example}"
        return line


# --- Step 1: candidate extraction (deterministic, via spaCy) ------------------
# spaCy gives reliable POS, so proper nouns (PROPN), function words (DET/ADP/PRON/
# AUX/…) and inflected forms are handled grammatically, not by a fragile prompt.
# A per-language stoplist still drops grammar meta-terms (tagged NOUN) + greetings.
_UPOS_TO_POS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        # Tagger + morphologizer give POS/lemma; parser + NER not needed → faster.
        _nlp = spacy.load(TARGET.spacy_model, disable=["parser", "ner"])
    return _nlp


def extract_candidates(transcript: str, exclude: set[str]) -> list[tuple[str, str]]:
    """Return [(surface, pos)] of card-worthy content-word tokens. spaCy handles
    POS + proper-noun/function-word filtering; the LLM does the lemmatization."""
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
_MODEL_POS = {"noun", "verb", "adjective", "adverb"}

_ENRICH_SYSTEM = f"""You are a {TARGET.name_en}→{NATIVE.name_en} lexicographer making
Anki cards for a {NATIVE.name_en} speaker (level A2). You receive {TARGET.name_en}
words AS THEY APPEARED in a conversation (possibly inflected, or a verb with an
attached pronoun), each with a rough part-of-speech HINT that may be WRONG. For EACH
input, echo the word and return its card. If the word is a conjugated verb, give the
INFINITIVE even when the hint says adjective/noun.
{{
  "cards": [
    {{
      "word": "<the EXACT input word, echoed verbatim so it can be matched back>",
      "term": "<DICTIONARY form: infinitive for verbs (drop attached pronouns), MASCULINE SINGULAR for adjectives, SINGULAR for nouns>",
      "pos": "noun|verb|adjective|adverb",
      "article": "<the {TARGET.name_en} definite singular article for NOUNS only (in {TARGET.name_en}, NOT {NATIVE.name_en}); empty '' for non-nouns>",
      "translation": "<{NATIVE.name_en} translation; for nouns include the {NATIVE.name_en} article if it has one>",
      "example": "<one short, natural A2 {TARGET.name_en} example sentence using the word>"
    }}
  ]
}}
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
        term = _demarkdown(e.get("term", "")).lower()
        if not term:
            continue
        pos = _demarkdown(e.get("pos", "")).lower()
        if pos not in _MODEL_POS:
            pos = spacy_pos  # fall back to spaCy's tag if the model's is unusable
        # Keep the article only if it's a real TARGET-language article (guards
        # against the model putting a native-language article here).
        article = _demarkdown(e.get("article", "")).lower()
        if article not in _TARGET_ARTICLES or pos != "noun":
            article = ""
        cards.append(
            Card(
                term=term,
                translation=_demarkdown(e.get("translation", "")),
                article=article,
                pos=pos,
                example=_demarkdown(e.get("example", "")),
            )
        )
    return cards


def curate_and_generate(transcript: str, exclude: set[str] | None = None, cap: int = 20) -> list[Card]:
    """spaCy extracts clean content-word tokens from the transcript (dropping proper
    nouns / function words grammatically); the LLM lemmatizes + enriches them into
    native-backed cards. Final dedup honors the exclude set and the LLM's lemmas."""
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
        key = _strip_article(c.term)
        if not c.translation or not key or key in _STOP or key in exclude or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

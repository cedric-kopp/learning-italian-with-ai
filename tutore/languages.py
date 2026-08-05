"""Language registry — every language-specific fact lives here.

Adding a language = one entry. The rest of the app reads target/native facts from
here via config.TARGET / config.NATIVE, so nothing else is hardcoded to a language.

Support tiers:
- Full: Latin script, non-tonal, has a Kokoro TTS voice + a spaCy model + good
  local-LLM support (es, fr, it, pt, en).
- Caveats: works but degraded — non-Latin script weakens the "Whisper transcript =
  pronunciation signal" idea, and G2P/spaCy support is thinner (ja).
- Unsupported here: tonal (zh) and languages with no Kokoro voice (de, nl, ru, …)
  — though any language can be the NATIVE one (card backs are text, never spoken).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    code: str  # ISO 639-1, e.g. "it"
    name_en: str  # English name, used in prompts + UI ("Italian")
    flag: str  # emoji
    whisper: str  # Whisper language code (usually == code)
    kokoro: str  # Kokoro pipeline lang_code (single letter)
    voice: str  # default Kokoro voice id
    spacy_model: str  # spaCy pipeline name for vocab extraction
    tutor_name: str  # default tutor persona name
    # Definite/indefinite articles, for dedup + article-stripping in card fronts.
    articles: tuple[str, ...] = ()
    # Grammar-term / particle backstop the spaCy POS filter can miss.
    stopwords: tuple[str, ...] = ()
    # A short "sorry, please repeat" line spoken when nothing intelligible is heard.
    repeat_prompt: str = "Sorry, could you repeat that?"
    # Non-empty => shown as a warning by the setup wizard.
    caveats: str = ""


# Italian keeps its tuned stopword/article lists so card quality doesn't regress.
_IT_STOP = (
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
)


LANGUAGES: dict[str, Language] = {
    "it": Language(
        code="it", name_en="Italian", flag="🇮🇹", whisper="it", kokoro="i",
        voice="im_nicola", spacy_model="it_core_news_sm", tutor_name="Marco",
        articles=("il", "lo", "la", "l'", "i", "gli", "le", "un", "uno", "una", "un'"),
        stopwords=_IT_STOP, repeat_prompt="Scusa, puoi ripetere?",
    ),
    "es": Language(
        code="es", name_en="Spanish", flag="🇪🇸", whisper="es", kokoro="e",
        voice="em_alex", spacy_model="es_core_news_sm", tutor_name="Sofía",
        articles=("el", "la", "los", "las", "un", "una", "unos", "unas"),
        repeat_prompt="Perdona, ¿puedes repetir?",
    ),
    "fr": Language(
        code="fr", name_en="French", flag="🇫🇷", whisper="fr", kokoro="f",
        voice="ff_siwis", spacy_model="fr_core_news_sm", tutor_name="Léa",
        articles=("le", "la", "les", "l'", "un", "une", "des"),
        repeat_prompt="Pardon, peux-tu répéter ?",
    ),
    "pt": Language(
        code="pt", name_en="Portuguese", flag="🇧🇷", whisper="pt", kokoro="p",
        voice="pm_alex", spacy_model="pt_core_news_sm", tutor_name="João",
        articles=("o", "a", "os", "as", "um", "uma", "uns", "umas"),
        repeat_prompt="Desculpa, podes repetir?",
    ),
    "en": Language(
        code="en", name_en="English", flag="🇬🇧", whisper="en", kokoro="a",
        voice="am_michael", spacy_model="en_core_web_sm", tutor_name="Alex",
        articles=("the", "a", "an"),
        repeat_prompt="Sorry, could you repeat that?",
    ),
    "ja": Language(
        code="ja", name_en="Japanese", flag="🇯🇵", whisper="ja", kokoro="j",
        voice="jf_alpha", spacy_model="ja_core_news_sm", tutor_name="Yuki",
        repeat_prompt="すみません、もう一度お願いします。",
        caveats="Non-Latin script: the pronunciation-via-transcript signal is "
        "weaker, and spaCy/G2P support is thinner. Needs the misaki[ja] G2P extra.",
    ),
}

DEFAULT_TARGET = "it"
DEFAULT_NATIVE = "en"


def get(code: str, fallback: str) -> Language:
    return LANGUAGES.get((code or "").strip().lower(), LANGUAGES[fallback])

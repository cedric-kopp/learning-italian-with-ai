"""Text-to-speech via Kokoro (multilingual, open-weight).

Kokoro uses a per-language pipeline keyed by a single-letter code (Italian 'i',
Spanish 'e', French 'f', Portuguese 'p', English 'a', Japanese 'j'). The language
facts + default voices come from the registry (config.TARGET / config.NATIVE);
`synthesize` receives an ISO code (the reply's language) and resolves the rest.
Latin languages need espeak-ng for G2P; Japanese needs misaki[ja].
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from . import languages
from .config import NATIVE, TARGET, TTS_VOICE_NATIVE, TTS_VOICE_TARGET

SAMPLE_RATE = 24000  # Kokoro's native output rate

_pipelines: dict[str, "object"] = {}


def _resolve(lang: str) -> tuple[str, str]:
    """Map an ISO language code to (Kokoro pipeline code, voice id)."""
    code = (lang or TARGET.code).lower()
    if code == TARGET.code:
        return TARGET.kokoro, TTS_VOICE_TARGET
    if code == NATIVE.code:
        return NATIVE.kokoro, TTS_VOICE_NATIVE
    L = languages.get(code, TARGET.code)  # fallback for any other code
    return L.kokoro, L.voice


def _get_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline  # imported lazily (pulls torch)

        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def synthesize(text: str, lang: str | None = None, speed: float = 0.9) -> bytes:
    """Render `text` to WAV bytes. `lang` is an ISO code (defaults to target).
    `speed` 1.0 = natural, <1 slower."""
    text = (text or "").strip()
    if not text:
        return _wav_bytes(np.zeros(1, dtype=np.float32))

    kokoro_code, voice = _resolve(lang or TARGET.code)
    pipeline = _get_pipeline(kokoro_code)

    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=voice, speed=float(speed)):
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        return _wav_bytes(np.zeros(1, dtype=np.float32))
    return _wav_bytes(np.concatenate(chunks))


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()

"""Text-to-speech via Kokoro (multilingual, open-weight).

Kokoro uses a per-language pipeline keyed by a single-letter lang code:
  'i' = Italian, 'a' = American English, 'b' = British English.
Italian needs espeak-ng installed (brew install espeak-ng) for G2P.
Pipelines are cached because construction loads the voice + G2P backend.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from .config import TTS_VOICE_EN, TTS_VOICE_IT

SAMPLE_RATE = 24000  # Kokoro's native output rate

_pipelines: dict[str, "object"] = {}


def _lang_code(lang: str) -> str:
    return "i" if lang.lower().startswith("it") else "a"


def _voice_for(lang: str) -> str:
    return TTS_VOICE_IT if lang.lower().startswith("it") else TTS_VOICE_EN


def _get_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline  # imported lazily (pulls torch)

        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def synthesize(text: str, lang: str = "it", speed: float = 0.9) -> bytes:
    """Render `text` to WAV bytes. `speed` 1.0 = natural, <1 slower."""
    text = (text or "").strip()
    if not text:
        return _wav_bytes(np.zeros(1, dtype=np.float32))

    pipeline = _get_pipeline(_lang_code(lang))
    voice = _voice_for(lang)

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

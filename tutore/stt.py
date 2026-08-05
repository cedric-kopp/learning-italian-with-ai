"""Speech-to-text via mlx-whisper (Apple Silicon).

Transcribes the learner's Italian speech. The transcript is doubly useful: it's
the LLM's input *and* an honest pronunciation signal — if Whisper mishears you,
an Italian probably would too. `avg_logprob` is exposed as a rough confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx_whisper

from .config import WHISPER_MODEL


@dataclass
class Transcription:
    text: str
    confidence: float  # 0..1, derived from avg logprob (higher = clearer speech)
    language: str


def transcribe(audio_path: str, language: str | None = "it") -> Transcription:
    """Transcribe an audio file. `audio_path` is decoded by ffmpeg internally.
    Pass language=None to let Whisper auto-detect (used for English-teacher mode)."""
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        language=language,
        # Slightly conservative decoding — steadier for learner speech.
        temperature=0.0,
        condition_on_previous_text=False,
    )

    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    confidence = _confidence_from_segments(segments)
    return Transcription(
        text=text,
        confidence=confidence,
        language=result.get("language", language),
    )


def _confidence_from_segments(segments: list[dict]) -> float:
    """Map mean segment avg_logprob (~ -1..0) to a 0..1 confidence."""
    if not segments:
        return 0.0
    logprobs = [s.get("avg_logprob", -1.0) for s in segments if "avg_logprob" in s]
    if not logprobs:
        return 0.0
    mean = sum(logprobs) / len(logprobs)
    # sigmoid-ish squash: -1.0 -> ~0.27, 0.0 -> ~0.73; clamp to [0,1].
    conf = 1.0 / (1.0 + math.exp(-(mean + 0.5) * 2.0))
    return max(0.0, min(1.0, conf))

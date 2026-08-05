"""LLM client — one interface, two interchangeable local engines.

Ollama and mlx_lm.server both speak the OpenAI Chat Completions API, so swapping
`LLM_BACKEND` changes only the base URL + model id (see config.py). Everything
below is backend-agnostic.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import LLM, NATIVE, TARGET
from .prompts import system_prompt

_client = OpenAI(base_url=LLM.base_url, api_key=LLM.api_key)


@dataclass
class Correction:
    your_text: str = ""
    correction: str = ""
    explanation: str = ""
    tag: str = ""


@dataclass
class VocabItem:
    term: str = ""
    translation: str = ""
    example: str = ""


@dataclass
class TurnResult:
    reply: str
    reply_lang: str = ""
    corrections: list[Correction] = field(default_factory=list)
    new_vocab: list[VocabItem] = field(default_factory=list)
    detected_level: str = ""
    raw: str = ""
    llm_ms: int = 0


def backend_info() -> dict[str, str]:
    return {"backend": LLM.backend, "model": LLM.model, "base_url": LLM.base_url}


def run_turn(
    *,
    mode: str,
    level: str,
    profile_block: str,
    history: list[dict[str, str]],
    user_text: str,
    length: str = "medium",
) -> TurnResult:
    """Send one conversational turn and parse the structured JSON reply."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(mode, level, profile_block, length)},
        *history,
        {"role": "user", "content": user_text},
    ]

    kwargs: dict[str, Any] = dict(
        model=LLM.model,
        messages=messages,
        temperature=0.4,  # lower -> steadier, fewer invented corrections
        max_tokens=800,  # room for English-teacher explanations without truncation
    )
    # Qwen3 defaults to emitting a long hidden reasoning trace (~30s/turn on this
    # hardware). reasoning_effort="none" fully turns it off — measured 40s -> ~2s.
    # (Ollama's extension of the OpenAI knob; ignored/tolerated elsewhere.)
    if "qwen3" in LLM.model.lower():
        kwargs["reasoning_effort"] = "none"

    t0 = time.perf_counter()
    resp = _create_resilient(kwargs)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    content = resp.choices[0].message.content or ""
    fallback = NATIVE.code if mode == "english_teacher" else TARGET.code
    result = _parse_turn(content, fallback_lang=fallback)
    result.llm_ms = elapsed_ms
    return result


def _create_resilient(kwargs: dict[str, Any]):
    """Call chat.completions, degrading gracefully for backends that reject
    optional params (JSON mode, reasoning_effort). Order matters: keep the speed
    win (reasoning_effort) as long as possible, drop JSON mode first."""
    # 1) full: JSON mode + whatever's in kwargs (incl. reasoning_effort).
    try:
        return _client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception:
        pass
    # 2) drop JSON mode.
    try:
        return _client.chat.completions.create(**kwargs)
    except Exception:
        pass
    # 3) drop reasoning_effort too (last resort).
    kwargs.pop("reasoning_effort", None)
    return _client.chat.completions.create(**kwargs)


_FIELD_LABEL_RE = re.compile(
    r"""(?im)^\s*["']?(reply[_ ]?lang|corrections?|new[_ ]?vocab|detected[_ ]?level|reply)["']?\s*[:=]""",
)


def _is_degenerate_reply(reply: str) -> bool:
    """True if the 'reply' is actually a leaked field label / metadata (a small
    model occasionally emits e.g. 'Replylang: en' as the reply value)."""
    r = reply.strip()
    if len(r) < 3 or r.lower() in {"en", "it", "…"}:
        return True
    return bool(re.match(r"(?i)^(reply)?[_ ]?lang\b|^detected[_ ]?level\b", r))


def _salvage_reply(content: str) -> str:
    """Recover a spoken reply from broken/partial JSON: prefer the "reply" value,
    else strip JSON syntax + field-label lines and keep the prose."""
    text = _strip_think(content)
    m = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        try:
            return _demarkdown(json.loads('"' + m.group(1) + '"'))
        except Exception:
            return _demarkdown(m.group(1))
    lines = []
    for ln in text.splitlines():
        s = ln.strip().strip(",")
        if not s or re.fullmatch(r"[\{\}\[\]]+", s) or _FIELD_LABEL_RE.match(s):
            continue
        lines.append(ln.strip())
    return _demarkdown(" ".join(lines))


def _parse_turn(content: str, fallback_lang: str) -> TurnResult:
    default = TARGET.repeat_prompt if fallback_lang == TARGET.code else NATIVE.repeat_prompt
    data = _extract_json(content)
    if data is None:
        # Broken JSON — salvage a clean reply, never show raw field labels.
        return TurnResult(reply=_salvage_reply(content) or default, reply_lang=fallback_lang, raw=content)

    corrections = [
        Correction(
            your_text=_demarkdown(c.get("your_text", "")),
            correction=_demarkdown(c.get("correction", "")),
            explanation=_demarkdown(c.get("explanation", "")),
            tag=str(c.get("tag", "")),
        )
        for c in data.get("corrections", []) or []
        if isinstance(c, dict)
    ]
    new_vocab = [
        VocabItem(
            term=str(v.get("term", "")).strip(),
            translation=str(v.get("translation", "")).strip(),
            example=str(v.get("example", "")).strip(),
        )
        for v in data.get("new_vocab", []) or []
        if isinstance(v, dict) and str(v.get("term", "")).strip()
    ]
    reply = _demarkdown(data.get("reply", ""))
    if _is_degenerate_reply(reply):
        salvaged = _salvage_reply(content)
        reply = salvaged if salvaged and not _is_degenerate_reply(salvaged) else default
    return TurnResult(
        reply=reply,
        reply_lang=str(data.get("reply_lang", fallback_lang)).strip() or fallback_lang,
        corrections=corrections,
        new_vocab=new_vocab,
        detected_level=str(data.get("detected_level", "")).strip(),
        raw=content,
    )


def _demarkdown(text: Any) -> str:
    """Strip markdown emphasis (gemma likes to wrap words in *asterisks*) so it
    doesn't leak into the spoken reply or correction cards."""
    return re.sub(r"[*_`]+", "", str(text)).strip()


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _extract_json(content: str) -> dict[str, Any] | None:
    text = _strip_think(content).strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the first balanced {...} object.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None

"""Benchmark local LLMs for the tutor task: latency + correction quality.

Runs the exact run_turn() prompt against several Ollama models on a fixed set of
Italian learner utterances (some correct, some with real errors) and prints a
side-by-side comparison. Usage:  uv run python scripts/bench_models.py
"""

from __future__ import annotations

import time

from openai import OpenAI

from tutore.prompts import system_prompt

MODELS = ["qwen3:14b", "qwen2.5:7b-instruct", "gemma3:4b"]

# (utterance, is_it_actually_wrong) — tests both over-correction and real catches.
CASES = [
    ("Ciao, come stai?", False),
    ("Grazie a tutti.", False),
    ("Ho dormito molto bene.", False),
    ("Ieri io ho andato al mare.", True),        # essere, not avere
    ("La casa è molto bello.", True),            # agreement: bella
    ("Vado a scuola con la macchina di mio amico.", True),  # del mio amico
]

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")


def run(model: str, text: str) -> tuple[float, dict]:
    import json

    sys = system_prompt("immersion", "A2", "- Working level: A2")
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": text}],
        temperature=0.4,
        max_tokens=500,
    )
    if "qwen3" in model.lower():
        kwargs["reasoning_effort"] = "none"
    t = time.perf_counter()
    try:
        r = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        r = client.chat.completions.create(**kwargs)
    dt = time.perf_counter() - t
    content = r.choices[0].message.content or ""
    try:
        import re
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        data = json.loads(content)
    except Exception:
        data = {"reply": content[:60], "corrections": ["<unparseable>"]}
    return dt, data


def main() -> None:
    for model in MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)
        times = []
        for text, wrong in CASES:
            try:
                dt, data = run(model, text)
            except Exception as e:
                print(f"  ERR {model}: {str(e)[:80]}")
                break
            times.append(dt)
            corr = data.get("corrections", [])
            n = len(corr) if isinstance(corr, list) else 0
            flag = "✓" if (n > 0) == wrong else "✗"  # matches expectation?
            expect = "should correct" if wrong else "should be clean"
            print(f"  [{flag}] {dt:5.1f}s  {text!r}  ({expect}) -> {n} correction(s)")
            if n and isinstance(corr, list) and isinstance(corr[0], dict):
                print(f"         fix: {corr[0].get('your_text','')!r} -> {corr[0].get('correction','')!r}")
        if times:
            print(f"  AVG: {sum(times)/len(times):.1f}s   (min {min(times):.1f}s, max {max(times):.1f}s)")


if __name__ == "__main__":
    main()

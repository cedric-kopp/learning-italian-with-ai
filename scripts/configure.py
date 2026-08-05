"""Interactive setup: choose the language you want to LEARN (and, optionally, your
native language for explanations + flashcard backs). Writes .env, pins the right
spaCy model in pyproject.toml, and runs `uv sync`. No source-diving required.

Usage:
  uv run python scripts/configure.py                 # interactive
  uv run python scripts/configure.py --target es      # non-interactive
  uv run python scripts/configure.py --target ja --native de --yes
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tutore.languages import DEFAULT_NATIVE, DEFAULT_TARGET, LANGUAGES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
PYPROJECT = ROOT / "pyproject.toml"
SPACY_VERSION = "3.8.0"


def model_url(model: str) -> str:
    base = "https://github.com/explosion/spacy-models/releases/download"
    return f"{base}/{model}-{SPACY_VERSION}/{model}-{SPACY_VERSION}-py3-none-any.whl"


def show_languages() -> None:
    print("\nLanguages you can LEARN (target):\n")
    for code, L in LANGUAGES.items():
        tier = "  ⚠ caveats" if L.caveats else ""
        print(f"  {L.flag}  {code}  {L.name_en}{tier}")
        if L.caveats:
            print(f"        {L.caveats}")
    print()


def upsert_env(updates: dict[str, str]) -> None:
    lines = (ENV.read_text() if ENV.exists() else ENV_EXAMPLE.read_text()).splitlines()
    seen: set[str] = set()
    out = []
    for ln in lines:
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=", ln)
        if m and m.group(1) in updates:
            key = m.group(1)
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(ln)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    ENV.write_text("\n".join(out) + "\n")


def swap_spacy_model(model: str) -> None:
    """Replace any pinned spaCy model URL dependency with the target's."""
    text = PYPROJECT.read_text()
    new_dep = f'    "{model} @ {model_url(model)}",'
    pattern = re.compile(r'^\s*"[a-z]{2}_core_[a-z]+_sm @ https://github\.com/explosion/spacy-models.*?",$', re.M)
    if pattern.search(text):
        text = pattern.sub(new_dep, text, count=1)
    PYPROJECT.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="language code to learn (e.g. es, fr, it, pt, ja)")
    ap.add_argument("--native", help="native/explanation language code (default en)")
    ap.add_argument("--app-name", default=None)
    ap.add_argument("--tutor", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the confirm prompt")
    args = ap.parse_args()

    interactive = not args.target
    if interactive:
        show_languages()

    target = (args.target or input(f"Language to LEARN [{DEFAULT_TARGET}]: ").strip() or DEFAULT_TARGET).lower()
    if target not in LANGUAGES:
        sys.exit(f"Unknown target '{target}'. Choose from: {', '.join(LANGUAGES)}")

    native = (args.native or (input(f"Your NATIVE language [{DEFAULT_NATIVE}]: ").strip() if interactive else "") or DEFAULT_NATIVE).lower()
    if native not in LANGUAGES:
        print(f"Note: '{native}' isn't in the registry; explanations still work, but add it for a proper flag/name.")

    T = LANGUAGES[target]
    app_name = args.app_name or (input("App name [Tutore]: ").strip() if interactive else "") or "Tutore"
    tutor = args.tutor or (input(f"Tutor name [{T.tutor_name}]: ").strip() if interactive else "") or T.tutor_name

    if T.caveats:
        print(f"\n⚠  {T.name_en}: {T.caveats}\n")

    updates = {
        "TARGET_LANG": target,
        "NATIVE_LANG": native,
        "APP_NAME": app_name,
        "TUTOR_NAME": tutor,
        "ANKI_DECK": f"{app_name} {T.name_en}",
    }
    print("\nWill write to .env:")
    for k, v in updates.items():
        print(f"  {k}={v}")
    print(f"And pin spaCy model: {T.spacy_model}")

    if interactive and not args.yes:
        if input("\nProceed? [Y/n]: ").strip().lower() in {"n", "no"}:
            sys.exit("Aborted.")

    upsert_env(updates)
    swap_spacy_model(T.spacy_model)
    print("\n==> uv sync (installs the spaCy model for your language)")
    subprocess.run(["uv", "sync"], cwd=ROOT, check=True)
    print(f"\n✓ Configured to learn {T.flag} {T.name_en}. Start with ./scripts/run-ollama.sh")
    if target == "ja":
        print("  Japanese also needs the misaki[ja] G2P extra for TTS — install if speech fails.")


if __name__ == "__main__":
    main()

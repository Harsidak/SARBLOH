"""``recall()``: search the game's timeline and the skills from the REPL. Pre-imported by the kernel.

    recall("#12")                  # one step and what it changed; "12-20" a range, "level 1" a level, or words
    recall("bfs walls", "skills")  # the skills that match, printed and loaded into SKILLS
    recall("", "skills")           # every skill by name, one line each (nothing is loaded)

It reads the memory files the host writes (the game's ``timeline.jsonl`` and the shared ``skills.json``, under
``$SARBLOH_MEMORY_DIR``), so it costs no move and no host request. The skills it finds go into ``SKILLS``, the
agent's dict of loaded skills (name -> text), which holds at most ``LOADED_SKILLS``: loading one more drops the
oldest. The host copies SKILLS after every call, like the other memory variables, and the curator reads it. A display
event of type ``MIME`` tells the host what was searched, for the trace.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

MIME = "application/vnd.sarbloh.recall+json"
LOADED_SKILLS = 5   # = harness.memory.lifecycle.LOADED_SKILLS


def _ns() -> dict[str, Any]:
    return sys.modules["__main__"].__dict__


def load_skills(skills: dict[str, str]) -> list[str]:
    """Put ``skills`` (name -> text) into SKILLS, newest last; past LOADED_SKILLS the oldest go. Returns the names
    now in SKILLS."""
    ns = _ns()
    held = ns.get("SKILLS")
    if not isinstance(held, dict):
        held = {}
    for name, text in skills.items():
        held.pop(name, None)
        held[name] = text
    while len(held) > LOADED_SKILLS:
        held.pop(next(iter(held)))
    ns["SKILLS"] = held
    return list(held)


def recall(query: str = "", scope: str = "all") -> None:
    """Print what the timeline and the skills hold about ``query``; ``scope`` is "all", "timeline" or "skills". The
    skills found are loaded into SKILLS (at most five; the oldest is dropped). Free: it reads memory, not the game."""
    root, game = os.environ.get("SARBLOH_MEMORY_DIR"), os.environ.get("SARBLOH_GAME")
    if not root or not game:
        print("recall: this kernel has no game memory")
        return
    from harness.memory.lessons import SkillBook
    from harness.memory.lifecycle import recall_text
    from harness.memory.timeline import Timeline

    book = SkillBook(Path(root) / "skills.json")
    cap = int(os.environ.get("SARBLOH_RECALL_TOKENS") or 1000)
    try:
        text, hits, found = recall_text(Timeline(Path(root) / game / "timeline.jsonl"), book, str(query or ""),
                                        str(scope or "all"), cap)
    except ValueError as exc:
        print(f"recall refused: {exc}")
        return
    held = load_skills({n: book.text(n) or "" for n in found}) if found else []
    if found:
        text += f"\n[loaded into SKILLS: {', '.join(found)}. SKILLS now holds {', '.join(held)}]"
    try:
        from rlm import emit
        emit({MIME: {"query": str(query or ""), "scope": str(scope or "all"), "hits": hits, "loaded": found,
                     "chars": len(text)}})
    except Exception:  # noqa: BLE001 - outside the kernel there is no host to tell
        pass
    print(text)

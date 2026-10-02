"""E039 lessons reconcile (switch ``memory.lessons_reconcile``): after every level won, one hidden JSON call (thinking
off, like the curator) compares the lessons graph with the game's wrong rulebook (E018) and corrects the lessons.

Input: the lessons of this game (newest first), then the most used lessons of other games, each with its id and where
it came from, under ``lessons_tokens``; the wrong rulebook of this game; the newest E022 review if there is one.
Output: lessons to drop (shown false, or useless), to rewrite (partly wrong or vague) and to merge (the same lesson
twice). ``GameMemory.reconcile_lessons`` applies them: only lessons the pass read can change, a lesson that other games
also taught can be rewritten but never dropped, and at most 10 changes per pass. The agent never sees the pass; it sees
the corrected lessons in its pinned memory. A reply that does not parse twice raises; the caller logs it and carries on.
"""

from __future__ import annotations

from typing import Any

from harness.agent.curator import _ask
from harness.memory.store import tokens

RECONCILE_SYSTEM = """You keep the lessons of an agent that is learning unknown turn-based grid games. The agent has just won a level. You never act. You return JSON only.

You get the lessons kept so far, each with its id and the games it came from, and the wrong rulebook of this game: rules the agent tested in this game and found false. Compare them and correct the lessons:
1. drop: a lesson that a wrong rule shows false, or that says nothing useful. Only a lesson from this game alone can be dropped.
2. rewrite: a lesson that is partly wrong or too vague. Give its new text: one short general sentence that names objects by colour and shape, not by id. A lesson that other games also taught may only be narrowed (say when it holds), not reversed.
3. merge: two or more lessons that say the same thing. Give their ids and one text for all.
Change only what the wrong rulebook and the lessons show. At most 10 changes; empty lists are fine.

Return exactly:
{"drop": [{"id": "rule:...", "why": "..."}], "rewrite": [{"id": "rule:...", "text": "...", "why": "..."}], "merge": [{"ids": ["rule:...", "rule:..."], "text": "...", "why": "..."}]}"""


def lesson_line(n: dict[str, Any], game: str) -> str:
    others = [g for g in n["games"] if g != game]
    where = ["this game"] if game in n["games"] else []
    if others:
        where.append(f"{len(others)} other game{'s' if len(others) > 1 else ''}")
    return f"- {n['id']} [{n['type']}] {n['text']}" + (f" ({' and '.join(where)})" if where else "")


def reconcile_input(memory: Any, level: int, lessons_tokens: int) -> tuple[str, set[str]]:
    """The input text and the ids of the lessons it shows. ``level`` is 0-based (the level just won)."""
    g, game = memory.lessons, memory.game
    with g.lock:
        nodes = [n for n in g.nodes.values() if n["type"] != "shape"]
    mine = sorted((n for n in nodes if game in n["games"]), key=lambda n: -n["created"])
    rest = sorted((n for n in nodes if game not in n["games"]), key=lambda n: (-n["uses"], -n["created"]))
    lines: list[str] = []
    shown: set[str] = set()
    for n in mine + rest:
        line = lesson_line(n, game)
        if tokens("\n".join(lines + [line])) > lessons_tokens:
            lines.append(f"(+{len(mine) + len(rest) - len(shown)} lessons not shown: leave them)")
            break
        lines.append(line)
        shown.add(n["id"])
    wrong = memory.wrong.render(lessons_tokens // 2) if memory.wrong is not None else ""
    reviews = memory.working.reviews
    parts = [f"<level>{level + 1} of {game} won</level>",
             "<lessons>\n" + ("\n".join(lines) or "(none)") + "\n</lessons>",
             "<wrong_rulebook>\n" + (wrong or "(empty)") + "\n</wrong_rulebook>"]
    if reviews:
        parts.append(f"<level_review>\n{memory.working.review_line(reviews[-1])}\n</level_review>")
    return "\n\n".join(parts), shown


def reconcile(*, llm: Any, memory: Any, level: int, max_tokens: int, lessons_tokens: int,
              overflow: type[Exception]) -> dict[str, Any]:
    """One pass after level ``level`` (0-based) is won. Returns what changed; {"skipped": why} when there is nothing
    to compare."""
    if memory.lessons is None:
        return {"skipped": "lessons graph off"}
    text, shown = reconcile_input(memory, level, lessons_tokens)
    if not shown:
        return {"skipped": "no lessons"}
    out = _ask(llm, RECONCILE_SYSTEM, text, max_tokens, overflow)
    return {"shown": len(shown), **memory.reconcile_lessons(out, shown)}

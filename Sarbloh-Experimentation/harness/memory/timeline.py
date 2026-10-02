"""Memory #1, the timeline: every step and every act, written by the host, never by the agent. It is not in the
context; the agent reads it with ``recall`` (by step, range, level or words) and the curator reads its last steps."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from harness.memory.store import append_jsonl, read_jsonl


class Timeline:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = read_jsonl(self.path)

    def append(self, row: dict[str, Any]) -> None:
        row = {"t": round(time.time(), 2), **row}
        self.rows.append(row)
        append_jsonl(self.path, row)

    def steps(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("kind") == "step"]

    def recent_steps(self, n: int) -> list[dict[str, Any]]:
        return self.steps()[-n:]

    def archive(self) -> Path | None:
        """Move the file aside (a new game in the same memory directory) and start empty."""
        if not self.path.exists():
            self.rows = []
            return None
        dest = self.path.with_name(f"{self.path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        self.path.replace(dest)
        self.rows = []
        return dest

    def search(self, query: str, limit: int = 40) -> list[dict[str, Any]]:
        """``#12`` one step, ``12-20`` a range, ``level 1`` a level, anything else: all words must appear."""
        q = (query or "").strip().lower()
        steps = self.steps()
        m = re.fullmatch(r"#?(\d+)\s*(?:-|\.\.|to)\s*#?(\d+)", q)
        if m:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            return [r for r in steps if a <= int(r.get("i", -1)) <= b][:limit]
        m = re.fullmatch(r"#(\d+)|step\s+(\d+)", q)
        if m:
            i = int(m.group(1) or m.group(2))
            return [r for r in self.rows if r.get("i") == i or i in (r.get("steps") or [])][:limit]
        m = re.fullmatch(r"level\s+(\d+)", q)
        if m:
            lvl = int(m.group(1))
            return [r for r in steps if r.get("level") == lvl][-limit:]
        words = [w for w in re.split(r"\s+", q) if w]
        if not words:
            return self.rows[-limit:]
        hits = [r for r in self.rows if all(w in render_row(r).lower() for w in words)]
        return hits[-limit:]


def render_row(r: dict[str, Any]) -> str:
    if r.get("kind") == "step":
        tail = []
        if r.get("level_up"):
            tail.append("LEVEL UP")
        if r.get("state") == "GAME_OVER":
            tail.append("GAME_OVER")
        return (f"#{r.get('i')} L{r.get('level')} {r.get('action')}: {r.get('change')}"
                + (f" [{', '.join(tail)}]" if tail else "") + (" [repeat]" if r.get("repeat_of") is not None else ""))
    if r.get("kind") == "act":
        return f"act turn {r.get('turn')} L{r.get('level')} steps {r.get('steps')}: plan: {r.get('plan')}"
    if r.get("kind") == "level_up":
        return f"LEVEL {r.get('level')} WON at step #{r.get('i')}, goal was: {r.get('goal')}"
    return str({k: v for k, v in r.items() if k != "t"})

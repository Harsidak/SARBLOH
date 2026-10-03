"""``observe()``: the game state in the REPL, read-only. Pre-imported by the kernel.

The host writes ``$RLM_SESSION_DIR/.prime/observation.json`` after every act and appends one line per step to
``history.jsonl``. ``observe()`` reads them, and it tells the host (a display event of type ``MIME``) to send the
full state (briefing, objects, board and picture) as the next message. The host sends it at most once per state.

    obs = observe()
    obs.board[r][c]            # the colour letter at row r, column c
    obs.objects                # list of dicts: id, letter, name, size, bbox [r0, c0, r1, c1], hash, corners,
                               #   parent, children, adjacent, hud, cells [(r, c), ...] (None above 256 cells)
    obs.change                 # the newest step's change line
    obs.briefing               # every briefing line (the message shows at most 30)
    obs.history                # the last 20 steps: {"i", "level", "action", "change", "state"}
    print(obs.ascii(r0, c0, r1, c1))   # a window of the board with row and column labels
    obs.step, obs.status, obs.background, obs.legend, obs.text
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MIME = "application/vnd.sarbloh.observe+json"
HISTORY = 20
_OLD_NAMES = {"letters": "obs.board", "find": "a list comprehension over obs.objects", "grid": "obs.board"}


def _dir() -> Path:
    return Path(os.environ.get("RLM_SESSION_DIR", ".")) / ".prime"


class Observation:
    """The state at the moment ``observe()`` was called. It does not change: call ``observe()`` again after an act."""

    def __init__(self, data: dict[str, Any], history: list[dict[str, Any]]) -> None:
        self.step: int | None = data.get("step")
        self.status: str = data.get("status") or ""
        self.change: str | None = data.get("change")
        self.board: list[str] = list(data.get("letters") or [])
        self.objects: list[dict[str, Any]] = list(data.get("objects") or [])
        self.briefing: list[str] = list(data.get("briefing") or [])
        self.background: str | None = data.get("background")
        self.legend: str = data.get("legend") or ""
        self.text: str = data.get("text") or "(no state yet)"
        self.history = history

    def ascii(self, r0: int = 0, c0: int = 0, r1: int | None = None, c1: int | None = None) -> str:
        """Rows r0..r1 and columns c0..c1 (inclusive) of the board as letters, with row and column labels."""
        rows = self.board
        if not rows:
            return "(no state yet)"
        r1 = len(rows) - 1 if r1 is None else min(r1, len(rows) - 1)
        c1 = len(rows[0]) - 1 if c1 is None else min(c1, len(rows[0]) - 1)
        r0, c0 = max(0, r0), max(0, c0)
        cols = range(c0, c1 + 1)
        out = ["    " + "".join(str(c // 10 % 10) for c in cols), "    " + "".join(str(c % 10) for c in cols)]
        out += [f"{r:>3} " + rows[r][c0:c1 + 1] for r in range(r0, r1 + 1)]
        return "\n".join(out)

    def __getattr__(self, name: str) -> Any:
        hint = f"; use {_OLD_NAMES[name]}" if name in _OLD_NAMES else ""
        raise AttributeError(f"an observation has no {name!r}{hint}. Fields: step, status, change, board, objects, "
                             "briefing, background, legend, history, text; and obs.ascii(r0, c0, r1, c1)")

    def __repr__(self) -> str:
        size = f"{len(self.board)}x{len(self.board[0])}" if self.board else "empty"
        return f"Observation(step {self.step}, {len(self.objects)} objects, board {size})"


def observe() -> Observation:
    """The current state. Free. The full state (briefing, objects, board and picture) is sent to you as the next
    message, once per state; the returned snapshot is for your code."""
    try:
        data = json.loads((_dir() / "observation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    try:
        from rlm import emit
        emit({MIME: {"step": data.get("step")}})
    except Exception:  # noqa: BLE001 - outside the kernel there is no host to tell
        pass
    return Observation(data, history(HISTORY))


def history(n: int = HISTORY) -> list[dict[str, Any]]:
    """The last ``n`` steps of the game, oldest first."""
    try:
        lines = (_dir() / "history.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-int(n):] if n else []:
        try:
            out.append(json.loads(line))
        except ValueError:   # a line the host is still writing
            continue
    return out


# --- host side ---------------------------------------------------------------------------------------------------

def write(session_dir: Path, data: dict[str, Any]) -> None:
    """Publish the state for ``observe()`` (atomic: a cell never reads half a file)."""
    d = Path(session_dir) / ".prime"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "observation.json.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, d / "observation.json")


def append_history(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    """One line per step for ``obs.history``."""
    d = Path(session_dir) / ".prime"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "history.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row.get(k) for k in ("i", "level", "action", "change", "state")}) + "\n")

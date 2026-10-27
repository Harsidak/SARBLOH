"""``observe()``: the game state in the REPL, read-only. Pre-imported by the kernel.

The host writes ``$RLM_SESSION_DIR/.prime/observation.json`` after every act and appends one line per step to
``history.jsonl``, and one line per step with the boards before and after it to ``moves.jsonl``. ``observe()``
reads them, and it tells the host (a display event of type ``MIME``) to add the
full state (briefing, objects and board) to the cell's own output. The host shows it at most once per state.

    obs = observe()
    obs.board[r][c]            # the colour letter at row r, column c
    obs.objects                # list of dicts: id, letter, name, size, bbox [r0, c0, r1, c1], hash, corners,
                               #   parent, children, adjacent, hud, cells [(r, c), ...] (None above 256 cells)
    obs.change                 # the newest step's change line
    obs.briefing               # every briefing line (the message shows at most 30)
    obs.history                # the last 20 steps, newest first: {"i", "level", "action", "change", "state"}
    obs.before                 # the board before the newest move (None before the first move)
    print(obs.ascii(r0, c0, r1, c1))   # a window of the board with row and column labels
    obs.step, obs.status, obs.background, obs.legend, obs.text
    moves()                    # every move of this level, oldest first: {"i", "level", "move", "before", "after",
                               #   "change", "state"}; moves(all_levels=True) the whole game
    frame_diff(before, after)  # what changed between two boards, as objects (default: obs.before -> the board now)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MIME = "application/vnd.sarbloh.observe+json"
HISTORY = 20
MOVES = "moves.jsonl"
MOVE_FIELDS = ("i", "level", "move", "before", "after", "change", "state")
DIFF_MAX = 30   # entries per frame_diff list
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
        self.before: list[str] | None = data.get("before")
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
        raise AttributeError(f"an observation has no {name!r}{hint}. Fields: step, status, change, board, before, "
                             "objects, briefing, background, legend, history, text; and obs.ascii(r0, c0, r1, c1)")

    def __repr__(self) -> str:
        size = f"{len(self.board)}x{len(self.board[0])}" if self.board else "empty"
        return f"Observation(step {self.step}, {len(self.objects)} objects, board {size})"


def _read() -> dict[str, Any]:
    try:
        return json.loads((_dir() / "observation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def snapshot() -> Observation:
    """The current state, like ``observe()``, but the host is not asked to send it as a message (``play()`` reads
    the board after every move with it)."""
    return Observation(_read(), history(HISTORY))


def observe() -> Observation:
    """The current state. Free. The full state (briefing, objects, board and picture) is sent to you as the next
    message, once per state; the returned snapshot is for your code."""
    data = _read()
    try:
        from rlm import emit
        emit({MIME: {"step": data.get("step")}})
    except Exception:  # noqa: BLE001 - outside the kernel there is no host to tell
        pass
    return Observation(data, history(HISTORY))


def history(n: int = HISTORY) -> list[dict[str, Any]]:
    """The last ``n`` steps of the game, newest first."""
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
    return out[::-1]


def moves(all_levels: bool = False) -> list[dict[str, Any]]:
    """Every move of this level so far, oldest first, the tries before a GAME_OVER included; ``all_levels=True``
    gives the whole game. Each is a dict: ``i``, ``level``, ``move`` (as written in a ``play()`` path), ``before``
    and ``after`` (boards like ``obs.board``), ``change`` and ``state``. Free."""
    try:
        lines = (_dir() / MOVES).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:   # a line the host is still writing
            continue
    if not all_levels:   # the moves after the last level up
        ups = [k for k, r in enumerate(rows) if r.get("level_up")]
        rows = rows[ups[-1] + 1:] if ups else rows
    return [{k: r.get(k) for k in MOVE_FIELDS} for r in rows]


def _letters(board: Any) -> list[str]:
    board = getattr(board, "board", board)   # an Observation
    return [r if isinstance(r, str) else "".join(str(x) for x in r) for r in board]


def _parts(board: list[str], seeds: list[tuple[int, int]], bg: str) -> list[tuple[str, frozenset[tuple[int, int]]]]:
    """The 4-connected one-letter areas that hold a seed cell, background left out."""
    h, w = len(board), len(board[0])
    seen: set[tuple[int, int]] = set()
    out = []
    for s in seeds:
        letter = board[s[0]][s[1]]
        if s in seen or letter == bg:
            continue
        seen.add(s)
        stack, cells = [s], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= n[0] < h and 0 <= n[1] < w and n not in seen and board[n[0]][n[1]] == letter:
                    seen.add(n)
                    stack.append(n)
        out.append((letter, frozenset(cells)))
    return out


def _box(cells: frozenset[tuple[int, int]]) -> list[int]:
    rs, cs = [r for r, _ in cells], [c for _, c in cells]
    return [min(rs), min(cs), max(rs), max(cs)]


def _shape(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _box(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _turned(shape: frozenset[tuple[int, int]]) -> dict[int, frozenset[tuple[int, int]]]:
    """The shape turned 90, 180 and 270 degrees clockwise."""
    out, s = {}, shape
    for deg in (90, 180, 270):
        h = max(r for r, _ in s) + 1
        s = _shape(frozenset((c, h - 1 - r) for r, c in s))
        out[deg] = s
    return out


def frame_diff(before: Any = None, after: Any = None) -> dict[str, Any]:
    """What changed between two boards (lists of strings like ``obs.board``, or observations), as objects: areas of
    one letter, 4-connected, background left out. With no arguments it compares ``obs.before`` with the board now.
    Returns ``changed`` (the number of cells) and the lists ``moved`` (``from`` and ``to`` boxes [r0, c0, r1, c1],
    ``by`` [rows, columns]), ``rotated`` (also ``turn``, degrees clockwise), ``changed_color`` (the same cells,
    ``from_letter`` and ``to_letter``), ``appeared`` and ``disappeared``, at most 30 entries each. Free."""
    if before is None and after is None:
        now = snapshot()
        before, after = now.before, now.board
    if before is None or after is None:
        raise ValueError("frame_diff needs two boards: there is no move yet, so obs.before is None")
    a, b = _letters(before), _letters(after)
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        raise ValueError(f"the boards differ in size: {len(a)} and {len(b)} rows")
    changed = [(r, c) for r in range(len(a)) for c in range(len(a[r])) if a[r][c] != b[r][c]]
    out: dict[str, Any] = {"changed": len(changed), "moved": [], "rotated": [], "changed_color": [], "appeared": [],
                           "disappeared": []}
    if not changed:
        return out
    count: dict[str, int] = {}
    for row in a:
        for x in row:
            count[x] = count.get(x, 0) + 1
    bg = max(count, key=count.get)
    left = _parts(b, changed, bg)

    def nearest(cands: list[Any], box: list[int]) -> Any:
        return min(cands, key=lambda n: abs(_box(n[1])[0] - box[0]) + abs(_box(n[1])[1] - box[1]))

    for letter, cells in _parts(a, changed, bg):
        box, shape = _box(cells), _shape(cells)
        same = [n for n in left if n[0] == letter and _shape(n[1]) == shape]
        if same:
            n = nearest(same, box)
            left.remove(n)
            to = _box(n[1])
            out["moved"].append({"letter": letter, "size": len(cells), "from": box, "to": to,
                                 "by": [to[0] - box[0], to[1] - box[1]]})
            continue
        turns = _turned(shape)
        turned = [n for n in left if n[0] == letter and len(n[1]) == len(cells) and _shape(n[1]) in turns.values()]
        if turned:
            n = nearest(turned, box)
            left.remove(n)
            deg = next(d for d, s in turns.items() if s == _shape(n[1]))
            out["rotated"].append({"letter": letter, "size": len(cells), "from": box, "to": _box(n[1]), "turn": deg})
            continue
        same_cells = [n for n in left if n[1] == cells]
        if same_cells:
            left.remove(same_cells[0])
            out["changed_color"].append({"from_letter": letter, "to_letter": same_cells[0][0], "size": len(cells),
                                         "bbox": box})
            continue
        out["disappeared"].append({"letter": letter, "size": len(cells), "bbox": box})
    out["appeared"] = [{"letter": n[0], "size": len(n[1]), "bbox": _box(n[1])} for n in left]
    for k in ("moved", "rotated", "changed_color", "appeared", "disappeared"):
        out[k] = out[k][:DIFF_MAX]
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


def append_moves(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    """One line per step for ``moves()``: the fields of ``MOVE_FIELDS`` and ``level_up``."""
    d = Path(session_dir) / ".prime"
    d.mkdir(parents=True, exist_ok=True)
    with (d / MOVES).open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: row.get(k) for k in (*MOVE_FIELDS, "level_up")}) + "\n")

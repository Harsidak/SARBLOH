"""``play()``: make game moves from Python. Pre-imported by the kernel.

The REPL is the agent's only game tool: it models the game in code, searches the model and sends the moves it finds,
in the same cell. ``play()`` is the only way to move: it sends the moves one at a time, each as a host request
(``AgentSession._host``, type ``arc.act``) that runs it exactly like the old act tool did (the same parsing, legality
check, change lines and stops: level up, GAME_OVER, game end), and it stops early at the first move that did not do
what the agent predicted or that changed nothing on the board.

    r = await play(["4", "4", "1"])                 # moves, no check
    r = await play(path, predict=step)              # stops at the first move step() got wrong
    r.lines        # one change line per move: "#12 A4(right): obj 4 R 3x3 moved right 3 -> ..."
    r.done, r.stopped, r.surprise, r.level_up, r.state
    r.obs          # the state after the last move

A move is "1".."5", "7", "reset", "6 r c" (a click at row r, column c), an int 1..7, or a tuple (6, r, c).

Each move also sends the agent's memory variables (GOAL, PLAN, HYPOTHESES, FINDINGS), so the host saves them before the
move runs; after a level up PLAN, HYPOTHESES and FINDINGS are emptied here, as the host empties them in memory.
"""

from __future__ import annotations

import itertools
import json
import sys
from typing import Any

from rlm import repl as _repl

MEMORY_VARS = ("GOAL", "PLAN", "HYPOTHESES", "FINDINGS")
_PLAY_IDS = itertools.count(1)   # numbers each play() call, so the trace can group its moves


def _plain(v: Any) -> Any:
    try:
        return json.loads(json.dumps(v, default=str))
    except (TypeError, ValueError):   # e.g. tuple keys
        return repr(v)


def memory_values() -> dict[str, Any]:
    """The memory variables of the agent's namespace as JSON values; a deleted one is left out."""
    ns = sys.modules["__main__"].__dict__
    return {k: _plain(ns[k]) for k in MEMORY_VARS if k in ns}


class ActResult:
    """What one host move request did."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.text: str = data.get("text") or ""
        self.lines: list[str] = [ln for ln in self.text.splitlines() if ln.startswith("#")]
        self.done: int = int(data.get("done") or 0)
        self.stopped: str | None = data.get("stopped")
        self.level_up: bool = bool(data.get("level_up"))
        self.state: str = data.get("state") or ""
        self.batch_stop: str | None = data.get("batch_stop")   # the host's reason to send no more of the path

    def __repr__(self) -> str:
        return f"ActResult(done={self.done}, stopped={self.stopped!r}, state={self.state!r})"


def _move(a: Any) -> str:
    if isinstance(a, str):
        return a
    if isinstance(a, (tuple, list)):
        return " ".join(str(int(x)) for x in a)
    return str(int(a))   # int, numpy int


async def _act(actions: list[Any], **extra: Any) -> ActResult:
    """Send ``actions`` to the host (each costs one move); ``extra`` goes with them (``predicted``, ``path_len``).
    Raises RuntimeError when the request is refused (nothing is spent then). Not shown to the agent: ``play()``
    is its only way to move."""
    args: dict[str, Any] = {"actions": [_move(a) for a in actions], "memory": memory_values(), **extra}
    reply = await _repl.host_request({"type": "arc.act", "args": args})
    if reply.get("status") != "ok":
        raise RuntimeError(str(reply.get("error") or "act failed"))
    res = ActResult(reply.get("result") or {})
    if res.level_up:
        sys.modules["__main__"].__dict__.update(PLAN="", HYPOTHESES=[], FINDINGS=[])
    return res


async def act(*_: Any, **__: Any) -> None:
    """Gone: kept only so an old habit gets a clear message instead of a NameError."""
    raise RuntimeError("act() is gone: make moves with `await play(path)` (a list of moves, made one at a time). "
                       "Add `predict=` to check each move against your model. Nothing was spent.")


# --- play(): a whole plan, stopped at the first surprise ----------------------------------------------------------

SHOW_CELLS = 12      # differing cells listed on a surprise
WINDOW_PAD = 2       # rows and columns shown around them
WINDOW_MAX = 24      # the window is cut to this many rows and columns


class PlayResult:
    """What one ``play()`` call did. ``surprise`` is None, or a dict: ``move`` (its index in the plan), ``action``,
    ``cells`` [(r, c, predicted, actual), ...] (board mode) or ``predicted`` and ``actual`` (state mode)."""

    def __init__(self) -> None:
        self.done: int = 0
        self.stopped: str | None = None
        self.surprise: dict[str, Any] | None = None
        self.lines: list[str] = []
        self.level_up: bool = False
        self.state: str = ""
        self.last: ActResult | None = None
        self.obs: Any = None

    def __repr__(self) -> str:
        return f"PlayResult(done={self.done}, stopped={self.stopped!r}, state={self.state!r})"


def _hud_cells(*observations: Any) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for obs in observations:
        for o in getattr(obs, "objects", None) or []:
            if not o.get("hud"):
                continue
            if o.get("cells"):
                cells.update((int(r), int(c)) for r, c in o["cells"])
            elif o.get("bbox"):
                r0, c0, r1, c1 = (int(x) for x in o["bbox"])
                cells.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    return cells


def _rows(board: Any) -> list[str]:
    return [row if isinstance(row, str) else "".join(str(x) for x in row) for row in board]


def _board_diff(want: Any, board: list[str], skip: set[tuple[int, int]]) -> list[tuple[int, int, str, str]]:
    """Cells where the board differs from the prediction: a whole board, or a dict {(r, c): letter}."""
    if isinstance(want, dict):
        out = []
        for (r, c), v in want.items():
            got = board[r][c] if 0 <= r < len(board) and 0 <= c < len(board[r]) else "?"
            if str(v) != got:
                out.append((int(r), int(c), str(v), got))
        return out
    want = _rows(want)
    if len(want) != len(board) or any(len(a) != len(b) for a, b in zip(want, board)):
        return [(-1, -1, f"{len(want)} rows", f"{len(board)} rows")]
    return [(r, c, want[r][c], board[r][c]) for r in range(len(board)) for c in range(len(board[r]))
            if want[r][c] != board[r][c] and (r, c) not in skip]


def _surprise_text(res: PlayResult, obs: Any) -> str:
    s = res.surprise or {}
    head = f"surprise at move {s.get('move')} ({s.get('action')}):"
    if "cells" not in s:
        return f"{head} predicted {s.get('predicted')!r}, got {s.get('actual')!r}"
    cells = s["cells"]
    if cells and cells[0][0] < 0:
        return f"{head} predicted {cells[0][2]}, the board has {cells[0][3]}"
    listed = ", ".join(f"({r},{c}) predicted {p} got {a}" for r, c, p, a in cells[:SHOW_CELLS])
    more = f" and {len(cells) - SHOW_CELLS} more" if len(cells) > SHOW_CELLS else ""
    rs, cs = [r for r, *_ in cells], [c for _, c, *_ in cells]
    r0, c0 = max(0, min(rs) - WINDOW_PAD), max(0, min(cs) - WINDOW_PAD)
    r1, c1 = min(max(rs) + WINDOW_PAD, r0 + WINDOW_MAX - 1), min(max(cs) + WINDOW_PAD, c0 + WINDOW_MAX - 1)
    return f"{head} {len(cells)} cells differ: {listed}{more}\nthe board now:\n{obs.ascii(r0, c0, r1, c1)}"


SURPRISE_NOTE = ("One rule in your model was wrong at this move. Fix only that rule in `step()`, then search again "
                 "from the board now and call `play()` with the new path.")
NOOP_NOTE = ("The same move hit a wall twice in a row (not one pixel changed), so the rest of the path was not sent. "
             "Check whether the move is blocked here before you send it again.")


def _changed(before: Any, after: Any, skip: set[tuple[int, int]]) -> bool:
    """Whether any cell outside ``skip`` (the HUD) differs between two observations."""
    a, b = before.board, after.board
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return True
    return any(x[c] != y[c] and (r, c) not in skip for r, (x, y) in enumerate(zip(a, b)) for c in range(len(x)))


async def play(path: Any, predict: Any = None, state: Any = None, skip_hud: bool = True,
               quiet: bool = False, stop_if_unchanged: bool = True) -> PlayResult:
    """Make the moves of ``path`` one at a time. Only a level up, the game's end or a wall hit twice (the same move
    twice in a row, not one pixel changed) stops the path; a wrong prediction is reported and the path goes on.

    ``predict(board, move)`` gets ``obs.board`` before the move and returns the board it expects after it: a list of
    rows, or a dict {(r, c): letter} with only the cells to check, or None to not check this move. With ``state``,
    ``predict(state(obs), move)`` is compared with ``==`` to ``state(obs)`` after the move instead. Cells of ``hud``
    objects are not compared unless ``skip_hud=False``. ``stop_if_unchanged`` is ignored. Prints one line per move and, on a surprise, the cells that differ, unless ``quiet``."""
    from observation import snapshot   # the kernel puts the skills folder on the path

    if isinstance(path, (str, int)) or (isinstance(path, tuple) and path and not
                                        isinstance(path[0], (tuple, list, str))):
        path = [path]
    path = list(path)
    res, obs = PlayResult(), snapshot()
    noop, play_id = False, next(_PLAY_IDS)
    del stop_if_unchanged   # kept so old calls still work: the host decides when a path stops
    for i, move in enumerate(path):
        want = None
        if predict is not None:
            try:
                want = predict(state(obs) if state else obs.board, move)
            except Exception as exc:  # noqa: BLE001 - the model's code; nothing is spent for this move
                res.stopped = f"predict failed at move {i}: {type(exc).__name__}: {exc}"
                break
        extra: dict[str, Any] = {"via": "play", "play_id": play_id, "predicted": want is not None}
        if i == 0:
            extra["path_len"] = len(path)   # the host refuses a path over its cap before any move
        try:
            r = await _act([move], **extra)
        except RuntimeError as exc:
            if res.done == 0:
                raise
            res.stopped = str(exc)
            break
        before, obs = obs, snapshot()
        res.last, res.state, res.level_up = r, r.state, r.level_up
        res.done += r.done
        res.lines += r.lines
        if r.stopped or r.done == 0:
            res.stopped = r.stopped or "no move made"
            break
        skip = _hud_cells(before, obs) if skip_hud else set()
        if want is not None and res.surprise is None:   # the first wrong prediction is kept; the path goes on
            if state:
                got = state(obs)
                if got != want:
                    res.surprise = {"move": i, "action": _move(move), "predicted": want, "actual": got}
            else:
                cells = _board_diff(want, obs.board, skip)
                if cells:
                    res.surprise = {"move": i, "action": _move(move), "cells": cells}
            if res.surprise:
                try:   # the host gives the next turn a short thinking budget: a surprise needs a small fix
                    await _repl.host_request({"type": "arc.surprise", "args": {"move": i, "action": _move(move)}})
                except Exception:  # noqa: BLE001 - outside the harness there is no host to tell
                    pass
        if r.batch_stop:   # level up, game end or a wall hit twice: the host's guardrails decide
            if i < len(path) - 1:
                res.stopped, noop = r.batch_stop, r.batch_stop == "wall hit twice"
            break
    res.obs = obs
    if not quiet:
        head = f"play: {res.done} of {len(path)} moves done" + (f"; stopped: {res.stopped}" if res.stopped else "")
        out = [head, *res.lines]
        if res.last is not None:   # the host's notes after the last move (level up, GAME_OVER, ...)
            out += [ln for ln in res.last.text.splitlines()[1:] if ln and not ln.startswith("#")]
        if res.surprise:
            out += [_surprise_text(res, obs), SURPRISE_NOTE]
        elif noop:
            out.append(NOOP_NOTE)
        print("\n".join(out))
    return res

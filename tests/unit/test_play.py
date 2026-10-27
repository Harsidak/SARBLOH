# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME), _os.path.join(_ROOT, _PRIME, "harness", "runtime", "skills")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""``play()`` in the kernel skills (acting.py) against a fake host and a fake game:
- without ``predict`` the whole path is made, one ``act()`` per move, each tagged ``via: "play"``;
- a whole-board prediction that matches never stops; one that differs stops at that move, with the cells listed;
- HUD cells are not compared unless ``skip_hud=False``; a dict prediction checks only its cells; None skips a move;
- ``state=`` compares the model's own state with ``==``;
- it stops at a level up and at GAME_OVER, and a failing ``predict`` stops before its move (nothing spent);
- a refused first move raises; the printout has the head line, the surprise window and the fix note;
- a move with no prediction that changes nothing (HUD aside) stops the path ("no change"), unless
  ``stop_if_unchanged=False``; a surprise tells the host (``arc.surprise``);
- each move carries ``play_id`` and ``predicted``, the first one the path length (the host's cap);
- ``act()`` is gone: calling it raises and sends nothing.

Run: uv run python tests/unit/test_play.py
"""
import asyncio
import contextlib
import io
import json
import tempfile
from pathlib import Path

FAILS, CHECKS = [], [0]


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(f"{name}: {detail}")


class FakeGame:
    """A 3x6 board: row 0 is a HUD counter (Y per move made), the player R walks on row 1, a wall W at column 4 stops
    it. Reaching column 5 by "2" (a teleport) is a level up; "5" is GAME_OVER."""

    def __init__(self, session: Path) -> None:
        self.session, self.c, self.moves, self.requests, self.refuse = session, 0, 0, [], False
        self.surprises = []
        self.last_noop = None
        self.publish()

    def board(self) -> list[str]:
        hud = ("Y" * self.moves + "B" * 6)[:6]
        row = ["B"] * 6
        row[4] = "W"
        row[self.c] = "R"
        return [hud, "".join(row), "BBBBBB"]

    def publish(self) -> None:
        d = self.session / ".prime"
        d.mkdir(parents=True, exist_ok=True)
        objs = [{"id": 1, "letter": "Y", "hud": True, "bbox": [0, 0, 0, 5], "cells": None}]
        (d / "observation.json").write_text(json.dumps({"step": self.moves, "letters": self.board(), "objects": objs}))

    async def host_request(self, req):
        if req["type"] == "arc.surprise":
            self.surprises.append(req["args"])
            return {"status": "ok", "result": {}}
        self.requests.append(req)
        if self.refuse:
            return {"status": "error", "error": "not run, nothing spent"}
        (a,) = req["args"]["actions"]
        self.moves += 1
        stopped, level_up, state, c0 = None, False, "NOT_FINISHED", self.c
        if a == "4" and self.c + 1 != 4:
            self.c += 1
        elif a == "3" and self.c > 0:
            self.c -= 1
        elif a == "2":
            stopped, level_up = "level up", True
        elif a == "5":
            stopped, state = "GAME_OVER", "GAME_OVER"
        self.publish()
        # the host's batch_stop guardrail: the same move twice in a row with nothing changed is a wall hit
        noop = self.c == c0 and a in "34"
        batch_stop = stopped or ("wall hit twice" if noop and self.last_noop == a else None)
        self.last_noop = a if noop else None
        text = f"act: 1 of 1 actions done{'; stopped: ' + stopped if stopped else ''}\n#{self.moves} A{a}: moved"
        if level_up:
            text += "\nLevel 1 done. This is level 2."
        return {"status": "ok", "result": {"text": text, "done": 1, "stopped": stopped, "level_up": level_up,
                                           "state": state, "batch_stop": batch_stop}}


def walk(board, move):
    """A model that does not know the wall."""
    row = list(board[1])
    c = row.index("R")
    nc = c + 1 if move == "4" else max(0, c - 1) if move == "3" else c
    row[c], row[4] = "B", "W"
    row[nc] = "R"
    return [board[0], "".join(row), board[2]]


def run(coro):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = asyncio.run(coro)
    return res, out.getvalue()


def main() -> None:
    import acting

    with tempfile.TemporaryDirectory() as tmp:
        _os.environ["RLM_SESSION_DIR"] = tmp

        def fresh():
            g = FakeGame(Path(tmp))
            acting._repl.host_request = g.host_request
            return g

        g = fresh()
        r, _ = run(acting.play(["4", "4", "3"]))
        check("no predict: all moves made", r.done == 3 and r.stopped is None, repr(r))
        check("one act per move", len(g.requests) == 3 and all(len(q["args"]["actions"]) == 1 for q in g.requests))
        check("tagged via play", all(q["args"].get("via") == "play" for q in g.requests))
        check("memory sent with each move", all("memory" in q["args"] for q in g.requests))
        check("lines collected", len(r.lines) == 3 and r.lines[0].startswith("#1"), r.lines)
        check("path length on the first move only", g.requests[0]["args"].get("path_len") == 3 and
              all("path_len" not in q["args"] for q in g.requests[1:]), [q["args"] for q in g.requests])
        ids = {q["args"].get("play_id") for q in g.requests}
        check("one play_id per call", len(ids) == 1 and None not in ids, ids)
        check("not predicted", all(q["args"].get("predicted") is False for q in g.requests))

        g = fresh()
        r, out = run(acting.play(["4", "4", "4", "4", "4"], predict=walk))
        check("surprise does not stop", r.stopped is None and r.surprise["move"] == 3 and r.done == 5, repr(r))
        cells = r.surprise["cells"] if r.surprise else []
        check("surprise cells", sorted(cells) == [(1, 3, "B", "R"), (1, 4, "R", "W")], cells)
        check("hud skipped", all(rr != 0 for rr, *_ in cells), cells)
        check("printout", out.startswith("play: 5 of 5 moves done\n") and
              "surprise at move 3 (4): 2 cells differ" in out and "the board now:" in out, out)
        check("obs is the board after", r.obs.board[1] == "BBBRWB", r.obs.board)
        check("surprise told to the host", g.surprises == [{"move": 3, "action": "4"}], g.surprises)
        check("fix note printed", acting.SURPRISE_NOTE in out, out)
        check("predicted moves flagged", all(q["args"].get("predicted") is True for q in g.requests))

        g = fresh()
        r, _ = run(acting.play(["4", "3"], predict=walk, skip_hud=False))
        check("hud compared when asked", r.done == 2 and r.surprise["move"] == 0 and
              any(rr == 0 for rr, *_ in r.surprise["cells"]), repr(r))

        g = fresh()
        r, _ = run(acting.play(["4", "4", "4", "4"], predict=lambda b, m: {(1, b[1].index("R") + 1): "R"}))
        check("dict prediction", r.done == 4 and r.surprise["move"] == 3 and
              r.surprise["cells"] == [(1, 4, "R", "W")], repr(r))

        g = fresh()
        r, _ = run(acting.play(["4", "4", "3"], predict=lambda b, m: None if m == "4" else walk(b, m)))
        check("None skips the check", r.done == 3 and r.stopped is None, repr(r))

        g = fresh()
        r, out = run(acting.play(["4"] * 6))
        check("wall hit twice stops the path", r.stopped == "wall hit twice" and r.done == 5 and len(g.requests) == 5 and
              acting.NOOP_NOTE in out, repr(r))

        g = fresh()
        r, _ = run(acting.play(["4", "4", "4", "4", "3"]))
        check("one no-op does not stop", r.done == 5 and r.stopped is None, repr(r))

        g = fresh()
        r, _ = run(acting.play(["4", "4", "4", "3"], predict=lambda b, m: b if m == "4" and b[1][3] == "R" else
                               walk(b, m)))
        check("a predicted no-change is not a stop", r.done == 4 and r.stopped is None, repr(r))

        g = fresh()
        col = lambda obs: obs.board[1].index("R")
        r, _ = run(acting.play(["4", "4", "4", "4"], predict=lambda s, m: s + 1, state=col))
        check("state mode", r.done == 4 and r.surprise == {"move": 3, "action": "4", "predicted": 4,
                                                                       "actual": 3}, repr(r.surprise))

        g = fresh()
        r, out = run(acting.play(["4", "2", "4"], predict=walk))
        check("level up stops", r.stopped == "level up" and r.level_up and r.done == 2 and len(g.requests) == 2,
              repr(r))
        check("level up note printed", "This is level 2." in out, out)

        g = fresh()
        r, _ = run(acting.play(["5", "4"]))
        check("GAME_OVER stops", r.stopped == "GAME_OVER" and r.state == "GAME_OVER" and r.done == 1, repr(r))

        g = fresh()
        def bad(b, m):
            if m == "3":
                raise KeyError("x")
            return walk(b, m)
        r, _ = run(acting.play(["4", "3", "4"], predict=bad))
        check("predict error stops before the move", r.done == 1 and len(g.requests) == 1 and
              r.stopped.startswith("predict failed at move 1: KeyError"), repr(r))

        g = fresh()
        g.refuse = True
        try:
            run(acting.play(["4"]))
            check("refused first move raises", False, "no error")
        except RuntimeError as exc:
            check("refused first move raises", "nothing spent" in str(exc), str(exc))

        g = fresh()
        r, out = run(acting.play("4", quiet=True))
        check("single move and quiet", r.done == 1 and out == "", out)

        check("no act() stub", not hasattr(acting, "act"))

    print(f"play: {CHECKS[0] - len(FAILS)}/{CHECKS[0]} checks passed")
    for f in FAILS:
        print("  FAIL", f)
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

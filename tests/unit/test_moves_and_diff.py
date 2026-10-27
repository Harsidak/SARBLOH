# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME), _os.path.join(_ROOT, _PRIME, "harness", "runtime", "skills")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""``moves()``, ``obs.before`` and ``frame_diff()`` in the kernel skills (observation.py), and the host side:
- ``append_moves`` writes one line per step; ``moves()`` gives this level's moves oldest first (the moves after the
  last level up), ``moves(all_levels=True)`` the whole game, each with the fields of ``MOVE_FIELDS``;
- ``obs.before`` comes from observation.json;
- ``frame_diff`` finds a moved, a turned, a recoloured, an appeared and a disappeared object, reports no change as
  empty lists, defaults to ``obs.before`` -> the board now, and raises when there is no move yet;
- ``AgentSession._move_rows`` writes the move as a ``play()`` path does ("4", "6 r c", "reset") with both boards;
- the kernel imports ``moves`` and ``frame_diff`` and the prompt documents them.

Run: uv run python tests/unit/test_moves_and_diff.py
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

FAILS, CHECKS = [], [0]


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(f"{name}: {detail}")


def board(*rows):
    return list(rows)


def main() -> None:
    import observation as ob

    with tempfile.TemporaryDirectory() as tmp:
        _os.environ["RLM_SESSION_DIR"] = tmp
        check("no file, no moves", ob.moves() == [])
        a, b, c = board("BBB", "BRB"), board("BBB", "BBR"), board("BRB", "BBB")
        ob.append_moves(Path(tmp), [
            {"i": 0, "level": 0, "move": "4", "before": a, "after": b, "change": "x", "state": "PLAYING"},
            {"i": 1, "level": 0, "move": "5", "before": b, "after": c, "change": "y", "state": "PLAYING",
             "level_up": True},
            {"i": 2, "level": 1, "move": "6 0 1", "before": c, "after": a, "change": "z", "state": "PLAYING"},
        ])
        now = ob.moves()
        check("this level only", [m["i"] for m in now] == [2], now)
        check("fields", set(now[0]) == set(ob.MOVE_FIELDS) and now[0]["before"] == c and now[0]["after"] == a, now)
        check("whole game, oldest first", [m["i"] for m in ob.moves(all_levels=True)] == [0, 1, 2])

        d = Path(tmp) / ".prime"
        (d / "observation.json").write_text(json.dumps({"letters": a, "before": c}))
        check("obs.before", ob.snapshot().before == c)
        check("default diff", ob.frame_diff()["changed"] == 2)
        (d / "observation.json").write_text(json.dumps({"letters": a}))
        check("no before", ob.snapshot().before is None)
        try:
            ob.frame_diff()
            check("no move yet raises", False, "no error")
        except ValueError as exc:
            check("no move yet raises", "no move yet" in str(exc), str(exc))

    bg = "B" * 8
    before = board(bg, "BRRBBBBB", "BBBBBBBB", "BYYYBBBB", "BBBYBBBB", "BBBBBBGB", bg, "BBBBBBBO")
    after = board(bg, "BBBRRBBB", "BBBBBBBB", "BBBYBBBB", "BBBYBBBB", "BBYYBBPB", bg, "BBBBBBBB")
    r = ob.frame_diff(before, after)
    check("moved", r["moved"] == [{"letter": "R", "size": 2, "from": [1, 1, 1, 2], "to": [1, 3, 1, 4], "by": [0, 2]}],
          r)
    check("rotated", len(r["rotated"]) == 1 and r["rotated"][0]["letter"] == "Y" and r["rotated"][0]["turn"] in
          (90, 180, 270), r["rotated"])
    check("changed color", r["changed_color"] == [{"from_letter": "G", "to_letter": "P", "size": 1,
                                                   "bbox": [5, 6, 5, 6]}], r["changed_color"])
    check("disappeared", r["disappeared"] == [{"letter": "O", "size": 1, "bbox": [7, 7, 7, 7]}], r["disappeared"])
    check("appeared empty", r["appeared"] == [], r["appeared"])
    r = ob.frame_diff(after, before)
    check("appeared", {"letter": "O", "size": 1, "bbox": [7, 7, 7, 7]} in r["appeared"], r["appeared"])
    r = ob.frame_diff(before, before)
    check("no change", r["changed"] == 0 and not any(r[k] for k in ("moved", "rotated", "appeared")), r)
    check("observations accepted", ob.frame_diff(SimpleNamespace(board=before), after)["changed"] > 0)
    check("lists of letters accepted", ob.frame_diff([list(x) for x in before], after)["changed"] > 0)
    try:
        ob.frame_diff(before, after[:3])
        check("size mismatch raises", False, "no error")
    except ValueError:
        check("size mismatch raises", True)

    from harness.agent.agent import AgentSession
    g0, g1 = [[5, 5], [5, 8]], [[5, 8], [5, 5]]
    arc = SimpleNamespace(transitions=[
        {"action": 4, "x": None, "y": None, "before": g0, "frames": [g1]},
        {"action": 6, "x": 7, "y": 3, "before": g1, "frames": [g0, g1]},
        {"action": 0, "x": None, "y": None, "before": g1, "frames": [g0]},
    ])
    fake = SimpleNamespace(arc=arc, _letters=AgentSession._letters)
    rows = AgentSession._move_rows(fake, [{"i": 0, "level": 0}, {"i": 1, "level": 0}, {"i": 2, "level": 0}])
    check("move as a path writes it", [m["move"] for m in rows] == ["4", "6 3 7", "reset"], rows)
    check("boards as letters", rows[0]["before"] == ["BB", "BR"] and rows[1]["after"] == ["BR", "BB"], rows)
    check("board before", AgentSession._board_before(fake) == ["BR", "BB"])
    check("no move, no board before", AgentSession._board_before(SimpleNamespace(arc=SimpleNamespace(
        transitions=[]), _letters=AgentSession._letters)) is None)

    kernel = (Path(_ROOT) / _PRIME / "harness" / "runtime" / "kernel.py").read_text(encoding="utf-8")
    check("kernel imports them", "from observation import moves, frame_diff" in kernel)
    from harness.agent import prompts
    check("prompt documents them", "moves()" in prompts.TOOLS_DETAILED and "frame_diff(" in prompts.TOOLS_DETAILED and
          "obs.before" in prompts.TOOLS_DETAILED)

    print(f"moves_and_diff: {CHECKS[0] - len(FAILS)}/{CHECKS[0]} checks passed")
    for f in FAILS:
        print("  FAIL", f)
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

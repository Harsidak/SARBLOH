"""World model workbench, pre-imported in the REPL as ``wm``: write the game's rules as code, check them against
every recorded transition, then search them for a plan. Everything here is free: no environment action is spent.

    def step(grid, action, x=None, y=None):     # your model: the grid after `action` (return a new array)
        g = grid.copy(); ...; return g
    wm.register(step, name="v1", ignore=[(0, 60, 63, 63)])  # ignore: boxes (x0, y0, x1, y1) not compared (HUD, counter)
    r = await wm.check()          # replay every recorded transition; r.ok, r.passed, r.failed, print(r)
    wm.predict(grid, [1, 1, 4])   # predicted grids after each action
    path = wm.plan(start_grid, goal=lambda g: ..., actions=[1, 2, 3, 4])  # shortest action list in the model, or None

An abstract model is also allowed: ``wm.register(step, perceive=parse)`` where ``parse(grid) -> state`` and
``step(state, action, x=None, y=None) -> state``; check then compares ``step(parse(before)) == parse(after)``.
Once registered, the host checks every ``act`` step against the model and stops a batch at the first wrong
prediction. ``wm.objects(grid)``, ``wm.summarize_change(a, b)`` and ``wm.background(grid)`` are perception helpers.
"""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from perception import Obj, as_rows, background, diff_cells, objects, state_key, summarize_change  # noqa: F401

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

__all__ = ["register", "check", "predict", "plan", "current", "objects", "summarize_change", "background",
           "state_key", "Model", "CheckReport"]


@dataclass
class Model:
    step: Callable[..., Any]
    name: str = "model"
    perceive: Callable[[Any], Any] | None = None
    ignore: list[tuple[int, int, int, int]] = field(default_factory=list)
    note: str = ""
    version: int = 1
    last_check: "CheckReport | None" = None

    def run(self, state: Any, action: int, x: int | None = None, y: int | None = None) -> Any:
        # A deep copy: a step that edits a list-of-rows grid or a dict state in place must not change the parent.
        s = state.copy() if _np is not None and isinstance(state, _np.ndarray) else copy.deepcopy(state)
        return self.step(s, action, x, y) if action == 6 or x is not None else self.step(s, action)


_MODEL: Model | None = None


def _arr(grid: Any) -> Any:
    grid = getattr(grid, "grid", grid)
    if _np is not None and not isinstance(grid, _np.ndarray):
        return _np.array(grid, dtype=_np.int8)
    return grid


def register(step: Callable[..., Any], name: str = "model", perceive: Callable[[Any], Any] | None = None,
             ignore: list[tuple[int, int, int, int]] | None = None, note: str = "") -> Model:
    """Make ``step`` the current world model. Re-registering replaces it and bumps the version."""
    global _MODEL
    if not callable(step):
        raise TypeError("wm.register needs a function step(grid, action, x=None, y=None) -> grid")
    version = (_MODEL.version + 1) if _MODEL is not None else 1
    boxes = [tuple(int(v) for v in box) for box in (ignore or [])]
    _MODEL = Model(step=step, name=name, perceive=perceive, ignore=boxes, note=note, version=version)
    _log({"kind": "register", "name": name, "version": version, "abstract": perceive is not None,
          "ignore": _MODEL.ignore, "note": note[:300]})
    return _MODEL


def current() -> Model | None:
    return _MODEL


def _masked(grid: Any, ignore: list[tuple[int, int, int, int]]) -> list[list[int]]:
    rows = as_rows(grid)
    if not ignore:
        return rows
    rows = [list(r) for r in rows]
    for x0, y0, x1, y1 in ignore:
        for y in range(max(0, y0), min(len(rows), y1 + 1)):
            for x in range(max(0, x0), min(len(rows[y]), x1 + 1)):
                rows[y][x] = -1
    return rows


def _same(a: Any, b: Any) -> bool:
    """Equality that also works for numpy arrays nested in dicts, lists and tuples."""
    if _np is not None and (isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray)):
        return _np.array_equal(_np.asarray(a), _np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(u, v) for u, v in zip(a, b))
    return bool(a == b)


def _compare(model: Model, before: Any, action: int, x: int | None, y: int | None, after: Any) -> tuple[bool, str]:
    """(match, detail) for one transition. Anything the model does wrong is a failure, never an exception."""
    try:
        if model.perceive is not None:
            pred = model.run(model.perceive(_arr(before)), action, x, y)
            real = model.perceive(_arr(after))
            eq = _same(pred, real)
            return eq, "" if eq else f"predicted state {_short(pred)} but saw {_short(real)}"
        pred = model.run(_arr(before), action, x, y)
        if pred is None:
            return False, "model returned None (return the new grid)"
        if _np is not None and _np.asarray(pred, dtype=object).ndim != 2:
            return False, f"model returned {type(pred).__name__}, expected a 2-D grid"
        p, r = _masked(pred, model.ignore), _masked(after, model.ignore)
    except Exception as exc:  # noqa: BLE001 - a crashing model is a failing model
        return False, f"model raised {type(exc).__name__}: {exc}"
    if len(p) != len(r) or (p and len(p[0]) != len(r[0])):
        return False, f"predicted shape {len(p)}x{len(p[0]) if p else 0}, real {len(r)}x{len(r[0]) if r else 0}"
    wrong = diff_cells(p, r)
    if not wrong:
        return True, ""
    xs, ys = [w[0] for w in wrong], [w[1] for w in wrong]
    sample = ", ".join(f"({a},{b}) predicted {o} real {n}" for a, b, o, n in wrong[:4])
    return False, f"{len(wrong)} px wrong in x={min(xs)}..{max(xs)}, y={min(ys)}..{max(ys)}: {sample}"


def _short(v: Any, n: int = 160) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


@dataclass
class CheckReport:
    name: str
    version: int
    tested: int
    passed: int
    skipped: int
    failures: list[dict[str, Any]]
    full: bool = True      # covered the whole record (start=0, every level); only a full pass certifies

    @property
    def ok(self) -> bool:
        return self.tested > 0 and self.passed == self.tested

    @property
    def failed(self) -> int:
        return self.tested - self.passed

    def __repr__(self) -> str:
        head = (f"wm.check {self.name} v{self.version}: {self.passed}/{self.tested} transitions match"
                f"{' (PASS)' if self.ok else ''}; {self.skipped} skipped (resets, level changes)")
        lines = [head]
        for f in self.failures[:5]:
            xy = "" if f["x"] is None else f"({f['x']},{f['y']})"
            lines.append(f"  #{f['i']} A{f['action']}{xy} level {f['level']}: {f['detail']}")
        if len(self.failures) > 5:
            lines.append(f"  ... {len(self.failures) - 5} more failures in r.failures")
        return "\n".join(lines)


async def check(start: int = 0, level: int | None = None, include_level_changes: bool = False,
                model: Model | None = None) -> CheckReport:
    """Replay every recorded transition from index ``start`` (optionally only one ``level``) through the model.
    RESET steps and level-up steps are skipped unless ``include_level_changes``: they load a new layout."""
    import arc

    m = model or _MODEL
    if m is None:
        raise RuntimeError("no world model registered: call wm.register(step) first")
    ts = await arc.transitions(start)
    tested = passed = skipped = 0
    failures: list[dict[str, Any]] = []
    t0 = time.time()
    for t in ts:
        if level is not None and t["level"] != level:
            continue
        if t["action"] == 0 or (t["level_up"] and not include_level_changes):
            skipped += 1
            continue
        tested += 1
        match, detail = _compare(m, t["before"], t["action"], t["x"], t["y"], t["after"])
        if match:
            passed += 1
        else:
            failures.append({"i": t["i"], "action": t["action"], "x": t["x"], "y": t["y"], "level": t["level"],
                             "detail": detail})
    rep = CheckReport(m.name, m.version, tested, passed, skipped, failures, full=start == 0 and level is None)
    m.last_check = rep
    _log({"kind": "check", "name": m.name, "version": m.version, "tested": tested, "passed": passed,
          "skipped": skipped, "first_failure": failures[0]["i"] if failures else None,
          "duration_s": round(time.time() - t0, 2)})
    return rep


def predict(grid: Any, actions: list[Any], model: Model | None = None) -> list[Any]:
    """Predicted grids (or states) after each action. An action is an id, or (6, x, y) for a click."""
    m = model or _MODEL
    if m is None:
        raise RuntimeError("no world model registered: call wm.register(step) first")
    s = m.perceive(_arr(grid)) if m.perceive is not None else _arr(grid)
    out = []
    for a in actions:
        aid, x, y = _action(a)
        s = m.run(s, aid, x, y)
        out.append(s)
    return out


def _action(a: Any) -> tuple[int, int | None, int | None]:
    if isinstance(a, (tuple, list)):
        return int(a[0]), (int(a[1]) if len(a) > 1 else None), (int(a[2]) if len(a) > 2 else None)
    return int(a), None, None


def _key(s: Any) -> Any:
    if _np is not None and isinstance(s, _np.ndarray):
        return s.tobytes() + repr((s.shape, s.dtype.str)).encode()
    try:
        hash(s)
        return s
    except TypeError:
        return repr(s)


def plan(start: Any, goal: Callable[[Any], bool], actions: list[Any] | None = None, max_depth: int = 40,
         max_nodes: int = 200_000, timeout_s: float = 30.0, model: Model | None = None) -> list[Any] | None:
    """Breadth-first search in the model: the shortest action list from ``start`` to a state where ``goal(state)``
    is true, or None. ``actions`` defaults to [1, 2, 3, 4, 5, 7]; add clicks as (6, x, y). The result can be passed
    to ``act`` in batches of up to 5 (clicks as "6 x y")."""
    m = model or _MODEL
    if m is None:
        raise RuntimeError("no world model registered: call wm.register(step) first")
    if m.last_check is None or not m.last_check.ok:
        print("warning: this model has not passed wm.check(); a plan from it is a guess")
    acts = actions or [1, 2, 3, 4, 5, 7]
    s0 = m.perceive(_arr(start)) if m.perceive is not None else _arr(start)
    if goal(s0):
        return []
    parent: dict[Any, tuple[Any, Any] | None] = {_key(s0): None}
    frontier = deque([(s0, 0)])
    t0, nodes = time.time(), 0
    while frontier:
        s, d = frontier.popleft()
        if d >= max_depth:
            continue
        ks = _key(s)
        for a in acts:
            aid, x, y = _action(a)
            try:
                n = m.run(s, aid, x, y)
            except Exception:  # noqa: BLE001 - an action the model cannot apply is a dead edge
                continue
            if n is None:
                continue
            k = _key(n)
            if k in parent:
                continue
            parent[k] = (ks, a)
            nodes += 1
            if goal(n):
                path = [a]
                cur = ks
                while parent[cur] is not None:
                    cur, act = parent[cur]
                    path.append(act)
                path.reverse()
                _log({"kind": "plan", "found": True, "length": len(path), "nodes": nodes,
                      "duration_s": round(time.time() - t0, 2)})
                return path
            if nodes >= max_nodes or time.time() - t0 > timeout_s:
                _log({"kind": "plan", "found": False, "nodes": nodes, "reason": "limit"})
                print(f"wm.plan: stopped after {nodes} states ({time.time() - t0:.1f}s); raise max_nodes/timeout_s "
                      f"or use an abstract state")
                return None
            frontier.append((n, d + 1))
    _log({"kind": "plan", "found": False, "nodes": nodes, "reason": "exhausted"})
    return None


def as_act(path: list[Any]) -> list[str]:
    """A plan as ``act`` arguments: ["1", "1", "6 12 40"]."""
    return [" ".join(str(v) for v in a) if isinstance(a, (tuple, list)) else str(a) for a in path]


# --- host hooks ----------------------------------------------------------------------------------------------
_PENDING: list[dict[str, Any]] = []


def _log(event: dict[str, Any]) -> None:
    """Queue an event for the host's trace; the host drains the queue with ``_host_sync`` after cells and acts."""
    _PENDING.append({"t": round(time.time(), 2), **event})


def _host_sync() -> dict[str, Any]:
    """The registered model and the queued events (called by the host, not by the agent)."""
    out = json.loads(json.dumps(_PENDING, default=_plain))   # numpy values must not lose the queue
    _PENDING.clear()
    m = _MODEL
    return {"model": None if m is None else f"{m.name} v{m.version}", "certified": _certified(m), "events": out}


def _plain(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


def _certified(m: Model | None) -> bool:
    """A full passing check, not voided since by a wrong prediction on an act step."""
    return bool(m is not None and m.last_check is not None and m.last_check.ok and m.last_check.full)


async def _host_check_at(i: int) -> dict[str, Any]:
    """Compare the model's prediction with recorded transition ``i`` (called by the host after each act step)."""
    import arc

    if _MODEL is None:
        return {"model": None}
    ts = await arc.transitions(i)
    if not ts:
        return {"model": f"{_MODEL.name} v{_MODEL.version}", "match": None, "detail": f"no transition #{i}"}
    t = ts[0]
    if t["action"] == 0 or t["level_up"]:
        return {"model": f"{_MODEL.name} v{_MODEL.version}", "match": None, "detail": "reset or level change"}
    match, detail = _compare(_MODEL, t["before"], t["action"], t["x"], t["y"], t["after"])
    if not match:
        _MODEL.last_check = None   # the certificate no longer covers the record
    return {"model": f"{_MODEL.name} v{_MODEL.version}", "match": match, "detail": detail[:400]}

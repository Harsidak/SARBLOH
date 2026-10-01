"""ARC-AGI-3 environment interface, pre-imported in the Prime Agent REPL as ``arc``.

Every call is a typed host request: the host owns the game, counts actions and keeps the lossless frame
store, so the model can never act outside the budget it was given (least-privilege action interface).
Only ``step`` and ``reset`` spend environment actions; everything else is free.

    obs = await arc.observe()          # current observation (free)
    obs = await arc.step(1)            # ACTION1 (costs one action)
    obs = await arc.step(6, x=10, y=3) # ACTION6 at column x=10, row y=3
    obs = await arc.reset()            # RESET: restart the current level (costs one action, loses its progress)
    ts  = await arc.transitions()      # every step of this game, oldest first (free); ts[-1]["after"]
    print(arc.show(obs.grid))          # compact text rendering, one hex digit per pixel
    arc.diff(a, b)                     # [(x, y, old, new), ...] pixels that changed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm import host_request as _host_request

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

ACTION_NAMES = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6",
                7: "ACTION7"}
_HEX = "0123456789abcdef"


class ArcError(RuntimeError):
    """The host refused a request: illegal action, budget spent, game over, a subagent calling step, or a
    "harness limit" (a rule of this harness, not of the game)."""


async def host_request(kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return await _host_request(kind, payload)
    except RuntimeError as exc:
        raise ArcError(str(exc).split(": ", 1)[-1]) from None


def _grid(rows: Any) -> Any:
    return _np.array(rows, dtype=_np.int8) if _np is not None else rows


@dataclass
class Observation:
    grid: Any                      # last frame, numpy int8 array [y][x] (list of lists without numpy)
    frames: list                   # all frames the engine returned for this step (animations); grid == frames[-1]
    state: str                     # NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int
    win_levels: int
    available_actions: list[int]   # legal ids for arc.step (RESET is not listed: it is arc.reset())
    action_count: int              # environment actions spent so far in this game
    level_action_count: int        # actions spent on the current level
    actions_left: int | None       # remaining action budget (None = unlimited)
    level_up: bool                 # this step completed a level
    info: dict

    # Aliases for names models commonly guess.
    level = property(lambda self: self.levels_completed)
    levels = property(lambda self: self.levels_completed)
    actions = property(lambda self: self.action_count)
    frame = property(lambda self: self.grid)

    def __repr__(self) -> str:
        shape = getattr(self.grid, "shape", None) or (len(self.grid), len(self.grid[0]) if len(self.grid) else 0)
        return (f"Observation(state={self.state}, levels={self.levels_completed}/{self.win_levels}, "
                f"actions={self.action_count} (level {self.level_action_count}), left={self.actions_left}, "
                f"available={self.available_actions}, level_up={self.level_up}, grid={tuple(shape)}, "
                f"frames={len(self.frames)})")


def _obs(payload: dict[str, Any]) -> Observation:
    frames = [_grid(f) for f in payload["frames"]]
    return Observation(
        grid=frames[-1], frames=frames, state=payload["state"], levels_completed=payload["levels_completed"],
        win_levels=payload["win_levels"], available_actions=payload["available_actions"],
        action_count=payload["action_count"], level_action_count=payload["level_action_count"],
        actions_left=payload.get("actions_left"), level_up=payload.get("level_up", False),
        info=payload.get("info", {}),
    )


async def observe() -> Observation:
    """The current observation. Free: no environment action is spent."""
    return _obs(await host_request("arc.observe"))


async def step(action: int | str, x: int | None = None, y: int | None = None) -> Observation:
    """Commit one environment action. Costs one action against the score. ACTION6 needs x (column) and y (row)."""
    if isinstance(action, str):
        name = action.upper()
        action = next((k for k, v in ACTION_NAMES.items() if v == name), None)
        if action is None:
            raise ValueError(f"unknown action name; use one of {list(ACTION_NAMES.values())[1:]}")
    if int(action) == 0:
        raise ArcError("arc.step(0) is RESET, which restarts the level and loses its progress. Call "
                       "`await arc.reset()` if you mean that; game actions are the ids in obs.available_actions.")
    payload: dict[str, Any] = {"action": int(action)}
    if x is not None or y is not None:
        payload.update({"x": int(x), "y": int(y)})
    return _obs(await host_request("arc.step", payload))


async def reset() -> Observation:
    """RESET: restarts the current level. Costs one action; the level's state is lost, your variables and memories
    are kept. Needed after GAME_OVER. Refused when the level is already at its start."""
    return _obs(await host_request("arc.reset"))


class Transition(dict):
    """One recorded step: a dict with keys i, action, x, y, level, before, after, frames, state, level_up, t.
    ``t["after"]`` and ``t.after`` both work."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"transition has no field {name!r}; fields: {sorted(self)}") from None

    def __missing__(self, key: Any) -> Any:
        raise KeyError(f"{key!r}: a transition is a dict, not a tuple; use t['before'], t['action'], t['x'], t['y'], "
                       f"t['after'], t['level'], t['state'], t['level_up']")


async def transitions(start: int = 0) -> list[Transition]:
    """Recorded steps of this game from index ``start``, oldest first. Free. Each is a dict with keys
    action, x, y, level, before (grid), after (grid), frames (all frames of the step), state, level_up."""
    res = await host_request("arc.transitions", {"start": int(start)})
    out = []
    for t in res["transitions"]:
        t = Transition(t)
        t["before"] = _grid(t["before"])
        t["frames"] = [_grid(f) for f in t["frames"]]
        t["after"] = t["frames"][-1]
        out.append(t)
    return out


def show(grid: Any, x0: int = 0, y0: int = 0, x1: int | None = None, y1: int | None = None) -> str:
    """Render a grid (or the window [y0:y1, x0:x1]) as text: one hex digit per pixel, row index on the left."""
    rows = grid.tolist() if hasattr(grid, "tolist") else grid
    rows = rows[y0:y1]
    width = len(rows[0][x0:x1]) if rows else 0
    lines = [f"{'':3} " + "".join(str((x0 + i) // 10 % 10) for i in range(width)),
             f"{'':3} " + "".join(str((x0 + i) % 10) for i in range(width))]
    for j, row in enumerate(rows):
        lines.append(f"{y0 + j:3d} " + "".join(_HEX[int(c) & 15] for c in row[x0:x1]))
    return "\n".join(lines)


def diff(a: Any, b: Any) -> list[tuple[int, int, int, int]]:
    """Pixels that differ between two grids (or two observations) of the same shape, as (x, y, old, new)."""
    a, b = getattr(a, "grid", a), getattr(b, "grid", b)
    ra = a.tolist() if hasattr(a, "tolist") else a
    rb = b.tolist() if hasattr(b, "tolist") else b
    return [(x, y, int(ra[y][x]), int(rb[y][x])) for y in range(len(ra)) for x in range(len(ra[y]))
            if ra[y][x] != rb[y][x]]


def give_up(reason: str = "") -> None:
    """Not provided: the run ends on WIN or when the host budget is spent. Keep making progress."""
    raise RuntimeError("give_up is not available; keep working within the budget")

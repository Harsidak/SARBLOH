"""ARC-AGI-3 environment interface, pre-imported in the Prime Agent REPL as ``arc``.

Every call is a typed host request: the host owns the game, counts actions and keeps the lossless frame
store, so the model can never act outside the budget it was given (least-privilege action interface).
Only ``step`` and ``reset`` spend environment actions; everything else is free.

    obs = await arc.observe()          # current observation (free)
    obs = await arc.step(1)            # ACTION1 (costs one action)
    obs = await arc.step(6, x=10, y=3) # ACTION6 at column x=10, row y=3
    obs = await arc.reset()            # RESET (costs one action)
    ts  = await arc.transitions()      # every recorded (before, action, after) of this game (free)
    print(arc.show(obs.grid))          # compact text rendering, one hex digit per cell
    arc.diff(a, b)                     # [(x, y, old, new), ...] cells that changed
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
    """The host refused a request (illegal action, budget spent, game over, subagent calling step)."""


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
    available_actions: list[int]   # legal action ids; 0 = RESET
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
            raise ValueError(f"unknown action name; use one of {list(ACTION_NAMES.values())}")
    payload: dict[str, Any] = {"action": int(action)}
    if x is not None or y is not None:
        payload.update({"x": int(x), "y": int(y)})
    return _obs(await host_request("arc.step", payload))


async def reset() -> Observation:
    """RESET: restarts the current level (costs one action)."""
    return await step(0)


async def transitions(start: int = 0) -> list[dict]:
    """Recorded transitions of this game from index ``start``: dicts with action, x, y, level,
    before (grid), after (last frame), frames (all frames), state, level_up. Free."""
    res = await host_request("arc.transitions", {"start": int(start)})
    out = []
    for t in res["transitions"]:
        t = dict(t)
        t["before"] = _grid(t["before"])
        t["frames"] = [_grid(f) for f in t["frames"]]
        t["after"] = t["frames"][-1]
        out.append(t)
    return out


def show(grid: Any, x0: int = 0, y0: int = 0, x1: int | None = None, y1: int | None = None) -> str:
    """Render a grid (or the window [y0:y1, x0:x1]) as text: one hex digit per cell, row index on the left."""
    rows = grid.tolist() if hasattr(grid, "tolist") else grid
    rows = rows[y0:y1]
    width = len(rows[0][x0:x1]) if rows else 0
    lines = [f"{'':3} " + "".join(str((x0 + i) // 10 % 10) for i in range(width)),
             f"{'':3} " + "".join(str((x0 + i) % 10) for i in range(width))]
    for j, row in enumerate(rows):
        lines.append(f"{y0 + j:3d} " + "".join(_HEX[int(c) & 15] for c in row[x0:x1]))
    return "\n".join(lines)


def diff(a: Any, b: Any) -> list[tuple[int, int, int, int]]:
    """Cells that differ between two grids of the same shape, as (x, y, old, new)."""
    ra = a.tolist() if hasattr(a, "tolist") else a
    rb = b.tolist() if hasattr(b, "tolist") else b
    return [(x, y, int(ra[y][x]), int(rb[y][x])) for y in range(len(ra)) for x in range(len(ra[y]))
            if ra[y][x] != rb[y][x]]


def give_up(reason: str = "") -> None:
    """Not provided: the run ends on WIN or when the host budget is spent. Keep making progress."""
    raise RuntimeError("give_up is not available; keep working within the budget")

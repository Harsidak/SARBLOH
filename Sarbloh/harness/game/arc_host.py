"""Host side of the ``arc`` kernel module: owns one ``harness.game.games.ArcGame``.

The host is the only thing that touches the environment. It validates every action, enforces the action budget,
keeps the lossless transition record (the frame store the model retrieves from with ``arc.transitions()``), and
refuses ``arc.step`` from subagents. It also stamps every action with the agent step that spent it (``step_ref``),
in ``run.history`` and, when the game records, in the ARC SDK recording's ``reasoning`` field.

E006 (``toolset: "dedicated"``): the root spends actions only through the ``act``/``reset_level`` tools
(``repl_actions = False`` refuses ``arc.step``/``arc.reset`` from the REPL). Every step gets a state key of the grid
before it, so a repeated (state, action) pair is flagged, and an object-level change summary; both go to ``on_step``
(the session's transcript) whatever path spent the action.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

import arcengine

from harness.runtime.skills.perception import state_key, summarize_change
from harness.game.games import ArcGame, GameState


_REF_KEYS = ("session", "turn", "call", "after_cell")
_REASONING_MAX = 15_000  # the SDK refuses a step whose reasoning is over 16 KB (arcengine MAX_REASONING_BYTES)


def _reasoning(ref: dict[str, Any], ctx: dict[str, Any] | None) -> dict[str, Any]:
    """The SDK ``reasoning`` blob: the ref, plus the step's thought/text/code on a cell's first action."""
    blob = dict(ref)
    for key, cap in (("code", 6000), ("say", 2000), ("thought", 6000)):
        if ctx and ctx.get(key):
            text = str(ctx[key])
            blob[key] = text[-cap:] if key == "thought" else text[:cap]  # a thought's conclusion is at its end
    while len(json.dumps(blob, separators=(",", ":"))) > _REASONING_MAX:
        key = max((k for k in ("thought", "code", "say") if k in blob), key=lambda k: len(blob[k]), default=None)
        if key is None:
            break
        blob[key] = blob[key][len(blob[key]) // 2:] if key == "thought" else blob[key][:len(blob[key]) // 2]
    return blob


def _frames(state: GameState) -> list[list[list[int]]]:
    out = []
    for f in state.raw.frame:
        out.append(f.tolist() if hasattr(f, "tolist") else [list(map(int, r)) for r in f])
    return out


class ArcHost:
    def __init__(self, game: ArcGame, max_actions: int | None, should_stop: Callable[[], bool],
                 tokens_spent: Callable[[], int], max_actions_per_cell: int | None = None,
                 stop_after_levels: int | None = None) -> None:
        self.game = game
        self.stop_after_levels = stop_after_levels   # end the game early after this many levels (local tests)
        self.repl_actions = True                     # False: the root acts only through the act/reset_level tools
        self.on_step: Callable[[dict[str, Any]], None] | None = None
        self._seen: dict[tuple, int] = {}            # (state key, action, x, y) -> first transition index
        self.last_step: dict[str, Any] | None = None  # the step event of the latest action
        self.max_actions = max_actions
        self.should_stop = should_stop
        self.tokens_spent = tokens_spent
        self.transitions: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._level_start = 0          # action_count when the current level started
        # Actions since the level began or was last reset. Mirrors the engine's `_action_count`, which `set_level()`
        # zeroes; a RESET at 0 changes nothing (ONLY_RESET_LEVELS) or restarts the whole game from level 1 (without it).
        self._since_reset = 0
        self._token_mark = 0
        self.max_actions_per_cell = max_actions_per_cell
        self.cell_actions = 0          # reset by the session before each cell
        self.cell_cap_hits = 0
        self.reflection_due: str | None = None   # set by the session; arc.step is refused while set
        # Set by the root session before each cell: session, turn and tool-call id, plus the model's thought, text
        # and code. Every action gets the ref; the cell's first action also carries the context in the recording.
        self.step_ref: dict[str, Any] | None = None

    # --- status ------------------------------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        run = self.game.run
        if self.stop_after_levels is not None and self.game.state.levels_completed >= self.stop_after_levels:
            return True
        return run is None or run.state != "playing" or self.game.state.won

    @property
    def budget_left(self) -> int | None:
        return None if self.max_actions is None else max(0, self.max_actions - self.game.action_count)

    def status_line(self) -> str:
        s = self.game.state
        return (f"level {s.levels_completed}/{self.game.number_of_levels}, state {s.engine_state.name}, "
                f"{self.game.action_count} actions spent, {self.budget_left if self.max_actions else 'unlimited'} left")

    def actions_line(self, start: int, limit: int = 12) -> str:
        """The actions from ``run.history[start]`` on, run-length compressed: ``A1x3 A6(12,40) RESET``."""
        names = [r.action.replace("ACTION", "A") + (f"({r.data.get('x')},{r.data.get('y')})" if r.data else "")
                 for r in (self.game.run.history[start:] if self.game.run else [])]
        runs: list[list[Any]] = []
        for n in names:
            if runs and runs[-1][0] == n:
                runs[-1][1] += 1
            else:
                runs.append([n, 1])
        parts = [n + (f"x{c}" if c > 1 else "") for n, c in runs]
        return " ".join(parts[:limit]) + (f" ...+{len(parts) - limit}" if len(parts) > limit else "")

    def observation(self, level_up: bool = False) -> dict[str, Any]:
        s = self.game.state
        return {
            "frames": _frames(s),
            "state": s.engine_state.name,
            "levels_completed": s.levels_completed,
            "win_levels": self.game.number_of_levels,
            # RESET (id 0) is left out: it is only `arc.reset()`. In the E005 smoke `for a in range(5): step(a)` spent
            # 100 of 150 actions on silent resets.
            "available_actions": [a for a in s.available_actions if a != 0],
            "action_count": self.game.action_count,
            "level_action_count": self.game.action_count - self._level_start,
            "actions_left": self.budget_left,
            "level_up": level_up,
            "info": {"game_id": self.game.game_id},
        }

    # --- host requests -----------------------------------------------------------------------------------
    def handle(self, req: dict[str, Any], depth: int, source: str = "repl") -> dict[str, Any]:
        """``source`` is "repl" for kernel requests and "tool" for the act/reset_level tools."""
        kind = req.get("type")
        with self._lock:
            if kind == "arc.observe":
                return self.observation()
            if kind == "arc.transitions":
                start = max(0, int(req.get("start", 0)))
                return {"transitions": self.transitions[start:]}
            if kind in ("arc.step", "arc.reset"):
                if depth > 0:
                    raise PermissionError("subagents cannot spend environment actions; report to your parent")
                if source == "repl" and not self.repl_actions:
                    raise PermissionError("in this harness the REPL cannot spend actions: use the `act` tool (with "
                                          "`expect`) for game actions and `reset_level` to restart the level")
                if kind == "arc.reset":
                    if self._since_reset == 0 and self.game.state.engine_state.name != "GAME_OVER":
                        raise ValueError("arc.reset() refused: the level is already at its start (no action since it "
                                         "began or was last reset), so a reset would cost an action and change nothing.")
                    return self._step({"action": 0}, source)
                if int(req.get("action", -1)) == 0:
                    raise ValueError("arc.step(0) is RESET, which restarts the level and loses its progress. Call "
                                     "`await arc.reset()` if you mean that; game actions are the ids in "
                                     "obs.available_actions.")
                return self._step(req, source)
        raise ValueError(f"unknown arc request {kind!r}")

    def _step(self, req: dict[str, Any], source: str = "repl") -> dict[str, Any]:
        if self.finished:
            raise RuntimeError(f"the game is over ({self.status_line()})")
        if self.should_stop():
            raise RuntimeError("the run is stopping (time budget spent); no more actions")
        if self.budget_left == 0:
            raise RuntimeError(f"action budget exhausted ({self.max_actions})")
        if self.reflection_due:
            raise RuntimeError(f"harness limit: reflection checkpoint pending ({self.reflection_due}). Write or update "
                               "a memory with rlm.harness before the next arc.step. This is a harness rule, not a "
                               "game rule.")
        if self.max_actions_per_cell and self.cell_actions >= self.max_actions_per_cell:
            self.cell_cap_hits += 1
            raise RuntimeError(f"harness limit: at most {self.max_actions_per_cell} arc.step/arc.reset calls per "
                               "ipython call, and this call has used them. Read the results, then act in a new "
                               "ipython call. This is a harness rule, not a game rule.")
        action_id = int(req["action"])
        legal = [a for a in self.game.state.available_actions if a != 0]  # as in observation(): RESET is arc.reset()
        try:
            ga = arcengine.GameAction.from_id(action_id)  # GameAction(int) raises: the enum values are not plain
        except (ValueError, KeyError):
            raise ValueError(f"unknown action id {action_id}; legal: {legal}") from None
        data: dict[str, Any] = {}
        if ga == arcengine.GameAction.ACTION6:
            if req.get("x") is None or req.get("y") is None:
                raise ValueError("ACTION6 needs x (column) and y (row)")
            data = {"x": max(0, min(63, int(req["x"]))), "y": max(0, min(63, int(req["y"])))}
        if action_id not in self.game.state.available_actions:
            raise ValueError(f"{ga.name} is not available now; legal: {legal}")
        before = self.game.state
        self.cell_actions += 1
        tokens = self.tokens_spent()
        ref = {k: v for k, v in (self.step_ref or {}).items() if k in _REF_KEYS}
        ref["k"] = self.cell_actions  # 1-based index of this action within its cell
        reasoning = _reasoning(ref, self.step_ref if self.cell_actions == 1 else None) if self.game.record else None
        new = self.game.execute_action(arcengine.ActionInput(id=ga, data=data),
                                       generated_tokens=max(0, tokens - self._token_mark), reasoning=reasoning,
                                       ref=ref)
        self._token_mark = tokens
        level_up = new.just_won_level
        self._since_reset = 0 if action_id == 0 or level_up else self._since_reset + 1
        before_grid, frames = _frames(before)[-1], _frames(new)
        i = len(self.transitions)
        pair = (state_key(before_grid), action_id, data.get("x"), data.get("y"))
        repeat_of = self._seen.get(pair)
        self._seen.setdefault(pair, i)
        self.transitions.append({
            "i": i, "action": action_id, "x": data.get("x"), "y": data.get("y"),
            "level": before.levels_completed, "before": before_grid, "frames": frames,
            "state": new.engine_state.name, "level_up": level_up, "t": round(time.time(), 2),
            "repeat_of": repeat_of,
        })
        if level_up:
            self._level_start = self.game.action_count
        try:
            change = summarize_change(before_grid, frames[-1])
        except Exception as exc:  # noqa: BLE001 - a summary must never fail a step
            change = {"changed": -1, "text": f"(summary failed: {exc})", "moves": []}
        self.last_step = {"event": "step", "i": i, "action": action_id, "x": data.get("x"), "y": data.get("y"),
                          "level": before.levels_completed, "level_after": new.levels_completed,
                          "state": new.engine_state.name, "level_up": level_up, "frames": len(frames),
                          "changed": change["changed"], "change": change["text"], "repeat_of": repeat_of,
                          "state_key": pair[0], "source": source,
                          **{k: v for k, v in ref.items() if k in ("turn", "call", "k")}}
        if self.on_step is not None:
            try:
                self.on_step(self.last_step)
            except Exception:  # noqa: BLE001
                pass
        return self.observation(level_up=level_up)

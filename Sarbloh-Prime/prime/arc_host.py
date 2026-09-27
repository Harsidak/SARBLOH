"""Host side of the ``arc`` kernel module: owns one ``sarbloh.harness.games.ArcGame``.

The host is the only thing that touches the environment. It validates every action, enforces the action budget,
keeps the lossless transition record (the frame store the model retrieves from with ``arc.transitions()``), and
refuses ``arc.step`` from subagents.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import arcengine

from sarbloh.harness.games import ArcGame, GameState


def _frames(state: GameState) -> list[list[list[int]]]:
    out = []
    for f in state.raw.frame:
        out.append(f.tolist() if hasattr(f, "tolist") else [list(map(int, r)) for r in f])
    return out


class ArcHost:
    def __init__(self, game: ArcGame, max_actions: int | None, should_stop: Callable[[], bool],
                 tokens_spent: Callable[[], int]) -> None:
        self.game = game
        self.max_actions = max_actions
        self.should_stop = should_stop
        self.tokens_spent = tokens_spent
        self.transitions: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._level_start = 0          # action_count when the current level started
        self._token_mark = 0

    # --- status ------------------------------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        run = self.game.run
        return run is None or run.state != "playing" or self.game.state.won

    @property
    def budget_left(self) -> int | None:
        return None if self.max_actions is None else max(0, self.max_actions - self.game.action_count)

    def status_line(self) -> str:
        s = self.game.state
        return (f"level {s.levels_completed}/{self.game.number_of_levels}, state {s.engine_state.name}, "
                f"{self.game.action_count} actions spent, {self.budget_left if self.max_actions else 'unlimited'} left")

    def observation(self, level_up: bool = False) -> dict[str, Any]:
        s = self.game.state
        return {
            "frames": _frames(s),
            "state": s.engine_state.name,
            "levels_completed": s.levels_completed,
            "win_levels": self.game.number_of_levels,
            "available_actions": s.available_actions,
            "action_count": self.game.action_count,
            "level_action_count": self.game.action_count - self._level_start,
            "actions_left": self.budget_left,
            "level_up": level_up,
            "info": {"game_id": self.game.game_id},
        }

    # --- host requests -----------------------------------------------------------------------------------
    def handle(self, req: dict[str, Any], depth: int) -> dict[str, Any]:
        kind = req.get("type")
        with self._lock:
            if kind == "arc.observe":
                return self.observation()
            if kind == "arc.transitions":
                start = max(0, int(req.get("start", 0)))
                return {"transitions": self.transitions[start:]}
            if kind == "arc.step":
                if depth > 0:
                    raise PermissionError("subagents cannot spend environment actions; report to your parent")
                return self._step(req)
        raise ValueError(f"unknown arc request {kind!r}")

    def _step(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.finished:
            raise RuntimeError(f"the game is over ({self.status_line()})")
        if self.should_stop():
            raise RuntimeError("the run is stopping (time budget spent); no more actions")
        if self.budget_left == 0:
            raise RuntimeError(f"action budget exhausted ({self.max_actions})")
        action_id = int(req["action"])
        try:
            ga = arcengine.GameAction.from_id(action_id)  # GameAction(int) raises: the enum values are not plain
        except (ValueError, KeyError):
            raise ValueError(f"unknown action id {action_id}; legal: {self.game.state.available_actions}") from None
        data: dict[str, Any] = {}
        if ga == arcengine.GameAction.ACTION6:
            if req.get("x") is None or req.get("y") is None:
                raise ValueError("ACTION6 needs x (column) and y (row)")
            data = {"x": max(0, min(63, int(req["x"]))), "y": max(0, min(63, int(req["y"])))}
        if action_id not in self.game.state.available_actions:
            raise ValueError(f"{ga.name} is not available now; legal: {self.game.state.available_actions}")
        before = self.game.state
        tokens = self.tokens_spent()
        new = self.game.execute_action(arcengine.ActionInput(id=ga, data=data),
                                       generated_tokens=max(0, tokens - self._token_mark))
        self._token_mark = tokens
        level_up = new.just_won_level
        self.transitions.append({
            "i": len(self.transitions), "action": action_id, "x": data.get("x"), "y": data.get("y"),
            "level": before.levels_completed, "before": _frames(before)[-1], "frames": _frames(new),
            "state": new.engine_state.name, "level_up": level_up, "t": round(time.time(), 2),
        })
        if level_up:
            self._level_start = self.game.action_count
        return self.observation(level_up=level_up)

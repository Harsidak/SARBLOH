"""Games: one ``ArcGame`` per environment, built from an ``arc_agi.Arcade`` in OFFLINE or COMPETITION mode.

Replaces taaf's Game / GameAPI / GameRun with the same semantics (scoring, level bookkeeping, the shared
competition scorecard), minus the pickling, diagnostics and deployment layers. OFFLINE and COMPETITION go
through this one code path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import arc_agi
import arcengine

# Without a logger= kwarg arc_agi installs its own stdout INFO handler; pass a quiet private one.
_ARCADE_LOGGER = logging.getLogger("sarbloh.harness.arcade")
_ARCADE_LOGGER.setLevel(logging.WARNING)
logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)

RunState = Literal["playing", "won", "gave_up", "cancelled", "crashed"]


@dataclass(frozen=True)
class GameState:
    """One engine observation. ``grid`` is the last (visible) frame."""

    raw: arcengine.FrameDataRaw
    just_won_level: bool = False

    @property
    def grid(self) -> tuple[tuple[int, ...], ...]:
        data = self.raw.frame[-1]
        rows = data.tolist() if hasattr(data, "tolist") else data
        return tuple(tuple(int(c) for c in row) for row in rows)

    @property
    def engine_state(self) -> arcengine.GameState:
        return self.raw.state

    @property
    def levels_completed(self) -> int:
        return int(self.raw.levels_completed)

    @property
    def available_actions(self) -> list[int]:
        """Legal action ids, RESET (0) always included."""
        raw = [int(a) for a in self.raw.available_actions]
        return raw if 0 in raw else [0, *raw]

    @property
    def won(self) -> bool:
        return self.raw.state == arcengine.GameState.WIN

    @property
    def game_over(self) -> bool:
        """Recoverable dead end (RESET recovers). Not the end of the run."""
        return self.raw.state == arcengine.GameState.GAME_OVER


@dataclass
class ActionRecord:
    action: str
    data: dict[str, Any]
    level: int
    generated_tokens: int
    wallclock_s: float


@dataclass
class GameRun:
    game_id: str
    number_of_levels: int
    baseline_actions: list[int] | None
    state: RunState = "playing"
    history: list[ActionRecord] = field(default_factory=list)
    actions_per_level: list[int] = field(default_factory=list)
    levels_completed: int = 0
    started_at: str = ""
    final_score: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.actions_per_level:
            self.actions_per_level = [0] * self.number_of_levels

    def compute_score(self) -> float:
        """ARC-AGI-3 per-game score (0-100), mirroring arc_agi's EnvironmentScoreCalculator.

        Level i (weight i+1) scores min(115, (baseline/actions)^2 * 100) if completed, else 0. The total is the
        weighted mean, capped at (weights of scoring levels / all weights) * 100.
        """
        if self.baseline_actions is None or self.number_of_levels == 0:
            return 0.0
        total = 0.0
        total_w = 0
        max_w = 0
        for i in range(self.number_of_levels):
            w = i + 1
            total_w += w
            acts = self.actions_per_level[i] if i < len(self.actions_per_level) else 0
            s = min(115.0, (self.baseline_actions[i] / acts) ** 2 * 100) if i < self.levels_completed and acts > 0 else 0.0
            if s > 0:
                max_w += w
            total += s * w
        return 0.0 if total_w == 0 else min(total / total_w, max_w / total_w * 100)

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "state": self.state,
            "number_of_levels": self.number_of_levels,
            "levels_completed": self.levels_completed,
            "baseline_actions": self.baseline_actions,
            "actions_per_level": self.actions_per_level,
            "actions": len(self.history),
            "tokens": sum(r.generated_tokens for r in self.history),
            "final_score": self.final_score,
            "started_at": self.started_at,
            "note": self.note,
            "history": [
                {"a": r.action, "d": r.data, "lvl": r.level, "tok": r.generated_tokens, "t": round(r.wallclock_s, 2)}
                for r in self.history
            ],
        }


@dataclass
class _SharedScorecard:
    """COMPETITION mode allows one scorecard: open once, close after the last game finishes."""

    arcade: arc_agi.Arcade
    scorecard_id: str | None = None
    active: int = 0
    closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def open_run(self) -> str:
        with self._lock:
            if self.closed:
                raise RuntimeError("competition scorecard already closed")
            if self.scorecard_id is None:
                self.scorecard_id = self.arcade.create_scorecard()
            self.active += 1
            return self.scorecard_id

    def finish_run(self) -> None:
        with self._lock:
            if self.scorecard_id is None or self.closed:
                return
            self.active -= 1
            if self.active <= 0:
                self.closed = True
                self.arcade.close_scorecard(self.scorecard_id)


class ArcGame:
    """One environment played once. The agent sees ``state``, ``run``, ``execute_action`` and ``finish``."""

    def __init__(self, arcade: arc_agi.Arcade, env_name: str, shared: _SharedScorecard | None) -> None:
        self.arcade = arcade
        self.env_name = env_name
        self._shared = shared
        self._scorecard_id: str | None = None
        self.env: Any = None
        self.run: GameRun | None = None
        self._state: GameState | None = None
        self._t0 = 0.0
        self.number_of_levels = 0

    @property
    def game_id(self) -> str:
        return self.run.game_id if self.run else self.env_name

    @property
    def state(self) -> GameState:
        assert self._state is not None, "start() first"
        return self._state

    @property
    def action_count(self) -> int:
        return len(self.run.history) if self.run else 0

    def start(self) -> GameState:
        self._scorecard_id = self._shared.open_run() if self._shared else self.arcade.create_scorecard()
        try:
            env = self.arcade.make(self.env_name, scorecard_id=self._scorecard_id)
            if env is None or env.observation_space is None:
                raise RuntimeError(f"Arcade.make({self.env_name!r}) returned no environment")
        except Exception:
            if self._shared:
                self._shared.finish_run()
            raise
        self.env = env
        # A mid-game RESET must keep the current level (see taaf.game_api for the engine detail).
        os.environ["ONLY_RESET_LEVELS"] = "true"
        info = env.environment_info
        initial = env.observation_space
        self.number_of_levels = int(initial.win_levels)
        baseline = list(info.baseline_actions) if info.baseline_actions else None  # hidden in COMPETITION
        self.run = GameRun(
            game_id=info.game_id,
            number_of_levels=self.number_of_levels,
            baseline_actions=baseline,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._t0 = time.monotonic()
        self._state = GameState(raw=initial)
        return self._state

    def execute_action(self, action: arcengine.ActionInput, generated_tokens: int = 0) -> GameState:
        assert self.run is not None and self.run.state == "playing", "game is not playing"
        if action.id.value not in self.state.available_actions:
            raise ValueError(f"{action.id.name} not in available actions {self.state.available_actions}")
        level_before = self.state.levels_completed
        resp = self.env.step(action.id, data=dict(action.data))
        if resp is None or not resp.frame:
            raise RuntimeError(f"engine returned no frame for {action.id.name} (non-RESET after GAME_OVER?)")
        new_state = GameState(raw=resp, just_won_level=int(resp.levels_completed) > level_before)
        run = self.run
        run.history.append(
            ActionRecord(
                action=action.id.name,
                data=dict(action.data),
                level=level_before,
                generated_tokens=int(generated_tokens),
                wallclock_s=time.monotonic() - self._t0,
            )
        )
        run.actions_per_level[min(level_before, self.number_of_levels - 1)] += 1
        run.levels_completed = max(run.levels_completed, new_state.levels_completed)
        if new_state.won:
            run.state = "won"
        self._state = new_state
        return new_state

    def finish(self, state: RunState | None = None) -> None:
        """Idempotent. Closes this game's scorecard (or releases the shared one) and fixes the score."""
        run = self.run
        if run is None or run.final_score is not None:
            return
        if state is not None and run.state == "playing":
            run.state = state
        elif run.state == "playing":
            run.state = "gave_up"
        try:
            if self._shared:
                self._shared.finish_run()
            elif self._scorecard_id is not None:
                self.arcade.close_scorecard(self._scorecard_id)
        except Exception as exc:  # noqa: BLE001 - finishing must never raise
            run.note = (run.note or "") + f" scorecard_close_error={exc!r}"
        run.final_score = run.compute_score()
        per_level = ",".join(
            f"{run.actions_per_level[i]}/{run.baseline_actions[i] if run.baseline_actions else '?'}"
            for i in range(run.number_of_levels)
        )
        print(
            f"[finished] {run.game_id} state={run.state} level={run.levels_completed}/{run.number_of_levels} "
            f"score={run.final_score:.2f} actions={len(run.history)} per-level={per_level}"
            + (f' note="{run.note}"' if run.note else ""),
            flush=True,
        )


def make_arcade(mode: str, *, environments_dir: str = "", base_url: str = "") -> arc_agi.Arcade:
    if mode == "competition":
        return arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=base_url,
            environments_dir="",
            logger=_ARCADE_LOGGER,
        )
    if mode == "offline":
        return arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=environments_dir,
            logger=_ARCADE_LOGGER,
        )
    raise ValueError(f"unknown mode {mode!r}")


def build_games(arcade: arc_agi.Arcade, mode: str, only: list[str] | None = None) -> list[ArcGame]:
    """One ``ArcGame`` per available environment. ``only`` filters by exact id or id prefix (e.g. "ls20")."""
    ids = [e.game_id for e in arcade.available_environments]
    if only:
        ids = [g for g in ids if any(g == o or g.split("-")[0] == o for o in only)]
    if not ids:
        raise RuntimeError(f"no environments to play (mode={mode}, filter={only})")
    shared = _SharedScorecard(arcade) if mode == "competition" else None
    return [ArcGame(arcade, gid, shared) for gid in ids]

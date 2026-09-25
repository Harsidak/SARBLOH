"""Play one game with the Duck ToolAgent: the port of Tufa's ``_HarnessGameSession`` (framework/solver.py).

Behaviour is kept identical to the Duck (auto-RESET on GAME_OVER, batched actions stop at a level boundary,
retry on retryable analyzer failures, per-game wall-clock budget). The viewer-event sidecars are dropped.

``game`` is duck-typed so the agent never imports the harness. It needs:
``state`` (grid, engine_state, levels_completed, available_actions, won, game_over, just_won_level),
``run`` (state, history, final_score, note), ``number_of_levels``, ``action_count``,
``execute_action(ActionInput, generated_tokens)`` and ``finish(state)``.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import arcengine

from sarbloh.agent.duck.action_names import to_engine_action, to_model_action, to_model_actions
from sarbloh.agent.duck.runtime_state import (
    RUNTIME_STATE_FILENAME,
    Frame,
    HistoryEntry,
    write_runtime_state,
)

ANALYZER_RETRY_BACKOFF_SECONDS = 1.0


def artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _analyzer_tokens(analyzer: Any) -> int:
    value = getattr(analyzer, "generated_tokens", None) if hasattr(analyzer, "generated_tokens") else getattr(
        analyzer, "total_tokens", 0
    )
    return max(0, int(value or 0))


def _level_number(game: Any) -> int:
    completed = game.state.levels_completed
    if game.state.won:
        return max(1, game.number_of_levels)
    return max(1, min(game.number_of_levels, completed + 1))


def _engine_action_names(game: Any) -> list[str]:
    names: list[str] = []
    for action_id in game.state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name != "RESET" and name not in names:
            names.append(name)
    return names


def _mouse_data(action_data: dict[str, Any] | None) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _action_display(action_name: str, action_data: dict[str, Any] | None = None) -> str:
    if action_name == "ACTION6":
        d = _mouse_data(action_data)
        return f"MOUSE(row={d['row']}, col={d['col']})"
    return to_model_action(action_name)


@dataclass
class GameSession:
    game: Any
    analyzer: Any
    run_dir: Path
    stop_event: threading.Event
    max_runtime_s: float | None
    max_actions: int | None = None
    soft_remaining: Callable[[], float | None] = lambda: None
    pass_index: int = 0
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    analysis_step: int = 0
    last_engine_action: str | None = None
    token_baseline: int = 0

    def __post_init__(self) -> None:
        stem = f"{artifact_stem(self.game.game_id)}_p{self.pass_index}"
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "artifacts" / f"{stem}_{RUNTIME_STATE_FILENAME}"
        self.transcript_path = self.run_dir / "transcripts" / f"{stem}.txt"

    # --- state -------------------------------------------------------------------------------------------
    def current_frame(self) -> Frame:
        return Frame(grid=self.game.state.grid, step=self.game.action_count, level=_level_number(self.game))

    def _write_state(self) -> None:
        write_runtime_state(self.state_path, current_frame=self.current_frame(), history=self.history_entries)

    def _elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def timing_payload(self) -> dict[str, float | None]:
        remaining = None if self.max_runtime_s is None else max(0.0, self.max_runtime_s - self._elapsed())
        return {"run_elapsed_seconds": self._elapsed(), "time_remaining_seconds": remaining}

    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        if configured is not None:
            candidates.append(float(configured))
        remaining = self.timing_payload()["time_remaining_seconds"]
        if remaining is not None:
            candidates.append(float(remaining))
        soft = self.soft_remaining()
        if soft is not None:
            candidates.append(soft)
        return max(0.1, min(candidates)) if candidates else None

    def should_stop(self) -> bool:
        run = self.game.run
        if run is None or run.state != "playing" or self.stop_event.is_set():
            return True
        if self.game.state.won:
            return True
        if self.max_runtime_s is not None and self._elapsed() >= self.max_runtime_s:
            return True
        return self.max_actions is not None and self.game.action_count >= self.max_actions

    # --- main loop ---------------------------------------------------------------------------------------
    def play(self) -> None:
        run = self.game.run
        self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_tokens(self.analyzer)
        if not self.history_entries:
            self.history_entries.append(HistoryEntry(action="", frame=self.current_frame()))
        self._write_state()
        try:
            retry_step: int | None = None
            while not self.should_stop():
                if self.game.state.game_over and self.last_engine_action != "RESET":
                    self._execute(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}), 1, 1, tokens=0)
                    continue
                if retry_step is None:
                    self.analysis_step += 1
                    step = self.analysis_step
                else:
                    step = retry_step
                self._write_state()
                result = self.analyzer.analyze(
                    self.state_path,
                    self.game.action_count,
                    valid_actions=_engine_action_names(self.game),
                    step_env=self.step_env,
                    transcript_path=self.transcript_path,
                    analysis_step=step,
                    request_timeout_seconds=self.request_timeout_seconds(),
                    should_stop=self.should_stop,
                )
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    retry_step = step
                    if self.should_stop():
                        break
                    time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue
                retry_step = None
                if getattr(result, "yielded_control", False):
                    retry_step = step
                    continue
        except Exception as exc:
            if run.final_score is None:
                run.note = f"error: {type(exc).__name__}: {exc}"
                self.game.finish("crashed")
        finally:
            if run.note is None:
                run.note = f"tokens={_analyzer_tokens(self.analyzer)}"
            self.game.finish("cancelled" if self.stop_event.is_set() else None)
            self.state_path.unlink(missing_ok=True)

    # --- the step_env tool -------------------------------------------------------------------------------
    def _normalize_actions(self, arguments: dict[str, Any]) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."
        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [{"action": arguments.get("action"), "row": arguments.get("row"), "col": arguments.get("col")}]
        actions: list[arcengine.ActionInput] = []
        for index, raw in enumerate(raw_actions, start=1):
            if not isinstance(raw, dict):
                return None, f"Action {index} must be a JSON object."
            name = to_engine_action(raw.get("action"))
            if not name:
                return None, f"Unknown action at index {index}: {raw.get('action')!r}"
            action_id = arcengine.GameAction.from_name(name)
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    data = {"x": max(0, min(63, int(raw["col"]))), "y": max(0, min(63, int(raw["row"])))}
                except (KeyError, TypeError, ValueError):
                    return None, f"MOUSE action at index {index} requires integer row and col arguments."
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }

    def _terminal_payload(self, requested: list[arcengine.ActionInput]) -> dict[str, Any]:
        st = self.game.state
        is_over, is_win = st.game_over, st.won
        return {
            "executed": False,
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.game.action_count,
            "level": _level_number(self.game),
            "score": st.levels_completed,
            "state": st.engine_state.name,
            "valid_actions": [],
            "board_changed": False,
            "done": is_win,
            "level_completed": False,
            "game_over": is_over,
            "run_complete": is_win,
            "batched": len(requested) > 1,
            "requested_count": len(requested),
            "executed_count": 0,
            "requested_actions": [_action_display(a.id.name, dict(a.data)) for a in requested],
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": "run_complete" if is_win else "game_over" if is_over else "stopped",
            **self.timing_payload(),
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested, error = self._normalize_actions(arguments)
        if error is not None or requested is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.should_stop() or self.game.state.game_over:
            return self._terminal_payload(requested)
        executed: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(requested)
        for batch_index, action in enumerate(requested, start=1):
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.state.available_actions:
                if executed:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(f"{_action_display(action.id.name, dict(action.data))} is not valid right now.")
            try:
                payload = self._execute(action, batch_index, batch_size)
            except Exception as exc:
                if executed:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)
            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break
        if not executed:
            return self._error_payload("No action was executed.")
        final = dict(executed[-1])
        final["reward"] = total_reward
        final["last_reward"] = executed[-1].get("reward", 0.0)
        final["batched"] = batch_size > 1
        final["requested_count"] = batch_size
        final["executed_count"] = len(executed)
        final["requested_actions"] = [_action_display(a.id.name, dict(a.data)) for a in requested]
        final["executed_actions"] = [str(p.get("action_display") or p.get("action_name") or "") for p in executed]
        final["board_changed"] = any(bool(p.get("board_changed")) for p in executed)
        final["stopped_early"] = len(executed) < batch_size
        if stop_reason is not None:
            final["stop_reason"] = stop_reason
        return final

    def _execute(self, action: arcengine.ActionInput, batch_index: int, batch_size: int, tokens: int | None = None) -> dict[str, Any]:
        prev_grid = self.game.state.grid
        prev_completed = self.game.state.levels_completed
        if tokens is None:
            current = _analyzer_tokens(self.analyzer)
            tokens = max(0, current - self.token_baseline)
            self.token_baseline = current
        new_state = self.game.execute_action(action, generated_tokens=tokens)
        self.last_engine_action = action.id.name
        display = _action_display(action.id.name, dict(action.data))
        frame = Frame(grid=new_state.grid, step=self.game.action_count, level=_level_number(self.game))
        self.history_entries.append(HistoryEntry(action=display, frame=frame))
        self._write_state()
        completed = new_state.levels_completed
        raw_state = new_state.engine_state
        is_win = raw_state == arcengine.GameState.WIN
        return {
            "executed": True,
            "action_num": self.game.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "reward": float(completed - prev_completed) / max(1.0, float(self.game.number_of_levels)),
            "state": raw_state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": prev_grid != new_state.grid,
            "done": is_win,
            "level_completed": bool(new_state.just_won_level and not is_win),
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": is_win,
            "action_name": action.id.name,
            "action_data": _mouse_data(action.data) if action.id == arcengine.GameAction.ACTION6 else dict(action.data),
            "action_display": display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            **self.timing_payload(),
        }


def make_duck_analyzer(config: dict[str, Any]) -> Any:
    """Build the Duck ToolAgent. ``sarbloh.config.apply_llm_env`` must have run before this import."""
    from sarbloh.agent.duck.tool_agent import ToolAgent

    return ToolAgent(
        model=config["llm"]["model_id"],
        timeout=config["analyzer_timeout"],
        save_request_logs=config["save_request_logs"],
        base_url=config["llm"]["base_url"],
        provider=config["llm"]["provider"],
    )


def play_game(
    game: Any,
    config: dict[str, Any],
    run_dir: Path,
    stop_event: threading.Event,
    soft_remaining: Callable[[], float | None] = lambda: None,
) -> None:
    """Entry point the harness calls, one thread per game. Never raises; failures mark the run crashed."""
    try:
        if config["agent"] != "duck":
            raise ValueError(f"unknown agent {config['agent']!r}")
        session = GameSession(
            game=game,
            analyzer=make_duck_analyzer(config),
            run_dir=run_dir,
            stop_event=stop_event,
            max_runtime_s=config["max_runtime_s_per_game"],
            max_actions=config["max_actions_per_game"],
            soft_remaining=soft_remaining,
        )
        session.play()
    except Exception as exc:  # noqa: BLE001
        if game.run is not None and game.run.final_score is None:
            game.run.note = f"error: {type(exc).__name__}: {exc}"
            game.finish("crashed")

"""Priority scheduler. Every game is alive from the start; only ``slots`` of them hold the GPU at a time.

A game must hold a slot to call the LLM. ``ScheduledLLM`` wraps the shared client for one game: its first ``chat``
waits for a slot, the game keeps the slot for ``quantum_calls`` LLM calls (its tool runs in between take
milliseconds), then gives it back and competes again. A turn-driven game (the agent) never loses it inside a turn:
it gives it back at the first turn end (``yield_turn``, after a move or a cap) once it has made ``quantum_calls``
calls since it got the slot. A game that waits for a slot loses its prefix in the server's cache (the other games
evict it), so every hand-over costs one request prefilled in full: handing over at every turn end (about two requests)
kept 10% of the prompt cached in the Qwen run of 2026-10-19. A level-up also sends it back to
compete at once, and so does a rewrite of the game's prompt (a drain, compaction or reset, ``mark_cold``) once it has
used its slot, unless the game is turn-driven: the next request is prefilled in full anyway, so the slot is handed over
when it costs nothing, not in the middle of a warm prefix the server would have to prefill again later. When a slot is free, the waiting game with the highest priority gets it;
ties go to the one that has waited longest.

Priority of a game playing level L+1 of n (L levels won), ``t`` output tokens and ``a`` actions spent on that level:

    (current_gain + tail_gain) x hope
    current_gain = ((L + 1) / (n (n + 1) / 2)) x (human_actions / (human_actions + a)) ** 2
    tail_gain    = (tail(n - (L + 1)) / 55) x tail_fraction
    hope         = 0.5 ** ((t / token_scale) ** 2 + (a / action_scale) ** 2)

``current_gain`` is the RHAE weight of the current level discounted by expected quadratic action efficiency
(``human_actions = 25``). ``tail_gain`` is the continuation value of the remaining levels unlocked after completing
this level (``8/55`` when >=3 levels remain, ``7/55`` when 2 remain, ``5/55`` when 1 remains, ``0`` on the final
level), preventing short 6-level games from starving deeper 8- to 10-level games on early levels, and fading linearly
over the final ``tail_fade_fraction`` of the run budget when there is no time left for future levels. ``hope`` falls
with both tokens and actions spent on the current level without a win.

A game that has never held a slot goes before every game that has, so each game gets at least one quantum early.

Standard library only.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Continuation value (on a 10-level / 55-weight reference scale) by levels remaining after the current level:
# >=3 remaining -> 8.0, 2 -> 7.0, 1 -> 5.0, 0 -> 0.0.
_TAIL_BY_REMAINING = (0.0, 5.0, 7.0, 8.0)
_REFERENCE_WEIGHT_SUM = 55.0


def priority(levels_won: int, n_levels: int | None, tokens_on_level: float, token_scale: float,
             actions_on_level: float = 0.0, action_scale: float = 115.0, human_actions: float = 25.0,
             tail_fraction: float = 1.0) -> float:
    n = max(1, int(n_levels or 8))
    level = min(max(0, int(levels_won)), n - 1)
    actions = max(0.0, float(actions_on_level))
    tokens = max(0.0, float(tokens_on_level))
    h = max(1.0, float(human_actions))
    efficiency = (h / (h + actions)) ** 2
    current_gain = ((level + 1) / (n * (n + 1) / 2)) * efficiency
    remaining = max(0, n - (level + 1))
    tail_raw = _TAIL_BY_REMAINING[min(remaining, 3)]
    tail_gain = (tail_raw / _REFERENCE_WEIGHT_SUM) * min(1.0, max(0.0, float(tail_fraction)))
    hope = 0.5 ** ((tokens / max(1.0, float(token_scale))) ** 2 + (actions / max(1.0, float(action_scale))) ** 2)
    return (current_gain + tail_gain) * hope


def _unpack_snapshot(snap: tuple[Any, ...]) -> tuple[int, int | None, float, float]:
    if len(snap) >= 4:
        return int(snap[0]), snap[1], float(snap[2]), float(snap[3])
    return int(snap[0]), snap[1], float(snap[2]), 0.0


class SchedulerStopped(RuntimeError):
    """The game stopped (run stop or its deadline) while waiting for a slot."""


@dataclass
class _Game:
    snapshot: Callable[[], tuple[Any, ...]]   # (levels won, number of levels, tokens on level[, actions on level])
    grants: int = 0
    wait_s: float = 0.0
    held_s: float = 0.0
    first_grant_s: float | None = None
    waiting_since: float | None = None
    seq: int = 0
    held_at: float | None = None


class PriorityScheduler:
    def __init__(self, slots: int, quantum_calls: int = 4, token_scale: float = 80000.0,
                 log_path: Path | None = None, clock: Callable[[], float] = time.monotonic,
                 action_scale: float = 115.0, human_actions: float = 25.0,
                 run_budget_s: float | None = None, tail_fade_fraction: float = 0.2) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots, self.quantum_calls, self.token_scale = int(slots), max(1, int(quantum_calls)), float(token_scale)
        self.action_scale, self.human_actions = float(action_scale), float(human_actions)
        self.run_budget_s = None if run_budget_s is None else max(0.0, float(run_budget_s))
        self.tail_fade_fraction = max(0.0, min(1.0, float(tail_fade_fraction)))
        self.log_path, self.clock = log_path, clock
        self._cond = threading.Condition()
        self._games: dict[str, _Game] = {}
        self._holders: set[str] = set()
        self._seq = itertools.count()
        self._t0 = clock()
        self._closed = False

    # --- bookkeeping ---------------------------------------------------------------------------------------
    def register(self, key: str, snapshot: Callable[[], tuple[Any, ...]]) -> None:
        with self._cond:
            self._games[key] = _Game(snapshot=snapshot)

    def _tail_fraction(self) -> float:
        if self.run_budget_s is None or self.run_budget_s <= 0.0 or self.tail_fade_fraction <= 0.0:
            return 1.0
        window = self.tail_fade_fraction * self.run_budget_s
        remaining = max(0.0, self.run_budget_s - (self.clock() - self._t0))
        return min(1.0, remaining / window) if window > 0.0 else 1.0

    def priority_of(self, key: str) -> float:
        won, n, tokens, actions = _unpack_snapshot(self._games[key].snapshot())
        return priority(won, n, tokens, self.token_scale, actions_on_level=actions,
                        action_scale=self.action_scale, human_actions=self.human_actions,
                        tail_fraction=self._tail_fraction())

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(self.clock() - self._t0, 1), **row}) + "\n")
        except OSError:
            pass

    def _next(self) -> str | None:
        """The waiting game to admit: a game never granted first, then highest priority, then longest waiting."""
        waiting = [k for k, g in self._games.items() if g.waiting_since is not None and k not in self._holders]
        if not waiting:
            return None
        return max(waiting, key=lambda k: (self._games[k].grants == 0, self.priority_of(k), -self._games[k].seq))

    # --- slots ---------------------------------------------------------------------------------------------
    # A free slot is handed straight to the best waiting game (``_pump``), whether or not its thread is inside
    # ``acquire`` yet: a game re-queued by ``handover`` may still be running its tool call when it wins the slot.
    def _queue(self, key: str) -> None:
        g = self._games[key]
        if g.waiting_since is None:
            g.waiting_since, g.seq = self.clock(), next(self._seq)

    def _grant(self, key: str) -> None:
        g, now = self._games[key], self.clock()
        waited = now - g.waiting_since
        g.waiting_since, g.held_at = None, now
        g.grants += 1
        g.wait_s += waited
        if g.first_grant_s is None:
            g.first_grant_s = now - self._t0
        self._holders.add(key)
        won, n, tokens, actions = _unpack_snapshot(g.snapshot())
        self._log({"event": "grant", "game": key, "priority": round(self.priority_of(key), 5),
                   "waited_s": round(waited, 1), "levels": won, "of": n, "tokens_on_level": int(tokens),
                   "actions_on_level": int(actions),
                   "holders": sorted(self._holders), "waiting": self._waiting_count()})

    def _pump(self) -> None:
        granted = False
        while not self._closed and len(self._holders) < self.slots:
            key = self._next()
            if key is None:
                break
            self._grant(key)
            granted = True
        if granted:
            self._cond.notify_all()

    def _drop(self, key: str, reason: str) -> None:
        g = self._games[key]
        self._holders.discard(key)
        held = self.clock() - (g.held_at or self.clock())
        g.held_s += held
        g.held_at = None
        self._log({"event": "release", "game": key, "reason": reason, "held_s": round(held, 1)})

    def acquire(self, key: str, should_stop: Callable[[], bool] = lambda: False, poll_s: float = 1.0) -> None:
        """Blocks until ``key`` holds a slot. Raises SchedulerStopped when ``should_stop()`` turns true or on close."""
        with self._cond:
            if key in self._holders:
                return
            self._queue(key)
            self._pump()
            while key not in self._holders:
                if self._closed or should_stop():
                    self._games[key].waiting_since = None
                    self._pump()
                    raise SchedulerStopped(f"{key}: stopped while waiting for a slot")
                self._cond.wait(timeout=poll_s)

    def handover(self, key: str, reason: str = "quantum") -> None:
        """Give the slot back and queue again in one step: the best waiting game gets it, which may be this one."""
        with self._cond:
            if key not in self._holders:
                return
            self._drop(key, reason)
            self._queue(key)
            self._pump()

    def release(self, key: str, reason: str = "") -> None:
        """Leave for good (the game ended): the slot goes to the best waiting game."""
        with self._cond:
            self._games[key].waiting_since = None
            if key in self._holders:
                self._drop(key, reason)
            self._pump()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _waiting_count(self) -> int:
        return sum(1 for k, g in self._games.items() if g.waiting_since is not None and k not in self._holders)

    def status(self) -> str:
        with self._cond:
            return f"holders={sorted(self._holders)} waiting={self._waiting_count()}"

    def stats(self) -> dict[str, Any]:
        with self._cond:
            now = self.clock()
            per = {}
            for k, g in self._games.items():
                held = g.held_s + ((now - g.held_at) if g.held_at is not None else 0.0)
                per[k] = {"grants": g.grants, "held_s": round(held, 1), "wait_s": round(g.wait_s, 1),
                          "first_grant_s": None if g.first_grant_s is None else round(g.first_grant_s, 1)}
            return {"slots": self.slots, "quantum_calls": self.quantum_calls, "token_scale": self.token_scale,
                    "action_scale": self.action_scale, "human_actions": self.human_actions,
                    "tail_fade_fraction": self.tail_fade_fraction,
                    "never_granted": sorted(k for k, v in per.items() if v["grants"] == 0), "per_game": per}


class ScheduledLLM:
    """One game's view of the shared LLM client: ``chat`` runs only while the game holds a slot. Everything else
    (``usage``, ``model``, ``cfg``, ``gate`` ...) is the shared client's."""

    def __init__(self, llm: Any, scheduler: PriorityScheduler, key: str, levels_won: Callable[[], int],
                 n_levels: int | None, should_stop: Callable[[], bool],
                 actions_on_level: Callable[[], int] = lambda: 0, turn_driven: bool = False) -> None:
        self._llm, self._sched, self._key = llm, scheduler, key
        # turn_driven: the agent ends its own turns (``yield_turn`` after a move or a cap), so a turn is never cut by
        # ``quantum_calls`` or by a cold prompt in the middle; the level-up hand-over stays.
        self.turn_driven = bool(turn_driven)
        self._levels_won, self._should_stop = levels_won, should_stop
        self._n_levels = n_levels
        self._actions_on_level = actions_on_level
        self._level = levels_won()
        self.tokens_on_level = 0
        self._calls_in_quantum = 0
        self._cold = False
        scheduler.register(key, self._snapshot)

    def _snapshot(self) -> tuple[int, int | None, float, float]:
        level = self._levels_won()
        if level != self._level:
            return (level, self._n_levels, 0.0, 0.0)
        return (level, self._n_levels, float(self.tokens_on_level), float(self._actions_on_level()))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        level = self._levels_won()
        if level != self._level:   # a level-up since the last call: a fresh, deeper level competes again at once
            self._level, self.tokens_on_level = level, 0
            self._calls_in_quantum = 0
            self._sched.handover(self._key, "level_up")
        elif self._cold and self._calls_in_quantum > 0 and not self.turn_driven:   # rewritten: hand over while cold
            self._calls_in_quantum = 0
            self._sched.handover(self._key, "cold")
        self._cold = False
        self._sched.acquire(self._key, self._should_stop)
        try:
            reply = self._llm.chat(*args, **kwargs)
            self.tokens_on_level += int(getattr(reply, "completion_tokens", 0) or 0)
            return reply
        finally:
            self._calls_in_quantum += 1
            if not self.turn_driven and self._calls_in_quantum >= self._sched.quantum_calls:
                self._calls_in_quantum = 0
                self._sched.handover(self._key, "quantum")

    def yield_turn(self) -> None:
        """The game's turn ended (a move, or a cap without one): the slot goes to the best waiting game now (this game
        queues again and may get it straight back). A turn-driven game does so only once it has made ``quantum_calls``
        calls since it got the slot; before that it keeps the slot for its next turn."""
        if self.turn_driven and self._calls_in_quantum < self._sched.quantum_calls:
            return
        if self._calls_in_quantum > 0:
            self._calls_in_quantum = 0
            self._sched.handover(self._key, "yield")

    def mark_cold(self) -> None:
        """The game's prompt was rewritten: its next ``chat`` hands the slot over first (if it has used it)."""
        self._cold = True

    def close(self) -> None:
        self._sched.release(self._key, "game_end")

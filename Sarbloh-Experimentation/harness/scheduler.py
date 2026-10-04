"""Priority scheduler. Every game is alive from the start; only ``slots`` of them hold the GPU at a time.

A game must hold a slot to call the LLM. ``ScheduledLLM`` wraps the shared client for one game: its first ``chat``
waits for a slot, the game keeps the slot for ``quantum_calls`` LLM calls (its tool runs in between take
milliseconds), then gives it back and competes again. A level-up also sends it back to compete at once, and so does a
rewrite of the game's prompt (a drain, compaction or reset, ``mark_cold``) once it has used its slot: the next request
is prefilled in full anyway, so the slot is handed over when it costs nothing, not in the middle of a warm prefix the
server would have to prefill again later. When a slot is free, the waiting game with the highest priority gets it;
ties go to the one that has waited longest.

Priority of a game playing level L+1 of n (L levels won), ``t`` output tokens spent on that level:

    gain x hope,   gain = (L + 1) / (n (n + 1) / 2),   hope = 0.5 ** ((t / token_scale) ** 2)

``gain`` is the weight the next level has in the game's RHAE (levels are weighted by index). ``hope`` falls with the
tokens a level has eaten without a win: with token_scale 80k (from the 2026-10-01 run: 6 of 9 level-1 wins came
before 52k tokens, the 8 games that never won burned ~150k) it is 0.76 at 50k and 0.09 at 150k. A game that just won a
level is deeper and fresh again, so it goes to the top; a game stuck for long sinks below the games not yet helped.

A game that has never held a slot goes before every game that has, so each game gets at least one quantum early. The
2026-10-01 local smoke (1 slot) showed why: tr87 (6 levels, 1/21) outranked ls20 (7 levels, 1/28) until it had
spent ~43k tokens on level 1, so ls20 never played in 8 minutes.

Standard library only. Our design, from the idea in the milestone-2 reference notebook (kaggle/reference/; its code is
not used).
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def priority(levels_won: int, n_levels: int | None, tokens_on_level: float, token_scale: float) -> float:
    n = max(1, int(n_levels or 8))
    level = min(max(0, int(levels_won)), n - 1)
    gain = (level + 1) / (n * (n + 1) / 2)
    hope = 0.5 ** ((max(0.0, tokens_on_level) / max(1.0, token_scale)) ** 2)
    return gain * hope


class SchedulerStopped(RuntimeError):
    """The game stopped (run stop or its deadline) while waiting for a slot."""


@dataclass
class _Game:
    snapshot: Callable[[], tuple[int, int | None, float]]   # (levels won, number of levels, tokens on this level)
    grants: int = 0
    wait_s: float = 0.0
    held_s: float = 0.0
    first_grant_s: float | None = None
    waiting_since: float | None = None
    seq: int = 0
    held_at: float | None = None


class PriorityScheduler:
    def __init__(self, slots: int, quantum_calls: int = 4, token_scale: float = 80000.0,
                 log_path: Path | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots, self.quantum_calls, self.token_scale = int(slots), max(1, int(quantum_calls)), float(token_scale)
        self.log_path, self.clock = log_path, clock
        self._cond = threading.Condition()
        self._games: dict[str, _Game] = {}
        self._holders: set[str] = set()
        self._seq = itertools.count()
        self._t0 = clock()
        self._closed = False

    # --- bookkeeping ---------------------------------------------------------------------------------------
    def register(self, key: str, snapshot: Callable[[], tuple[int, int | None, float]]) -> None:
        with self._cond:
            self._games[key] = _Game(snapshot=snapshot)

    def priority_of(self, key: str) -> float:
        won, n, tokens = self._games[key].snapshot()
        return priority(won, n, tokens, self.token_scale)

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
        won, n, tokens = g.snapshot()
        self._log({"event": "grant", "game": key, "priority": round(self.priority_of(key), 5),
                   "waited_s": round(waited, 1), "levels": won, "of": n, "tokens_on_level": int(tokens),
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
                    "never_granted": sorted(k for k, v in per.items() if v["grants"] == 0), "per_game": per}


class ScheduledLLM:
    """One game's view of the shared LLM client: ``chat`` runs only while the game holds a slot. Everything else
    (``usage``, ``model``, ``cfg``, ``gate`` ...) is the shared client's."""

    def __init__(self, llm: Any, scheduler: PriorityScheduler, key: str, levels_won: Callable[[], int],
                 n_levels: int | None, should_stop: Callable[[], bool]) -> None:
        self._llm, self._sched, self._key = llm, scheduler, key
        self._levels_won, self._should_stop = levels_won, should_stop
        self._level = levels_won()
        self.tokens_on_level = 0
        self._calls_in_quantum = 0
        self._cold = False
        scheduler.register(key, lambda: (self._levels_won(), n_levels, self.tokens_on_level))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        level = self._levels_won()
        if level != self._level:   # a level-up since the last call: a fresh, deeper level competes again at once
            self._level, self.tokens_on_level = level, 0
            self._calls_in_quantum = 0
            self._sched.handover(self._key, "level_up")
        elif self._cold and self._calls_in_quantum > 0:   # its prompt was rewritten: hand over while it is cold
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
            if self._calls_in_quantum >= self._sched.quantum_calls:
                self._calls_in_quantum = 0
                self._sched.handover(self._key, "quantum")

    def mark_cold(self) -> None:
        """The game's prompt was rewritten: its next ``chat`` hands the slot over first (if it has used it)."""
        self._cold = True

    def close(self) -> None:
        self._sched.release(self._key, "game_end")

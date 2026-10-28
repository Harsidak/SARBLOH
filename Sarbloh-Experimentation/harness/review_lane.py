"""The review lane: stuck reviews run beside the games, so a stuck game does not wait for its own turn to be reviewed
(run 10-27: tn36 waited 2.2 h for a slot after 4 h, so its review only came at 5.6 h).

From config memory.review_at_s of run time (3 h), every POLL_S the lane looks for the most stuck games
(``AgentSession.review_due``: the most reply tokens on their level since its last review) and runs each review
(``AgentSession.review_job``) on a thread of its own with the shared LLM client, while the game goes on playing or
waiting for its slot. memory.review_lanes reviews run at once on server requests of their own (the server takes the
scheduler's slots plus these); up to memory.review_borrow more run on scheduler slots no game is waiting for
(``PriorityScheduler.borrow``), given back as soon as the review ends. The finished review is handed to its game
(``offer_review``), whose own thread applies it before its next request.

Standard library only.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

POLL_S = 5.0


class ReviewLane:
    def __init__(self, llm: Any, *, start_at: float, lanes: int = 1, borrow: int = 0, scheduler: Any = None,
                 log_path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.llm, self.start_at, self.scheduler = llm, float(start_at), scheduler
        self.lanes, self.borrow = max(0, int(lanes)), max(0, int(borrow))
        self.log_path, self.clock = log_path, clock
        self._sessions: dict[str, Any] = {}
        self._running: dict[str, str] = {}   # game -> "lane" (its own request) or "borrowed" (a scheduler slot)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.counts = {"lane": 0, "borrowed": 0, "errors": 0}

    # --- games ---------------------------------------------------------------------------------------------
    def register(self, key: str, session: Any) -> None:
        """The lane reviews this game from now on (the game no longer reviews itself at turn ends)."""
        session.review_lane = True
        with self._lock:
            self._sessions[key] = session

    def unregister(self, key: str) -> None:
        with self._lock:
            self._sessions.pop(key, None)

    # --- the loop ------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="review-lane")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(POLL_S):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the lane must never take the run down
                self._log({"event": "lane_error", "traceback": traceback.format_exc()[-2000:]})

    def tick(self, now: float | None = None) -> list[str]:
        """One look: starts the reviews that are due and fit (a free lane, else a borrowed slot), the most stuck game
        first. Returns the games whose review started."""
        now = self.clock() if now is None else now
        if now < self.start_at or self._stop.is_set():
            return []
        with self._lock:
            pending = [(k, s) for k, s in self._sessions.items() if k not in self._running]
        due = []
        for key, s in pending:
            try:
                d = s.review_due(now)
            except Exception:  # noqa: BLE001 - a game in an odd state is skipped this time
                d = None
            if d is not None:
                due.append((d, key, s))
        due.sort(key=lambda x: -x[0])
        started = []
        for d, key, s in due:
            with self._lock:
                in_lane = sum(1 for v in self._running.values() if v == "lane")
                borrowed = sum(1 for v in self._running.values() if v == "borrowed")
            if in_lane < self.lanes:
                kind = "lane"
            elif borrowed < self.borrow and self.scheduler is not None and self.scheduler.borrow(f"review:{key}"):
                kind = "borrowed"
            else:
                break
            with self._lock:
                self._running[key] = kind
            self.counts[kind] += 1
            self._log({"event": "start", "game": key, "kind": kind, "tokens_since_review": d})
            self._spawn(self._run, key, s, kind)
            started.append(key)
        return started

    def _spawn(self, fn: Callable[..., None], *args: Any) -> None:
        threading.Thread(target=fn, args=args, daemon=True, name=f"review-{args[0]}").start()

    def _run(self, key: str, session: Any, kind: str) -> None:
        t0 = self.clock()
        try:
            res = session.review_job(self.llm)
            res["lane"] = kind
            session.offer_review(res)
            if "error" in res:
                self.counts["errors"] += 1
            self._log({"event": "done", "game": key, "kind": kind, "level": res.get("level", 0) + 1,
                       "seconds": round(self.clock() - t0, 1), "error": res.get("error"),
                       "reply_tokens": (res.get("rev") or {}).get("reply_tokens")})
        except Exception:  # noqa: BLE001
            self.counts["errors"] += 1
            self._log({"event": "done", "game": key, "kind": kind, "traceback": traceback.format_exc()[-2000:]})
        finally:
            if kind == "borrowed" and self.scheduler is not None:
                self.scheduler.give_back(f"review:{key}")
            with self._lock:
                self._running.pop(key, None)

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(self.clock(), 1), **row}, default=str) + "\n")
        except OSError:
            pass

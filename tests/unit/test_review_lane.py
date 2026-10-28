# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The review lane and the scheduler slots it borrows: nothing before its start time; the most stuck game first; one
review per lane, more only on a free scheduler slot (none when the slots are full), given back when the review ends;
a game under review is not picked twice; a game the lane registers stops reviewing itself; the result goes back to
its game with the lane's kind; a failure is counted. ``ScheduledLLM.wait_slot`` takes the slot before a request is
built (a level-up hands it over first), and ``chat`` after it does not wait again. The server takes 11 requests."""
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from harness.llm import qwen_flash_next as qfn
from harness.review_lane import ReviewLane
from harness.scheduler import PriorityScheduler, ScheduledLLM

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- scheduler: borrow and give back ---------------------------------------------------------------------------------
log = Path(tempfile.mkdtemp()) / "scheduler.jsonl"
sched = PriorityScheduler(2, 4, log_path=log)
sched.register("g1", lambda: (0, 5, 0.0, 0.0))
sched.register("g2", lambda: (0, 5, 0.0, 0.0))
sched.acquire("g1")
ok("borrow_free_slot", sched.borrow("review:g9") and "review:g9" in sched._holders)
ok("borrow_refused_when_full", not sched.borrow("review:g8"))
got = []
t = threading.Thread(target=lambda: (sched.acquire("g2"), got.append("g2")))
t.start()
t.join(0.3)
ok("game_waits_while_borrowed", not got)
sched.give_back("review:g9")
t.join(2)
ok("give_back_hands_slot_to_waiting_game", got == ["g2"] and "review:g9" not in sched._holders)
sched.give_back("g1")   # a game's own slot is never taken by give_back
ok("give_back_ignores_games", "g1" in sched._holders)
rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
ok("borrow_logged", [r["event"] for r in rows if r["event"] in ("borrow", "give_back")] == ["borrow", "give_back"])
ok("stats_still_work", set(sched.stats()["per_game"]) == {"g1", "g2"})

# --- ScheduledLLM.wait_slot -------------------------------------------------------------------------------------------
sched = PriorityScheduler(1, 4)
level = [0]
inner = SimpleNamespace(calls=0)


def _chat(*a, **k):
    inner.calls += 1
    return SimpleNamespace(completion_tokens=5)


inner.chat = _chat
sl = ScheduledLLM(inner, sched, "g1", levels_won=lambda: level[0], n_levels=5, should_stop=lambda: False,
                  turn_driven=True)
sl.wait_slot()
ok("wait_slot_takes_slot", "g1" in sched._holders and inner.calls == 0)
sl.chat([])
ok("chat_after_wait_runs", inner.calls == 1 and sl.tokens_on_level == 5)
level[0] = 1
handed = []
orig = sched.handover
sched.handover = lambda key, reason="quantum": (handed.append(reason), orig(key, reason))
sl.wait_slot()
ok("wait_slot_level_up_hands_over", handed == ["level_up"] and sl.tokens_on_level == 0 and "g1" in sched._holders)
sl.chat([])
ok("chat_no_second_handover", handed == ["level_up"])


# --- the lane -----------------------------------------------------------------------------------------------------------
class Game:
    def __init__(self, due, fail=False):
        self.due, self.fail, self.offered, self.review_lane = due, fail, [], False

    def review_due(self, now):
        return self.due

    def review_job(self, llm):
        if self.fail:
            raise RuntimeError("boom")
        return {"level": 2, "rev": {"reply_tokens": 10}}

    def offer_review(self, res):
        self.offered.append(res)


now = [0.0]
sched = PriorityScheduler(3, 4)
for k in ("a", "b"):
    sched.register(k, lambda: (0, 5, 0.0, 0.0))
    sched.acquire(k)
lane_log = Path(tempfile.mkdtemp()) / "review_lane.jsonl"
lane = ReviewLane(SimpleNamespace(), start_at=100.0, lanes=1, borrow=1, scheduler=sched, log_path=lane_log,
                  clock=lambda: now[0])
jobs = []
lane._spawn = lambda fn, *args: jobs.append((fn, args))   # run the reviews when the test says so
games = {"a": Game(150000), "b": Game(300000), "c": Game(None), "d": Game(120000)}
for k, g in games.items():
    lane.register(k, g)
ok("register_marks_game", all(g.review_lane for g in games.values()))
ok("nothing_before_start", lane.tick() == [] and not jobs)
now[0] = 100.0
started = lane.tick()
ok("most_stuck_first", started == ["b", "a"], str(started))
ok("lane_then_borrowed", lane._running == {"b": "lane", "a": "borrowed"} and "review:a" in sched._holders
   and "review:b" not in sched._holders, str(lane._running))
ok("no_more_when_full", lane.tick() == [], str(lane._running))
for fn, args in jobs:
    fn(*args)
jobs.clear()
ok("results_go_back_with_kind", games["b"].offered[0]["lane"] == "lane" and games["a"].offered[0]["lane"] == "borrowed")
ok("borrowed_slot_given_back", "review:a" not in sched._holders and lane._running == {})
games["a"].due = games["b"].due = None
games["d"].due = 120000
sched.register("e", lambda: (0, 5, 0.0, 0.0))
sched.acquire("e")   # every scheduler slot taken by a game: only the lane's own request is left
games["c"].due = 200000
started = lane.tick()
ok("lane_only_when_slots_full", started == ["c"] and lane._running == {"c": "lane"}, str(started))
lane.unregister("d")
for fn, args in jobs:
    fn(*args)
jobs.clear()
games["c"].due = None   # reviewed: not due until it spends more tokens
ok("unregistered_not_picked", lane.tick() == [] or "d" not in lane._running)
bad = Game(500000, fail=True)
lane.register("x", bad)
lane.tick()
for fn, args in jobs:
    fn(*args)
ok("failure_counted", lane.counts["errors"] == 1 and not bad.offered and "x" not in lane._running, str(lane.counts))
rows = [json.loads(x) for x in lane_log.read_text(encoding="utf-8").splitlines()]
ok("lane_logged", [r["event"] for r in rows].count("start") >= 3 and any(r.get("traceback") for r in rows))
lane.close()
ok("closed_lane_starts_nothing", lane.tick() == [])

# --- the server takes the lane's request -----------------------------------------------------------------------------
f = qfn.SGL_FLASH
ok("server_eleven_requests", f["--max-running-requests"] == "11" and f["--cuda-graph-max-bs-decode"] == "11")

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
raise SystemExit(1 if FAILS else 0)

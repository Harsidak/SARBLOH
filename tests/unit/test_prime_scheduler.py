# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # PRIME_DIR=Sarbloh tests the older copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E111 priority scheduler without a model (experiments/E111_priority_scheduler): the priority formula, who gets a
free slot, the quantum hand-over, the level-up re-compete, stop wakes a waiter, no starvation of fresh games, and
the run.py wiring (all games alive, only `slots` on the LLM at once)."""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from harness.config import DEFAULT, build_config
from harness.scheduler import PriorityScheduler, ScheduledLLM, SchedulerStopped, priority

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. the formula ----------------------------------------------------------------------------------------------
# Sarbloh mirrors the experimentation copy since 2026-10-10: the scheduler is on by default with 10 slots.
ok("default_on", DEFAULT["scheduler"]["enabled"] is True and build_config({})["scheduler"]["slots"] == 10)
fresh7 = priority(0, 7, 0, 80000)
ok("fresh_is_first_level_plus_tail", abs(fresh7 - (1 / 28 + 8 / 55)) < 1e-12, fresh7)
ok("tail_zero_is_first_level_weight", abs(priority(0, 7, 0, 80000, tail_fraction=0.0) - 1 / 28) < 1e-12)
ok("deeper_beats_fresh", priority(1, 7, 0, 80000) > fresh7)
ok("hope_076_at_50k", abs(priority(0, 7, 50000, 80000) / fresh7 - 0.5 ** ((50 / 80) ** 2)) < 1e-12)
ok("hope_009_at_150k", round(priority(0, 7, 150000, 80000) / fresh7, 2) == 0.09)
ok("actions_decay_priority", priority(0, 7, 0, 80000, actions_on_level=100) < fresh7 * 0.5)
ok("long_game_not_starved_by_slightly_spent_short_game",
   priority(0, 8, 0, 40000) > priority(0, 6, 16000, 40000))
ok("stuck_below_fresh", priority(0, 7, 120000, 80000) < priority(0, 7, 0, 80000))
ok("short_game_level_worth_more", priority(0, 6, 0, 80000) > priority(0, 10, 0, 80000))
ok("unknown_levels_ok", priority(0, None, 0, 80000) == priority(0, 8, 0, 80000))
ok("won_all_clamped", priority(9, 7, 0, 80000) == priority(6, 7, 0, 80000))


# --- 2. grants by priority, ties by waiting order ----------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


state = {"a": (0, 7, 0), "b": (0, 7, 100000), "c": (1, 7, 0)}
log = Path(tempfile.mkdtemp(prefix="e111_")) / "scheduler.jsonl"
s = PriorityScheduler(1, quantum_calls=2, token_scale=80000, log_path=log)
for k in state:
    s.register(k, lambda k=k: state[k])
s.acquire("a")
order = []


def waiter(k):
    s.acquire(k)
    order.append(k)


ts = [threading.Thread(target=waiter, args=(k,)) for k in ("b", "c")]
for t in ts:
    t.start()
ok("both_waiting", wait_until(lambda: s._waiting_count() == 2))
s.release("a", "test")
ok("best_priority_first", wait_until(lambda: order == ["c"]), order)
s.release("c", "test")
ok("then_the_other", wait_until(lambda: order == ["c", "b"]), order)
s.release("b", "test")
for t in ts:
    t.join(2)
rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
ok("log_grants_and_releases", [r["event"] for r in rows].count("grant") == 3
   and [r["event"] for r in rows].count("release") == 3 and "priority" in rows[0], rows[:2])

# ties: equal priority -> the one that started waiting first
s2 = PriorityScheduler(1)
for k in ("x", "y", "z"):
    s2.register(k, lambda: (0, 7, 0))
s2.acquire("x")
got = []
ty = threading.Thread(target=lambda: (s2.acquire("y"), got.append("y")))
ty.start()
wait_until(lambda: s2._waiting_count() == 1)
tz = threading.Thread(target=lambda: (s2.acquire("z"), got.append("z")))
tz.start()
wait_until(lambda: s2._waiting_count() == 2)
s2.release("x")
ok("tie_goes_to_longest_waiting", wait_until(lambda: got == ["y"]), got)
s2.release("y")
wait_until(lambda: got == ["y", "z"])
s2.release("z")
ty.join(2)
tz.join(2)

# --- 3. stop and close wake a waiter -----------------------------------------------------------------------------
s3 = PriorityScheduler(1)
s3.register("p", lambda: (0, 7, 0))
s3.register("q", lambda: (0, 7, 0))
s3.acquire("p")
flag, res = threading.Event(), []


def stopper():
    try:
        s3.acquire("q", should_stop=flag.is_set, poll_s=0.05)
        res.append("granted")
    except SchedulerStopped:
        res.append("stopped")


t3 = threading.Thread(target=stopper)
t3.start()
wait_until(lambda: s3._waiting_count() == 1)
flag.set()
t3.join(2)
ok("stop_wakes_waiter", res == ["stopped"] and s3._waiting_count() == 0, res)
res.clear()
t4 = threading.Thread(target=stopper)
flag.clear()
t4.start()
wait_until(lambda: s3._waiting_count() == 1)
s3.close()
t4.join(2)
ok("close_wakes_waiter", res == ["stopped"], res)
try:
    PriorityScheduler(0)
    ok("zero_slots_rejected", False)
except ValueError:
    ok("zero_slots_rejected", True)


# --- 4. ScheduledLLM: quantum, level-up, delegation --------------------------------------------------------------
class FakeLLM:
    def __init__(self, tokens=1000, delay=0.0):
        self.calls, self.tokens, self.delay = 0, tokens, delay
        self.model, self.usage = "fake", SimpleNamespace(calls=0)
        self.inflight, self.max_inflight = 0, 0
        self._lock = threading.Lock()

    def chat(self, *a, **k):
        with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        time.sleep(self.delay)
        with self._lock:
            self.inflight -= 1
            self.calls += 1
        return SimpleNamespace(completion_tokens=self.tokens)


s4 = PriorityScheduler(1, quantum_calls=3)
lv = {"g": 0}
fake = FakeLLM()
g = ScheduledLLM(fake, s4, "g", levels_won=lambda: lv["g"], n_levels=7, should_stop=lambda: False)
ok("delegates_attributes", g.model == "fake" and g.usage is fake.usage)
g.chat([])
g.chat([])
ok("holds_within_quantum", "g" in s4._holders and g.tokens_on_level == 2000)
g.chat([])
ok("quantum_handover_alone_keeps_slot", "g" in s4._holders and s4._games["g"].grants == 2)
g.chat([])
ok("holding_needs_no_new_grant", s4._games["g"].grants == 2)
lv["g"] = 1
g.chat([])
ok("level_up_recompetes_and_resets_tokens", s4._games["g"].grants == 3 and g.tokens_on_level == 1000,
   (s4._games["g"].grants, g.tokens_on_level))
g.close()
ok("close_releases", "g" not in s4._holders)


class Boom(Exception):
    pass


class FailLLM(FakeLLM):
    def chat(self, *a, **k):
        raise Boom()


s5 = PriorityScheduler(1, quantum_calls=1)
f = ScheduledLLM(FailLLM(), s5, "f", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False)
try:
    f.chat([])
except Boom:
    pass
ok("error_still_counts_quantum", s5._games["f"].grants == 2)

# the 2026-10-01 local smoke bug: at the end of a quantum the best game must keep its slot, a better waiter takes it
pr = {"low": (0, 7, 150000), "high": (1, 7, 0)}
s8 = PriorityScheduler(1, quantum_calls=1)
for k in pr:
    s8.register(k, lambda k=k: pr[k])
s8.acquire("low")
th8 = threading.Thread(target=s8.acquire, args=("high",))
th8.start()
wait_until(lambda: s8._waiting_count() == 1)
s8.handover("low")
th8.join(2)
ok("better_waiter_takes_slot_at_handover", s8._holders == {"high"} and s8._games["low"].waiting_since is not None)
s8.handover("high")
ok("best_game_keeps_slot_at_handover", s8._holders == {"high"} and s8._games["high"].grants == 2)
s8.release("high", "game_end")
ok("queued_game_granted_without_its_thread", s8._holders == {"low"})
s8.acquire("low")
ok("acquire_returns_when_already_granted", s8._holders == {"low"} and s8._games["low"].grants == 2)
s8.release("low", "game_end")
ok("all_released", s8._holders == set() and s8._waiting_count() == 0)

# --- 5. many games, few slots: never more than `slots` on the LLM, nobody starves --------------------------------
s6 = PriorityScheduler(2, quantum_calls=2)
shared = FakeLLM(tokens=30000, delay=0.01)
games = {f"g{i}": ScheduledLLM(shared, s6, f"g{i}", levels_won=lambda: 0, n_levels=7, should_stop=lambda: False)
         for i in range(6)}


def play(w):
    for _ in range(6):
        w.chat([])
    w.close()


th = [threading.Thread(target=play, args=(w,)) for w in games.values()]
for t in th:
    t.start()
for t in th:
    t.join(20)
st = s6.stats()
ok("never_more_than_slots", shared.max_inflight <= 2, shared.max_inflight)
ok("everyone_finished", shared.calls == 36, shared.calls)
ok("nobody_never_granted", st["never_granted"] == [] and all(v["grants"] >= 3 for v in st["per_game"].values()),
   st["per_game"])
# a stuck game sinks below a fresh one
s7 = PriorityScheduler(1)
s7.register("stuck", lambda: (0, 7, 150000))
s7.register("fresh", lambda: (0, 7, 0))
s7._games["stuck"].waiting_since, s7._games["stuck"].seq = 0.0, 0
s7._games["fresh"].waiting_since, s7._games["fresh"].seq = 1.0, 1
ok("fresh_beats_stuck_even_if_later", s7._next() == "fresh")
# the second 2026-10-01 local smoke: 1 slot, tr87 (6 levels) outranks ls20 (7 levels) at 0 tokens, so ls20 never
# played. A game never granted goes first at the next hand-over; after its quantum the better game gets the slot back.
pr9 = {"tr87": (0, 6, 1000), "ls20": (0, 7, 0)}
s9 = PriorityScheduler(1, quantum_calls=4)
for k in pr9:
    s9.register(k, lambda k=k: pr9[k])
s9.acquire("tr87")
th9 = threading.Thread(target=s9.acquire, args=("ls20",))
th9.start()
wait_until(lambda: s9._waiting_count() == 1)
ok("smoke_setup_tr87_outranks_ls20", s9.priority_of("tr87") > s9.priority_of("ls20"))
s9.handover("tr87")
th9.join(2)
ok("never_granted_goes_first", s9._holders == {"ls20"}, s9._holders)
s9.handover("ls20")
ok("then_priority_again", s9._holders == {"tr87"} and s9._games["ls20"].waiting_since is not None, s9._holders)
s9.release("tr87", "game_end")
s9.release("ls20", "game_end")

# --- 6. run.py wiring with fake games: all alive, scheduler stats in the summary ---------------------------------
from harness import run as prime_run  # noqa: E402

seen = {"max_live": 0, "live": 0}
lock = threading.Lock()


class FakeGame:
    def __init__(self, gid):
        self.game_id, self.env_name, self.run, self.number_of_levels = gid, gid, None, 7
        self.state = SimpleNamespace(levels_completed=0)

    def start(self):
        pass

    def finish(self, *_):
        pass


def fake_play_game(game, cfg, llm, run_dir, stop_event, soft_end, scheduler=None, lane=None):
    w = ScheduledLLM(llm, scheduler, game.game_id, levels_won=lambda: 0, n_levels=7, should_stop=stop_event.is_set)
    with lock:
        seen["live"] += 1
        seen["max_live"] = max(seen["max_live"], seen["live"])
    for _ in range(3):
        w.chat([])
    w.close()
    time.sleep(0.05)
    with lock:
        seen["live"] -= 1
    return {"end_reason": "fake", "turns": 3}


_real_play, _real_trace = prime_run.play_game, prime_run.trace.build_run
prime_run.play_game = fake_play_game
prime_run.trace.build_run = lambda run_dir: "skipped"
try:
    cfg = build_config({"concurrency": 2, "scheduler": {"enabled": True, "slots": 2, "quantum_calls": 1}})
    llm = FakeLLM(delay=0.01)
    llm.usage = SimpleNamespace(to_json=lambda: {})
    out = Path(tempfile.mkdtemp(prefix="e111_run_"))
    summ = prime_run.run_games([FakeGame(f"t{i}") for i in range(5)], cfg, llm, out, time.time() + 60)
    ok("all_games_alive_at_once", seen["max_live"] == 5, seen)
    ok("llm_never_over_slots", llm.max_inflight <= 2, llm.max_inflight)
    ok("summary_has_scheduler", summ.get("scheduler", {}).get("slots") == 2
       and len(summ["scheduler"]["per_game"]) == 5 and summ["scheduler"]["never_granted"] == [], summ.get("scheduler"))
    ok("scheduler_log_written", (out / "scheduler.jsonl").exists())
    seen.update(max_live=0, live=0)
    cfg_off = build_config({"concurrency": 2, "scheduler": {"enabled": False}})
    prime_run.play_game = lambda game, cfg, llm, run_dir, stop_event, soft_end, scheduler=None, lane=None: (
        ok("off_passes_no_scheduler", scheduler is None) or {"end_reason": "fake"})
    summ_off = prime_run.run_games([FakeGame("u0")], cfg_off, llm, Path(tempfile.mkdtemp()), time.time() + 60)
    ok("off_no_scheduler_in_summary", "scheduler" not in summ_off)
finally:
    prime_run.play_game, prime_run.trace.build_run = _real_play, _real_trace

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

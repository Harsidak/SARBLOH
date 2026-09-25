# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test for COMPONENT X: SCORE-CHURN EXPLOITER.

SCOPE NOTE: this component lives ONLY in `my_agent original.py`, the quarantined
metric-exploit build. It is deliberately absent from `my_agent.py`, which stays an
honest capability track. This file therefore loads the alternate build BY PATH
(its filename contains a space, so it cannot be imported normally) and skips
cleanly if that build is missing.

Doctrine: the component stays in the agent file; this file only feeds it inputs
whose correct answer we know by construction. Bar is "passes this fixed set".
Never delete a case.

WHAT IS BEING PINNED. The exploit rests on three facts read out of the shipped
sources, and a mistake in any of them silently converts the exploit into the
score-DESTROYING bug it is derived from:
  * base_game.handle_reset(): `_action_count == 0` -> full_reset() -> `_score = 0`
  * base_game.set_level() zeroes `_action_count`, and a level-up calls it
  * scorecard.Card.set_levels_completed() appends on ANY change, drops included,
    and _calculate_score() reads entry i as "level i+1, COMPLETED"
So the emitted cycle must be exactly: RESET (the 1-action wipe), then the FULL
recorded trajectory (the re-win) -- and it must stop once `win_levels` change
events exist, because the scorer reads no further.

Run:  python test_churn.py
"""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALT = os.path.join(HERE, "my agent original.py")
if not os.path.exists(ALT):
    ALT = os.path.join(HERE, "my_agent original.py")

if not os.path.exists(ALT):
    print(f"SKIP: quarantined build not found at {ALT}")
    sys.exit(0)

# The quarantined build is a verbatim notebook cell (line 1 is a `%%writefile`
# magic) under a filename containing a space. eval/agent_loader.py owns both
# workarounds so this test and the eval harness load it identically.
sys.path.insert(0, os.path.join(HERE, "eval"))
from agent_loader import load_agent_module      # noqa: E402

ChurnExploiter = load_agent_module(ALT).ChurnExploiter

FAILS = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def ok(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}" + (f": {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAILS.append(name)


def drive(ce, n, score_of):
    """Emit up to n scripted actions, feeding back the levels_completed that a
    deterministic engine would produce. `score_of(action, prev)` is the engine."""
    out, prev = [], 0
    for _ in range(n):
        a = ce.next_action()
        if a is None:
            break
        out.append(a)
        ce.note_action(a)
        s = score_of(a, prev)
        ce.note_score(s, prev)
        prev = s
    return out


print("=== 1. it stays out of the way until a level is actually won ===")

ce = ChurnExploiter(win_levels=8)
check("starts_recording", ce.phase, "record")
check("silent_while_recording", ce.next_action(), None)
for a in ("RESET", "ACTION6_r1_c1", "ACTION6_r2_c2"):
    ce.note_action(a)
check("history_is_recorded", ce._hist, ["RESET", "ACTION6_r1_c1", "ACTION6_r2_c2"])
# A score that does not move is not an event and must not arm anything.
ce.note_score(0, 0)
check("no_event_without_a_change", ce.events, 0)
check("still_recording", ce.phase, "record")


print("\n=== 2. the first win freezes the exact trajectory that produced it ===")

ce.note_score(1, 0)
check("win_counted_as_an_event", ce.events, 1)
check("armed_by_the_win", ce.phase, "armed")
# The replay is the suffix AFTER the last RESET. On level 1 a level_reset and a
# full_reset restore the same board, so that suffix reproduces the win exactly --
# and the leading RESET is dropped because the churn's own wipe-RESET already put
# the game there. Everything before it is dead weight the scorer charges for.
check("replay_is_the_suffix_after_the_last_reset", ce._replay,
      ["ACTION6_r1_c1", "ACTION6_r2_c2"])
# Recording stops: post-win actions belong to the churn, not to the win.
ce.note_action("ACTION6_r9_c9")
check("recording_stops_after_arming", len(ce._hist), 3)


print("\n=== 3. the emitted cycle is RESET-then-replay, in that order ===")

# The lone leading RESET is the whole exploit: it lands while the engine's
# _action_count is 0 (the level-up zeroed it), so it is a full_reset -> wipe.
first = [ce.next_action() for _ in range(3)]
check("cycle_starts_with_the_wipe_reset", first[0], "RESET")
check("cycle_continues_with_the_replay", first[1:],
      ["ACTION6_r1_c1", "ACTION6_r2_c2"])
check("phase_is_churn", ce.phase, "churn")
# The trim is the whole economics of the exploit: one cycle must cost exactly
# 1 (the wipe) + len(trimmed replay), never 1 + len(the full history). Draining
# `first` above consumed one whole cycle, so nothing may be left over.
ok("one_cycle_costs_wipe_plus_trimmed_replay",
   len(first) == 1 + len(ce._replay) and not ce._plan,
   f"cycle={first} replay={ce._replay} leftover={ce._plan}")


print("\n=== 3b. trimming is exact, not heuristic ===")

# Only the LAST RESET matters: everything before it was undone by it.
check("trim_takes_the_last_reset",
      ChurnExploiter._trim(["A", "RESET", "B", "RESET", "C", "D"]), ["C", "D"])
# No RESET in the history -> the history already starts at the initial state.
check("trim_is_identity_without_a_reset",
      ChurnExploiter._trim(["A", "B"]), ["A", "B"])
# A trailing RESET means the win came from the untouched initial board.
check("trim_of_a_trailing_reset_is_empty",
      ChurnExploiter._trim(["A", "RESET"]), [])
check("trim_of_nothing_is_nothing", ChurnExploiter._trim([]), [])


print("\n=== 4. end-to-end: the cycle produces one event per half-cycle ===")

ce2 = ChurnExploiter(win_levels=8)
for a in ("RESET", "ACTION6_r1_c1", "ACTION6_r2_c2"):
    ce2.note_action(a)
ce2.note_score(1, 0)


def engine(action, prev):
    """A deterministic stand-in for the real engine: a RESET at _action_count==0
    wipes to 0; replaying the recorded trajectory re-wins level 1."""
    engine.count = getattr(engine, "count", 0)
    if action == "RESET":
        if engine.count == 0:
            engine.seq = []
            return 0                       # full_reset
        engine.count = 0
        engine.seq = []
        return prev                        # level_reset: score survives
    engine.count += 1
    engine.seq = getattr(engine, "seq", []) + [action]
    if engine.seq == ["ACTION6_r1_c1", "ACTION6_r2_c2"]:
        engine.count = 0                   # set_level() zeroes it on a level-up
        return prev + 1
    return prev


engine.count = 0
engine.seq = []
emitted = drive(ce2, 200, engine)
ok("churn_terminates", len(emitted) < 200, f"{len(emitted)} actions emitted")
check("banked_exactly_win_levels_events", ce2.events, 8)
check("stops_when_spent", ce2.phase, "spent")
check("nothing_emitted_once_spent", ce2.next_action(), None)
# The point of the exercise: 8 credited "levels" for a handful of actions.
ok("cost_is_small", len(emitted) <= 40, f"{len(emitted)} actions for 8 events")


print("\n=== 5. the kill switch really kills it ===")

os.environ["ARC_NO_CHURN"] = "1"
try:
    ce3 = ChurnExploiter(win_levels=8)
    ce3.note_action("ACTION6_r1_c1")
    ce3.note_score(1, 0)
    check("disabled_records_nothing", ce3._hist, [])
    check("disabled_counts_no_events", ce3.events, 0)
    check("disabled_emits_nothing", ce3.next_action(), None)
    check("disabled_never_arms", ce3.phase, "record")
finally:
    del os.environ["ARC_NO_CHURN"]


print("\n=== 6. degenerate inputs do not produce a runaway ===")

# win_levels unknown (0) must not mean "churn forever with no stop condition"
# in a way that crashes; it means "no cap known", and the action budget stops it.
ce4 = ChurnExploiter(win_levels=0)
ce4.note_action("ACTION6_r5_c5")
ce4.note_score(1, 0)
out = [ce4.next_action() for _ in range(5)]
ok("uncapped_still_emits_a_valid_cycle", out[0] == "RESET" and None not in out,
   f"{out}")

# A win with an EMPTY history (score moved before any action was emitted) must
# not emit a bare RESET loop -- that is the score-wipe bug with no re-win.
ce5 = ChurnExploiter(win_levels=8)
ce5.note_score(1, 0)
check("empty_history_replay", ce5._replay, [])
seq = [ce5.next_action() for _ in range(4)]
ok("empty_history_is_all_resets", all(s == "RESET" for s in seq), f"{seq}")


print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)

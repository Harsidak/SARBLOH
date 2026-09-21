# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 5.13: ReplayCache + efficiency governor.

Doctrine: the component stays in my_agent.py; this file only imports it and
feeds hand-built trajectories whose correct answer we know by construction.

WHY THIS COMPONENT IS DANGEROUS AND THEREFORE HEAVILY PINNED. It is the only
part of the agent that issues SCORED actions from a recording rather than from
a decision. If it re-issues a route into a world that has moved on, it spends
real budget on nonsense -- the exact failure RHAE squares. Its entire safety
argument is one property:

    a route step is issued ONLY when the live frame still matches the frame
    that step was recorded from, HUD-masked; the first mismatch retires the
    whole route.

so the worst case is one wasted action. That property, the cycle compression
that must never invent a shortcut, and the arm/disarm rules around deaths and
score wipes are what this file pins.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_replay.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_agent import ReplayCache                                    # noqa: E402

FAILS = []
WARNS = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def ok(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}  [{detail}]")
        FAILS.append(name)


def G(v, n=6):
    """A tiny board whose single cell carries the state id."""
    g = np.zeros((n, n), dtype=np.int8)
    g[0, 0] = v
    return g


def hud(v, n=6, t=0):
    """Same board, with a live 'timer' cell that a HUD mask should ignore."""
    g = G(v, n)
    g[n - 1, n - 1] = t
    return g


def MASK(n=6):
    m = np.zeros((n, n), dtype=bool)
    m[n - 1, n - 1] = True
    return m


def walk(rc, states, actions):
    """Record a straight-line trajectory: state[i] --action[i]--> state[i+1]."""
    for s, a in zip(states, actions):
        rc.note_action(s, a)


ALL = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"}

# =====================================================================
print("--- 1. the efficiency governor: cycles are excised ------------------")
# =====================================================================
# Walk 1 -> 2 -> 3 -> 2 -> 4. Visiting 2 twice means everything between the two
# visits returned the world to where it already was, so it bought nothing.
rc = ReplayCache()
walk(rc, [G(1), G(2), G(3), G(2)], ["a", "b", "c", "d"])
rc.note_level_up(1)
route = rc.routes[1]
check("cycle_excised_len", len(route), 2)
check("cycle_excised_actions", [a for _g, a in route], ["a", "d"])
check("compressed_out_counted", rc.stats["compressed_out"], 2)
check("stored_counted", rc.stats["stored"], 1)

# A trajectory with no repeat must survive untouched. Over-compressing is the
# one error this class cannot recover from cheaply, so it is pinned hard.
rc2 = ReplayCache()
walk(rc2, [G(1), G(2), G(3)], ["a", "b", "c"])
rc2.note_level_up(1)
check("acyclic_untouched", [a for _g, a in rc2.routes[1]], ["a", "b", "c"])
check("acyclic_compressed_out", rc2.stats["compressed_out"], 0)

# Repeated visits to the START are excised too (the common livelock shape).
rc3 = ReplayCache()
walk(rc3, [G(1), G(2), G(1), G(2), G(3)], ["a", "b", "c", "d", "e"])
rc3.note_level_up(1)
check("start_cycle_excised", [a for _g, a in rc3.routes[1]], ["c", "d", "e"])

# The governor compares RAW, so a live HUD simply prevents compression rather
# than inventing a shortcut between two states that only look alike.
rc4 = ReplayCache()
walk(rc4, [hud(1, t=1), hud(2, t=2), hud(1, t=3)], ["a", "b", "c"])
rc4.note_level_up(1)
check("live_hud_blocks_compression", len(rc4.routes[1]), 3)

# =====================================================================
print("\n--- 2. replay re-issues a stored route ------------------------------")
# =====================================================================
rc = ReplayCache()
walk(rc, [G(1), G(2), G(3)], ["ACTION1", "ACTION2", "ACTION3"])
rc.note_level_up(1)
ok("not_replaying_after_levelup", not rc.replaying())

rc.note_wipe(0)                       # full_reset wiped level 1; re-walk it
ok("wipe_arms_stored_route", rc.replaying(), f"level={rc._level}")
check("replay_step1", rc.next_action(G(1), ALL), "ACTION1")
check("replay_step2", rc.next_action(G(2), ALL), "ACTION2")
check("replay_step3", rc.next_action(G(3), ALL), "ACTION3")
ok("replay_exhausted", not rc.replaying())
check("replayed_counted", rc.stats["replayed"], 3)
check("no_divergence", rc.stats["diverged"], 0)

# =====================================================================
print("\n--- 3. THE SAFETY PROPERTY: divergence costs one action -------------")
# =====================================================================
rc = ReplayCache()
walk(rc, [G(1), G(2), G(3)], ["ACTION1", "ACTION2", "ACTION3"])
rc.note_level_up(1)
rc.note_wipe(0)
check("diverge_step1_ok", rc.next_action(G(1), ALL), "ACTION1")
# The world is not where the recording says it should be.
check("diverge_returns_none", rc.next_action(G(9), ALL), None)
ok("diverge_stops_replay", not rc.replaying())
check("diverge_counted", rc.stats["diverged"], 1)
ok("diverged_route_retired", 1 in rc.retired)
ok("diverged_route_dropped", 1 not in rc.routes)
# ...and it is never armed again, so the wasted action is paid at most once.
rc.note_wipe(0)
ok("retired_route_never_rearmed", not rc.replaying())

# A shape change (new layout) must read as divergence, not raise.
rc = ReplayCache()
walk(rc, [G(1), G(2)], ["ACTION1", "ACTION2"])
rc.note_level_up(1)
rc.note_wipe(0)
check("shape_change_is_divergence", rc.next_action(G(1, n=8), ALL), None)

# An action the game no longer offers is divergence too: re-issuing it would
# resolve to something else entirely.
rc = ReplayCache()
walk(rc, [G(1), G(2)], ["ACTION5", "ACTION2"])
rc.note_level_up(1)
rc.note_wipe(0)
check("unavailable_action_is_divergence",
      rc.next_action(G(1), {"ACTION1", "ACTION2"}), None)

# =====================================================================
print("\n--- 4. comparison is HUD-masked, both sides, at replay time ---------")
# =====================================================================
# Recorded with the timer reading 1; replayed while it reads 7. The board is
# the same board. Masking must be applied to BOTH sides or every game with a
# live HUD diverges on step 1 -- the asymmetry that once broke the world model.
rc = ReplayCache()
walk(rc, [hud(1, t=1), hud(2, t=2)], ["ACTION1", "ACTION2"])
rc.note_level_up(1)
rc.note_wipe(0)
check("masked_match_replays", rc.next_action(hud(1, t=7), ALL, MASK()), "ACTION1")
# Without the mask the same pair is a genuine mismatch.
rc2 = ReplayCache()
walk(rc2, [hud(1, t=1), hud(2, t=2)], ["ACTION1", "ACTION2"])
rc2.note_level_up(1)
rc2.note_wipe(0)
check("unmasked_same_pair_diverges", rc2.next_action(hud(1, t=7), ALL, None), None)
# Masking may NOT hide a real difference in the play area.
rc3 = ReplayCache()
walk(rc3, [hud(1, t=1), hud(2, t=2)], ["ACTION1", "ACTION2"])
rc3.note_level_up(1)
rc3.note_wipe(0)
check("mask_does_not_hide_play_area",
      rc3.next_action(hud(5, t=1), ALL, MASK()), None)

# =====================================================================
print("\n--- 5. deaths: the walk back to the frontier ------------------------")
# =====================================================================
# arcengine level_reset()s on GAME_OVER, so the level restarts and the walk to
# the point of death is spent again. Replay it -- minus the action that killed.
rc = ReplayCache()
walk(rc, [G(1), G(2), G(3), G(4), G(5)],
     ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"])
rc.note_death(0)
ok("death_arms_frontier", rc.replaying())
check("frontier_counted", rc.stats["frontier"], 1)
check("fatal_action_dropped", len(rc._replay), 4)
check("frontier_is_prefix", [a for _g, a in rc._replay],
      ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
check("frontier_first_step", rc.next_action(G(1), ALL), "ACTION1")

# A frontier route is NOT a level route, so divergence must not retire a level.
rc.next_action(G(99), ALL)
ok("frontier_divergence_retires_nothing", not rc.retired, str(rc.retired))

# Too short to be worth the risk.
rc = ReplayCache()
walk(rc, [G(1), G(2)], ["ACTION1", "ACTION2"])
rc.note_death(0)
ok("short_frontier_not_armed", not rc.replaying(), f"len={len(rc._replay)}")

# The dead life's partial segment must not leak into the next route.
rc = ReplayCache()
walk(rc, [G(1), G(2), G(3), G(4), G(5)], ["a", "b", "c", "d", "e"])
rc.note_death(0)
walk(rc, [G(1), G(7)], ["x", "y"])
rc.note_level_up(1)
check("segment_cleared_by_death", [a for _g, a in rc.routes[1]], ["x", "y"])

# =====================================================================
print("\n--- 6. bookkeeping that keeps routes honest -------------------------")
# =====================================================================
# Replayed actions are recorded too: the banked route must start at the level
# start, not in the middle of it.
rc = ReplayCache()
walk(rc, [G(1), G(2)], ["ACTION1", "ACTION2"])
rc.note_level_up(1)
rc.note_wipe(0)
a1 = rc.next_action(G(1), ALL)
rc.note_action(G(1), a1)
a2 = rc.next_action(G(2), ALL)
rc.note_action(G(2), a2)
walk(rc, [G(3)], ["ACTION4"])
rc.note_level_up(2)
check("replayed_steps_are_in_the_next_route",
      [a for _g, a in rc.routes[2]], ["ACTION1", "ACTION2", "ACTION4"])

# A level's route is written once. Re-banking must not overwrite a route that
# already worked with a worse one.
rc = ReplayCache()
walk(rc, [G(1)], ["short"])
rc.note_level_up(1)
rc.note_wipe(0)
walk(rc, [G(1), G(2), G(3)], ["long1", "long2", "long3"])
rc.note_level_up(1)
check("first_route_wins", [a for _g, a in rc.routes[1]], ["short"])

# Completing a level while replaying toward it is counted, and the cache chains
# to the next stored route so a multi-level wipe is re-walked end to end.
rc = ReplayCache()
walk(rc, [G(1)], ["ACTION1"])
rc.note_level_up(1)
walk(rc, [G(2)], ["ACTION2"])
rc.note_level_up(2)
rc.note_wipe(0)
check("wipe_arms_level_1", rc._level, 1)
check("chained_route_level_1", rc.next_action(G(1), ALL), "ACTION1")
rc.note_level_up(1)
check("completed_counted", rc.stats["completed"], 1)
check("chains_to_level_2", rc._level, 2)
check("chained_route_step", rc.next_action(G(2), ALL), "ACTION2")

# An ordinary level-up (no stored route ahead) must not arm anything.
rc = ReplayCache()
walk(rc, [G(1)], ["a"])
rc.note_level_up(1)
ok("levelup_without_stored_route_does_not_arm", not rc.replaying())

# =====================================================================
print("\n--- 7. degenerate inputs never raise --------------------------------")
# =====================================================================
rc = ReplayCache()
rc.note_level_up(1)
ok("empty_segment_stores_nothing", 1 not in rc.routes)
rc.note_death(0)
ok("death_with_no_segment_is_safe", not rc.replaying())
rc.note_wipe(0)
ok("wipe_with_no_routes_is_safe", not rc.replaying())
check("next_action_when_idle", rc.next_action(G(1), ALL), None)

rc = ReplayCache()
rc.note_action("not a grid", "a")             # must not raise
ok("unrecordable_frame_drops_segment", rc._seg == [])

rc = ReplayCache()
walk(rc, [G(1)], ["a"])
rc.note_level_up(1)
rc.note_wipe(0)
check("empty_valid_set_is_divergence", rc.next_action(G(1), set()), None)

# The cap exists so a 1500-action flail cannot be stored as a "route".
rc = ReplayCache()
for i in range(ReplayCache.MAX_SEGMENT + 50):
    rc.note_action(G(i % 100), f"a{i}")
ok("segment_capped", len(rc._seg) <= ReplayCache.MAX_SEGMENT + 1,
   f"len={len(rc._seg)}")

rc = ReplayCache()
walk(rc, [G(1)], ["a"])
rc.new_game()
ok("new_game_clears_everything",
   not rc.routes and not rc._seg and not rc.replaying() and not rc.retired)

# =====================================================================
print("\n--- 8. the agent wires it up ----------------------------------------")
# =====================================================================
from my_agent import MyAgent                                        # noqa: E402

_a = MyAgent(game_id="unit-test")
ok("agent_owns_a_replay_cache", isinstance(getattr(_a, "replay", None), ReplayCache))
ok("kill_switch_documented", "ARC_NO_REPLAY" in open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_agent.py"),
    encoding="utf-8").read())

print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

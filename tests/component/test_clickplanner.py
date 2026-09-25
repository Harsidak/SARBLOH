# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 5.6: CLICK PLANNER.

Doctrine: the component STAYS in my_agent.py. This file only imports it and
feeds hand-built inputs whose correct answer we know by construction, so a
failure is visible for the ClickPlanner alone -- no game engine, no LLM.

WHY THIS FILE EXISTS. 10 of the 25 real games are click-only, and this was the
largest component in the agent with no test of its own. The first per-branch
measurement (scratchpad/route_diag.py) put 67% of lp85's entire action budget
inside this planner's search, and the CP_DEBUG log then showed that search
destroying six won levels in a single run.

THE INVARIANT THIS FILE EXISTS TO PIN (the RESET/score contract, read out of
arcengine/base_game.py rather than guessed):
  * `_action_count` increments on every NON-reset action            (L279)
  * `set_level` zeroes it -- and `level_reset` calls `set_level`    (L160)
  * `handle_reset`: `_action_count == 0` -> `full_reset()`          (L313)
  * `full_reset` sets `_score = 0`                                  (L321)
=> TWO RESETS WITH NO ACTION BETWEEN THEM WIPE EVERY LEVEL ALREADY WON.
The planner cannot read that counter (on Kaggle the engine is remote), so it
mirrors it in `_acts_since_reset` and must never emit the second RESET.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_clickplanner.py
"""
import sys
from collections import deque

import numpy as np

from my_agent import ClickPlanner, StateEncoder

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
        print(f"  PASS  {name}" + (f": {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAILS.append(name)


def board(seed=0):
    """A deterministic non-uniform 64x64 board (uniform frames hash alike)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 4, size=(64, 64)).astype(np.int32)


print("=== 1. the mirror tracks the engine's action counter ===")

cp = ClickPlanner()
check("mirror_starts_zero", cp._acts_since_reset, 0)
cp.note_action("ACTION6_r1_c1")
cp.note_action("ACTION6_r2_c2")
check("mirror_counts_non_reset", cp._acts_since_reset, 2)
cp.note_action("RESET")
check("mirror_zeroed_by_reset", cp._acts_since_reset, 0)
cp.note_action("ACTION5")
check("mirror_counts_any_action_not_just_clicks", cp._acts_since_reset, 1)
# An empty action string is not an action; it must not move the mirror, or a
# no-op step would make a dangerous RESET look safe.
cp.note_action("")
check("mirror_ignores_empty", cp._acts_since_reset, 1)

# A level-up zeroes the engine's counter via set_level WITHOUT any RESET being
# emitted. If the mirror missed that, the planner would believe an action had
# been spent and would emit a RESET that full-resets the fresh score.
cp2 = ClickPlanner()
cp2.note_action("ACTION6_r1_c1")
cp2.note_action("ACTION6_r2_c2")
cp2.new_level()
check("mirror_zeroed_by_level_up", cp2._acts_since_reset, 0)


print("\n=== 2. the search never emits back-to-back RESETs (the score wipe) ===")

# Drive the real search loop. Entering search stages a lone ["RESET"]; the very
# next plan is ["RESET", *path, b]. Before the fix those two RESETs landed with
# nothing between them, which is a full_reset -> score 0.
cp3 = ClickPlanner()
cp3._enter_search([(30, 2), (30, 58)])
cp3._acts_since_reset = 60          # coverage ran first, as _reset_search requires

emitted = []
b = board(1)
for i in range(12):
    step = cp3._search_step(b)
    if step is None:
        break
    emitted.append(step)
    cp3.note_action(step)
    # Each distinct plan lands on a distinct board, as a real button puzzle would.
    b = board(2 + i)

ok("search_emitted_something", len(emitted) >= 3, f"{emitted[:6]}")
pairs = [(emitted[i], emitted[i + 1]) for i in range(len(emitted) - 1)]
ok("no_back_to_back_resets",
   not any(x == "RESET" and y == "RESET" for x, y in pairs),
   f"emitted={emitted[:8]}")
# The first RESET is legitimate and must still happen: 60 coverage clicks moved
# the board, so returning to the level's initial state is real work.
check("first_step_is_the_legitimate_reset", emitted[0], "RESET")

# The root probe must not cost a second action once we are already at the root.
cp4 = ClickPlanner()
cp4._enter_search([(30, 2), (30, 58)])
cp4._acts_since_reset = 0           # already at the level's initial state
first = cp4._search_step(board(3))
ok("no_reset_when_already_at_root", first != "RESET",
   f"got {first!r} -- a RESET here would read as _action_count==0 -> full_reset")


print("\n=== 3. the root is hashed from the board that really is the root ===")

# When the lone-RESET root probe is skipped, the board in hand IS the root. The
# planner must settle `_pending` against THAT board -- not drop through to
# coverage and hash the root from a frame some later click has already moved.
cp5 = ClickPlanner()
cp5._enter_search([(30, 2), (30, 58)])
cp5._acts_since_reset = 0
root_board = board(7)
cp5._search_step(root_board)
ok("root_recorded", cp5._root is not None, f"root={cp5._root}")
check("root_is_the_board_in_hand", cp5._root, cp5._hash(root_board))
ok("root_enqueued_for_expansion", cp5._root in cp5._dist,
   f"dist keys={list(cp5._dist)[:3]}")


print("\n=== 4. skipping a RESET does not desync the replay bookkeeping ===")

# The edge under test must still be attributed to the parent the plan named.
# A skipped RESET that left `_plan_i` wrong would record the child under the
# wrong parent -- silent graph corruption, far worse than a wasted action.
cp6 = ClickPlanner()
cp6._enter_search([(30, 2), (30, 58)])
cp6._acts_since_reset = 0
cp6._search_step(board(7))                    # settles root, stages first edge
pend = cp6._pending
ok("pending_is_an_expand", pend is not None and pend[0] == "expand", f"{pend}")
if pend is not None and pend[0] == "expand":
    parent, btn = pend[1], pend[2]
    check("expand_parent_is_root", parent, cp6._root)
    # Finish the staged plan, then hand back a distinct child board.
    guard = 0
    while cp6._plan_i < len(cp6._plan) and guard < 8:
        st = cp6._search_step(board(7))
        if st is None:
            break
        cp6.note_action(st)
        guard += 1
    child = board(9)
    cp6._process_result(cp6._hash(child))
    check("edge_recorded_under_named_parent",
          cp6._G.get(parent, {}).get(btn), cp6._hash(child))


print("\n=== 5. regression guard: search still terminates and yields clicks ===")

# The skip logic introduced a loop. A plan of nothing but skippable RESETs must
# not spin: every plan past the root carries a button, so the loop always breaks
# on a real click.
cp7 = ClickPlanner()
cp7._enter_search([(30, 2), (30, 58)])
cp7._acts_since_reset = 0
clicks, resets, guard = 0, 0, 0
# A FIXED board: every button leads back to the same state, so the reachable
# graph is a single node and dedup must exhaust it. (Feeding a fresh board every
# step instead would model a game with infinite states -- the search would be
# right to keep going, and the case would be testing the harness, not the code.)
bb = board(11)
while guard < 40:
    st = cp7._search_step(bb)
    guard += 1
    if st is None:
        break
    cp7.note_action(st)
    if st == "RESET":
        resets += 1
    else:
        clicks += 1
ok("search_terminates_on_exhausted_graph", guard < 40,
   f"{guard} steps; search_done={cp7._search_done}")
ok("search_yields_clicks", clicks > 0, f"{clicks} clicks / {resets} resets")
# The whole point of the fix: resets are now a minority of the search's spend.
ok("resets_are_not_the_majority", resets <= clicks,
   f"{resets} resets vs {clicks} clicks")


print("\n=== 6. fresh_cell: the escalation modality rung ===")

# Not changed by this work, but it is the #2 consumer of lp85's budget (269
# actions at a 1.9% change rate) and had no test at all. Pin what it promises.
cp8 = ClickPlanner()
g = board(5)
seen = set()
for _ in range(50):
    a = cp8.fresh_cell(g)
    if a is None:
        break
    seen.add(a)
    rc = cp8._parse(a)
    if rc:
        cp8._exact.add(rc)
ok("fresh_cell_never_repeats_a_pixel", len(seen) == len(cp8._exact),
   f"{len(seen)} distinct offers, {len(cp8._exact)} pixels marked")
ok("fresh_cell_in_bounds",
   all(0 <= cp8._parse(a)[0] < 64 and 0 <= cp8._parse(a)[1] < 64 for a in seen),
   f"{len(seen)} offers all inside 64x64")
# Lethal cells are skipped: dying again is not new information.
cp9 = ClickPlanner()
cp9.mark_loss("ACTION6_r30_c30")
offers = {cp9.fresh_cell(board(5)) for _ in range(200)} - {None}
ok("fresh_cell_avoids_lethal",
   "ACTION6_r30_c30" not in offers,
   f"{len(offers)} distinct offers, none lethal")


print("\n=== 7. the click WORLD MODEL: navigate, don't teleport (objective O2) ===")

# `_G` is an exact record of observed transitions. Until now it was BUILT and
# never USED for movement: every frontier probe was priced as a fresh replay from
# the root, ["RESET", *path, button], so discovering one edge cost depth+2 SCORED
# actions even when its parent was the board already on screen. RHAE squares the
# action ratio, so that is the single most expensive habit in the planner.

cpA = ClickPlanner()
cpA._enter_search([(30, 2), (30, 58)])
cpA._acts_since_reset = 5           # a RESET here would cost a real action
# Hand-build a known graph:  root -0-> s1 -0-> s2   (and root -1-> s3)
root, s1, s2, s3 = 111, 222, 333, 444
cpA._root = root
cpA._dist = {root: (), s1: (0,), s2: (0, 0), s3: (1,)}
# s2's button 0 is already known (a self-loop), so button 1 is its only untried
# edge -- that is the probe the planner must pick.
cpA._G = {root: {0: s1, 1: s3}, s1: {0: s2}, s2: {0: s2}}
cpA._queue = deque([root, s1, s2, s3])

# Standing on s2, whose button 1 is untried. Teleporting would cost
# RESET + 0 + 0 + 1 = 4 actions. We are ALREADY on s2, so it costs 1.
cpA._cur = s2
ok("nav_dists_finds_self", cpA._nav_dists().get(s2) == (),
   f"{ {k: v for k, v in cpA._nav_dists().items()} }")
cpA._build_next_plan()
check("probe_from_here_is_one_action", cpA._plan, [1])
ok("no_reset_in_a_navigation_plan", "RESET" not in cpA._plan, f"{cpA._plan}")
check("probe_attributed_to_the_state_we_stand_on", cpA._pending, ("expand", s2, 1))
ok("saving_is_accounted", cpA._saved >= 3, f"saved={cpA._saved} actions")

# Standing on the root, the cheapest untried probe is one step away: root's own
# button 1 is taken (-> s3), so s1's button 1 costs nav(0) + 1 = 2, beating the
# teleport price of RESET + 0 + 1 = 3.
cpB = ClickPlanner()
cpB._enter_search([(30, 2), (30, 58)])
cpB._acts_since_reset = 5
cpB._root = root
cpB._dist = {root: (), s1: (0,), s3: (1,)}
cpB._G = {root: {0: s1, 1: s3}}
cpB._queue = deque([root, s1, s3])
cpB._cur = root
cpB._build_next_plan()
ok("walks_a_known_edge_instead_of_resetting",
   "RESET" not in cpB._plan and len(cpB._plan) == 2, f"{cpB._plan}")

# When the current board is NOT in the graph, navigation is impossible and the
# planner must fall back to the teleport -- never invent a path.
cpC = ClickPlanner()
cpC._enter_search([(30, 2), (30, 58)])
cpC._acts_since_reset = 5
cpC._root = root
cpC._dist = {root: (), s1: (0,)}
cpC._G = {root: {0: s1}, s1: {0: s1}}     # s1's untried edge is button 1
cpC._queue = deque([s1])
cpC._cur = 999999                    # a board never seen before
cpC._build_next_plan()
check("unknown_board_falls_back_to_teleport", cpC._plan[0], "RESET")
check("teleport_uses_the_root_path", cpC._plan, ["RESET", 0, 1])

# O4: a lethal edge is never planned through, by either route.
cpD = ClickPlanner()
cpD._enter_search([(30, 2), (30, 58)])
cpD._acts_since_reset = 5
cpD._root = root
cpD._dist = {root: (), s1: (0,)}
cpD._G = {root: {0: s1}}
cpD._queue = deque([s1])
cpD._dead_edges = {(s1, 0), (s1, 1)}     # both of s1's buttons are fatal
cpD._cur = s1
built = cpD._build_next_plan()
ok("fully_lethal_frontier_is_retired", built is False,
   f"built={built} plan={cpD._plan}")


print("\n=== 8. the world model is FALSIFIABLE (objective O1) ===")

# A recorded edge is only valid while the game really is deterministic and has no
# hidden state -- tu93 proved some games do. A navigation plan therefore carries
# the board it PREDICTS at each step, and a mismatch must delete the offending
# edge, not silently record the probe under a parent we are not standing on.

gA, gB, gC = board(21), board(22), board(23)
hA, hB, hC = ClickPlanner._hash(gA), ClickPlanner._hash(gB), ClickPlanner._hash(gC)

cpE = ClickPlanner()
cpE._enter_search([(30, 2), (30, 58)])
cpE._acts_since_reset = 5
cpE._root = hA
cpE._dist = {hA: (), hB: (0,), hC: (0, 0)}
cpE._G = {hA: {0: hB}, hB: {0: hC}}
cpE._queue = deque([hC])
cpE._cur = hA
cpE._build_next_plan()
check("nav_plan_staged", cpE._plan, [0, 0, 0])
check("predictions_recorded", cpE._expect, [hB, hC])

# Step 1 lands where predicted -> no falsification, plan continues.
cpE._plan_i = 1
cpE._search_step(gB)
check("truthful_step_is_not_falsified", cpE._falsified, 0)
ok("edge_survives_a_correct_prediction", cpE._G.get(hA, {}).get(0) == hB,
   f"{cpE._G.get(hA)}")

# Step 2 lands somewhere else -> the edge hB -0-> hC is a lie. Delete it.
cpF = ClickPlanner()
cpF._enter_search([(30, 2), (30, 58)])
cpF._acts_since_reset = 5
cpF._root = hA
cpF._dist = {hA: (), hB: (0,), hC: (0, 0)}
cpF._G = {hA: {0: hB}, hB: {0: hC}}
cpF._queue = deque([hC])
cpF._cur = hA
cpF._build_next_plan()
cpF._plan_i = 2                      # both nav steps emitted
cpF._search_step(board(99))          # reality: a board the model did not predict
check("divergence_counted", cpF._falsified, 1)
ok("lying_edge_deleted", 0 not in cpF._G.get(hB, {}), f"{cpF._G.get(hB)}")
ok("truthful_edge_kept", cpF._G.get(hA, {}).get(0) == hB, f"{cpF._G.get(hA)}")
ok("predictions_dropped_on_divergence", cpF._expect is None, f"expect={cpF._expect}")
# The in-flight probe must NOT have been recorded. Attributing a result to a
# parent we are demonstrably not standing on is silent graph corruption, strictly
# worse than a wasted action -- so `_G` may only have SHRUNK by the deleted lie.
check("no_edge_invented_by_the_divergence", cpF._G, {hA: {0: hB}, hB: {}})
# Re-planning is allowed (and expected) -- but from the board really on screen.
# That board is unknown to the graph, so the only legal route is the teleport.
ok("replans_via_teleport_from_an_unknown_board",
   cpF._plan and cpF._plan[0] == "RESET", f"plan={cpF._plan}")


print("\n=== 9. regression: navigation did not break termination or the wipe guard ===")

# Re-run case 5's contract with the world model live: a fixed board (every button
# returns to the same state) must still exhaust, still yield clicks, and still
# never emit back-to-back RESETs.
cpG = ClickPlanner()
cpG._enter_search([(30, 2), (30, 58)])
cpG._acts_since_reset = 0
emitted, guard = [], 0
bb = board(31)
while guard < 60:
    st = cpG._search_step(bb)
    guard += 1
    if st is None:
        break
    emitted.append(st)
    cpG.note_action(st)
ok("still_terminates", guard < 60, f"{guard} steps, done={cpG._search_done}")
ok("still_yields_clicks", any(s != "RESET" for s in emitted), f"{emitted[:8]}")
pairs = [(emitted[i], emitted[i + 1]) for i in range(len(emitted) - 1)]
ok("still_no_back_to_back_resets",
   not any(x == "RESET" and y == "RESET" for x, y in pairs), f"{emitted[:8]}")

# And on a graph that really does branch, the world model must make the search
# CHEAPER than the teleport-only version it replaced -- that is the whole point.
class _Sim:
    """A deterministic 2-button toy: state is an (a, b) counter pair, capped."""
    def __init__(self):
        self.s = (0, 0)
    def reset(self):
        self.s = (0, 0)
    def click(self, b):
        a, c = self.s
        self.s = (min(a + 1, 2), c) if b == 0 else (a, min(c + 1, 2))
    def grid(self):
        g = np.zeros((8, 8), dtype=np.int32)
        g[0, 0], g[1, 1] = self.s
        return g

def drive(planner, sim, budget):
    """Run the planner against the toy; return the SCORED actions it spent."""
    planner._enter_search([(30, 2), (30, 58)])
    planner._acts_since_reset = 7
    spent = 0
    for _ in range(budget):
        st = planner._search_step(sim.grid())
        if st is None:
            break
        spent += 1
        planner.note_action(st)
        if st == "RESET":
            sim.reset()
        else:
            sim.click(planner._buttons.index(planner._parse(st)))
    return spent

cpH = ClickPlanner()
spent_wm = drive(cpH, _Sim(), 400)
ok("world_model_explores_the_toy", len(cpH._dist) == 9,
   f"{len(cpH._dist)} distinct states found (3x3 grid of counters)")
ok("world_model_prefers_navigation", cpH._nav_plans > cpH._reset_plans,
   f"nav={cpH._nav_plans} teleport={cpH._reset_plans}")
ok("world_model_saved_actions", cpH._saved > 0,
   f"{cpH._saved} scored actions avoided; total spend {spent_wm}")
# Teleport-only lower bound: 9 states x 2 buttons, each replayed from the root at
# its BFS depth (0,1,1,2,2,2,3,3,4 for a 3x3 lattice) plus the RESET and the probe.
_teleport_lb = sum(2 * (d + 2) for d in (0, 1, 1, 2, 2, 2, 3, 3, 4))
ok("cheaper_than_teleporting", spent_wm < _teleport_lb,
   f"{spent_wm} actions vs teleport-only lower bound {_teleport_lb}")


print("\n=== 10. exhaustion refutes the BUTTON SET, not planning ===")

# Measured on lp85 level 2: search exhausted a 2-button graph (60 states, all 120
# edges) in 180 actions, then coverage burned the remaining 1320 at a 1.5% frame-
# change rate. A fully explored graph over button set B proves the level is not
# reachable through B -- it says nothing about a B that includes a button coverage
# had not yet found. Treating exhaustion as terminal throws that distinction away.
cpX = ClickPlanner()
b2 = [(30, 6), (30, 58)]
check("a_fresh_set_is_live", cpX._abstraction_is_live(b2), True)

cpX._exhausted_sets.append(frozenset(b2))
check("an_exhausted_set_is_refuted", cpX._abstraction_is_live(b2), False)
# A subset can only reach boards the superset already reached -> also refuted.
check("a_subset_of_a_refuted_set_is_refuted",
      cpX._abstraction_is_live([(30, 6)]), False)
# One new button = a genuinely different model of the action space. Worth budget.
# (2, 2) and not the (26, 14) this originally used: buttons are compared under the
# detector's own clustering radius, and (26,14) is Chebyshev 8 from (30,6) -- inside
# BUTTON_CLUSTER_DIST=10, so _detect_buttons calls them ONE button. The check's intent
# is "a genuinely NEW button", so it needs a coordinate that genuinely is one. The
# near-duplicate case it used to (accidentally) cover is now pinned explicitly in
# section 12 under near_duplicate_*.
check("a_set_with_a_new_button_is_live",
      cpX._abstraction_is_live(b2 + [(2, 2)]), True)
# Disjoint sets were never tested at all.
check("a_disjoint_set_is_live", cpX._abstraction_is_live([(2, 2), (4, 4)]), True)
# Refutations must accumulate, not replace: both sets stay refuted.
cpX._exhausted_sets.append(frozenset(b2 + [(26, 14)]))
cpX._exhausted_sets.append(frozenset(b2 + [(2, 2)]))
check("refutations_accumulate", cpX._abstraction_is_live(b2 + [(2, 2)]), False)
check("earlier_refutation_survives", cpX._abstraction_is_live(b2), False)
# Bounded by construction: each re-entry needs a strictly new button, and buttons
# are capped, so this cannot loop forever.
ok("re_entry_needs_a_strictly_new_button",
   not cpX._abstraction_is_live(list(cpX._exhausted_sets[-1])),
   f"{len(cpX._exhausted_sets)} sets refuted")

# A new LEVEL is a new layout, so old refutations do not transfer.
cpX.new_level()
check("new_level_clears_refutations", cpX._exhausted_sets, [])
check("new_level_makes_the_old_set_live_again", cpX._abstraction_is_live(b2), True)

# The exhaustion path itself must record the set it just refuted.
cpY = ClickPlanner()
cpY._enter_search([(1, 1), (2, 2)])
cpY._plan, cpY._plan_i, cpY._pending = [], 0, None
cpY._queue = deque()                      # nothing left to expand -> exhaustion
_g = np.zeros((8, 8), dtype=np.int16)
_out = cpY._search_step(_g)
check("exhaustion_yields_to_coverage", _out, None)
check("exhaustion_records_the_refuted_set",
      cpY._exhausted_sets, [frozenset([(1, 1), (2, 2)])])
check("exhaustion_leaves_search", cpY.searching, False)
check("exhaustion_returns_to_discover", cpY._phase, "discover")

# O4: entering search emits a RESET, and a RESET at action_count == 0 is a FULL
# reset that destroys every level already won. The mirror must veto that entry.
cpZ = ClickPlanner()
cpZ._total_clicks = 999
cpZ.note_action("RESET")                  # counter now reads 0
check("reset_mirror_reads_zero", cpZ._acts_since_reset, 0)
cpZ._reactive = {i: rc for i, rc in enumerate([(1, 1), (40, 40)])}
_before = cpZ._phase
cpZ.act(_g, StateEncoder(), 0.0)
check("never_enters_search_on_a_fresh_counter", cpZ._phase, _before)


print("\n=== 11. a RE-ENTERED search never emits a RESET (O4) ===")

# MEASURED REGRESSION, 2026-08-01: the first version of re-entry called
# _enter_search() unchanged, so it opened with a RESET to establish the root. On
# lp85 that RESET was read by the engine as a full_reset and WIPED a banked level
# ("SCORE REGRESSION 1->0 prev_action='RESET'") even though the planner's mirror
# read 300 actions since the last RESET. The mirror cannot be trusted mid-level,
# so the count of RESETs a re-entry is allowed to gamble on is ZERO. Instead the
# graph is rooted on the board already in front of us, which needs no action at all.
cpR = ClickPlanner()
cpR._enter_search([(1, 1), (2, 2)], from_reset=False)
check("re_entry_stages_no_opening_plan", cpR._plan, [])
check("re_entry_has_no_pending_root", cpR._pending, None)
check("re_entry_disables_teleport", cpR._root_is_reset, False)

_g11 = np.arange(64, dtype=np.int16).reshape(8, 8)
_first = cpR._search_step(_g11)
ok("re_entry_first_action_is_not_a_reset", _first != "RESET", f"emitted {_first!r}")
ok("re_entry_first_action_is_a_click",
   isinstance(_first, str) and _first.startswith("ACTION6"), f"emitted {_first!r}")
check("re_entry_roots_on_the_board_in_front_of_us",
      cpR._root, ClickPlanner._hash(_g11))
# The whole economic point: standing on the root, a probe is ONE action.
check("re_entry_probe_costs_one_action", len(cpR._plan), 1)

# No teleport may ever appear in a RESET-free episode, however deep the graph gets.
cpR2 = ClickPlanner()
cpR2._enter_search([(1, 1), (2, 2)], from_reset=False)
_emitted = []
for _i in range(60):
    _st = cpR2._search_step(np.full((8, 8), _i % 7, dtype=np.int16))
    if _st is None:
        break
    _emitted.append(_st)
    cpR2.note_action(_st)
ok("no_reset_anywhere_in_a_re_entered_episode", "RESET" not in _emitted,
   f"{len(_emitted)} actions emitted, resets={_emitted.count('RESET')}")

# The FIRST entry of a level keeps its RESET: that root is what makes a teleport
# plan mean anything, and coverage has guaranteed real clicks before it.
cpR3 = ClickPlanner()
cpR3._enter_search([(1, 1), (2, 2)], from_reset=True)
check("first_entry_still_roots_via_reset", cpR3._plan, ["RESET"])
check("first_entry_keeps_teleport", cpR3._root_is_reset, True)
# Defaulting matters: every pre-existing call site passes no flag and must be
# unchanged by this work.
cpR4 = ClickPlanner()
cpR4._enter_search([(1, 1), (2, 2)])
check("default_entry_is_the_reset_rooted_one", cpR4._plan, ["RESET"])

# And an unreachable frontier is NOT exhaustion: re-attach where we stand.
cpR5 = ClickPlanner()
cpR5._enter_search([(1, 1), (2, 2)], from_reset=False)
cpR5._cur = 12345
cpR5._adopt_root(12345)
cpR5._cur = 67890                      # navigation landed off the known component
ok("disconnected_board_is_adopted_not_declared_exhausted",
   cpR5._build_next_plan() and 67890 in cpR5._dist,
   f"dist={sorted(cpR5._dist)}")
# ...but adoption must stay OFF while teleport is live, or it would stage
# ["RESET", b] for a board RESET does not produce.
cpR6 = ClickPlanner()
cpR6._enter_search([(1, 1), (2, 2)], from_reset=True)
cpR6._adopt_root(999)
check("no_adoption_while_teleport_is_live", 999 in cpR6._dist, False)


print("\n=== 12. a PROOF and a BUDGET STOP are not the same claim (O1) ===")

# MEASURED, lp85 seed 0, the O1 diagnostic (2026-08-02). `_build_next_plan` gives up
# for three different reasons and the caller used to read all three as "this button
# set is refuted":
#   * the frontier emptied            -> a PROOF about the GAME
#   * len(_dist) > MAX_SEARCH_STATES  -> a statement about OUR MEMORY BUDGET
#   * every frontier node > MAX_REPLAY-> a statement about OUR ACTION BUDGET
# lp85 episode 1 reached 251 states against a cap of 250. The cap trip was filed as a
# refutation of a 3-button set, which -- via the subset rule, correct for real proofs --
# also banned the 2-button subset that had been working. A falsification claim that can
# be manufactured by running out of memory has no teeth, which is precisely what O1
# forbids. Proofs propagate to subsets; budget stops bind only the identical set.
cpP = ClickPlanner()
_bt = [(1, 1), (2, 2)]

# -- the reason is reported at all three exits --
cpP._enter_search(_bt, from_reset=False)
check("stop_reason_starts_clear", cpP._stop_reason, "")
cpP._queue = deque()
check("empty_frontier_returns_false", cpP._build_next_plan(), False)
check("empty_frontier_is_a_proof", cpP._stop_reason, "exhausted")

cpQ = ClickPlanner()
cpQ._enter_search(_bt, from_reset=False)
cpQ._dist = {i: () for i in range(cpQ.MAX_SEARCH_STATES + 2)}
check("state_cap_returns_false", cpQ._build_next_plan(), False)
check("state_cap_is_a_budget_stop", cpQ._stop_reason, "cap")

# Out of replay range: a frontier node exists with an untried button, but it is
# neither navigable from here nor within MAX_REPLAY of the root.
cpU = ClickPlanner()
cpU._enter_search(_bt, from_reset=True)
_far = 4242
cpU._dist = {_far: tuple([0] * (cpU.MAX_REPLAY + 5))}
cpU._G = {_far: {}}
cpU._queue = deque([_far])
cpU._cur = None                        # nothing on screen to navigate from
check("out_of_replay_range_returns_false", cpU._build_next_plan(), False)
check("out_of_replay_range_is_a_budget_stop", cpU._stop_reason, "unreachable")

# A successful plan must CLEAR the reason, or the next stop inherits a stale one.
cpV = ClickPlanner()
cpV._enter_search(_bt, from_reset=False)
cpV._stop_reason = "cap"
cpV._cur = 7
cpV._adopt_root(7)
ok("a_staged_plan_clears_the_reason",
   cpV._build_next_plan() and cpV._stop_reason == "", f"reason={cpV._stop_reason!r}")
# And a fresh episode never reads the previous episode's reason.
cpV._stop_reason = "cap"
cpV._enter_search(_bt, from_reset=False)
check("enter_search_clears_the_reason", cpV._stop_reason, "")

# -- the filing decision: which list does the set land in --
def _drive_to_stop(planner, buttons, rig):
    """Enter search, rig the stop condition, take one step, return the planner.

    from_reset=True deliberately: a re-entered (RESET-free) search adopts the board on
    screen as a frontier node, which REFILLS an emptied queue and is exactly right --
    an unreachable frontier is not exhaustion. Rigging an empty queue therefore only
    reaches the proof branch on a root-entered episode."""
    planner._enter_search(buttons, from_reset=True)
    planner._plan, planner._plan_i, planner._pending = [], 0, None
    rig(planner)
    planner._search_step(np.zeros((8, 8), dtype=np.int16))
    return planner

_pr = _drive_to_stop(ClickPlanner(), _bt, lambda p: setattr(p, "_queue", deque()))
check("a_proof_is_filed_as_refuted", _pr._exhausted_sets, [frozenset(_bt)])
check("a_proof_is_not_filed_as_suspended", _pr._suspended_sets, [])

def _rig_cap(p):
    p._dist = {i: () for i in range(p.MAX_SEARCH_STATES + 2)}
_cp = _drive_to_stop(ClickPlanner(), _bt, _rig_cap)
check("a_cap_trip_is_filed_as_suspended", _cp._suspended_sets, [frozenset(_bt)])
ok("a_cap_trip_is_NOT_a_refutation", _cp._exhausted_sets == [],
   f"exhausted={_cp._exhausted_sets}")
check("a_budget_stop_still_yields_to_coverage", _cp._phase, "discover")

# -- and the reason the distinction was worth making: how each propagates --
cpW = ClickPlanner()
# Coordinates deliberately >BUTTON_CLUSTER_DIST apart. Buttons are compared under the
# detector's clustering radius, so a "3-button set" like [(1,1),(2,2),(3,3)] is not a
# possible input at all -- _detect_buttons would merge those into one representative.
# A test about how SETS propagate needs coords that are genuinely different buttons.
_s3 = [(5, 5), (25, 25), (45, 45)]
cpW._suspended_sets.append(frozenset(_s3))
check("the_identical_suspended_set_is_dead", cpW._abstraction_is_live(_s3), False)
# THE ASYMMETRY. A subset spans a SMALLER space, so the cap that stopped the
# superset may not stop it at all. This is the lp85 case, and the old code killed it.
check("a_subset_of_a_SUSPENDED_set_is_still_live",
      cpW._abstraction_is_live([(5, 5), (25, 25)]), True)
check("a_superset_of_a_suspended_set_is_live",
      cpW._abstraction_is_live(_s3 + [(60, 60)]), True)
# Contrast the same shape against a real proof, which DOES propagate downward.
cpW._exhausted_sets.append(frozenset([(9, 40), (30, 9), (50, 2)]))
check("a_subset_of_a_REFUTED_set_is_dead",
      cpW._abstraction_is_live([(9, 40), (30, 9)]), False)
# Both lists are per-level: a new layout invalidates proofs and budget stops alike.
cpW.new_level()
check("new_level_clears_suspensions", cpW._suspended_sets, [])
check("new_level_clears_refutations_too", cpW._exhausted_sets, [])
ok("new_level_revives_a_suspended_set", cpW._abstraction_is_live(_s3), "")

# -- BUTTON IDENTITY: the same equivalence used to DETECT buttons must be used to
#    decide which button sets have already been tried.
# MEASURED, sb26 seed 0 (2026-08-02): eight search entries whose button sets sat 2, 2,
# 3, 4, 6 and 7 px from an already-refuted set -- all inside BUTTON_CLUSTER_DIST=10, the
# radius at which _detect_buttons merges reactive coords into ONE physical button. Exact-
# pixel set comparison let that jitter manufacture a "new abstraction" every time, and
# each one bought a fresh search episode on the same two buttons.
cpJ = ClickPlanner()
_ref = [(58, 22), (58, 42)]
cpJ._exhausted_sets.append(frozenset(_ref))
check("the_refuted_set_itself_is_dead", cpJ._abstraction_is_live(_ref), False)
# The real sb26 sequence, verbatim from the diagnostic. Every one of these is the same
# two buttons re-detected a few pixels off.
for _n, _jit in enumerate([[(58, 18), (58, 42)], [(57, 43), (59, 25)],
                           [(57, 41), (59, 25)], [(57, 41), (61, 25)]]):
    check(f"near_duplicate_of_a_refuted_set_is_dead_{_n}",
          cpJ._abstraction_is_live(_jit), False)
# ...but a button genuinely outside the radius is still a new abstraction. The fix must
# not be a blanket ban on anything that looks vaguely similar.
check("a_button_outside_the_radius_is_still_new",
      cpJ._abstraction_is_live([(58, 22), (58, 42), (20, 20)]), True)
check("near_duplicate_plus_a_far_button_is_live",
      cpJ._abstraction_is_live([(58, 18), (10, 10)]), True)
# Exactly at the radius is OUTSIDE it (strict <), matching _detect_buttons' own test.
check("exactly_at_the_cluster_radius_is_a_new_button",
      cpJ._abstraction_is_live([(58, 22 + ClickPlanner.BUTTON_CLUSTER_DIST)]), True)
check("one_inside_the_radius_is_not",
      cpJ._abstraction_is_live([(58, 22 + ClickPlanner.BUTTON_CLUSTER_DIST - 1)]), False)
# Suspensions need coverage BOTH ways, so a jittered near-duplicate is blocked but a
# strict superset is not.
cpK = ClickPlanner()
cpK._suspended_sets.append(frozenset([(58, 22), (58, 42)]))
check("a_jittered_suspended_set_is_dead",
      cpK._abstraction_is_live([(58, 18), (58, 42)]), False)
check("a_jittered_SUBSET_of_a_suspended_set_is_live",
      cpK._abstraction_is_live([(58, 18)]), True)


print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

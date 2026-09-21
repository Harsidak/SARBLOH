# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""test_execplanner.py -- COMPONENT 5.7 ExecPlanner (my_agent.py:3094).

The RHAE-critical component, and the reason stated once so every case reads
against it:

    SEARCH IS FREE, ACTIONS ARE NOT. RHAE squares the action ratio -- 10x human
    actions leaves ~1% of the level -- and internal reasoning costs nothing. So
    every action ExecPlanner spends must be one a VERIFIED model chose, and the
    search that chose it must have happened entirely inside the model.

That splits into four properties, and this file is those four:

  1. PLANNING IS UNSCORED AND OPTIMAL. `_bfs_path` reads the grid, mutates
     nothing, and returns a SHORTEST path. A search that returns a detour spends
     the difference in real actions, squared.
  2. ONLY A GREEN THEORY MAY COMMIT. `certify()` backtests the movement theory
     against the WHOLE level record. Red means at most one vetted step -- never a
     committed queue -- because a queue built on a falsified model spends N
     actions before reality gets a word in.
  3. A REFUTED MODEL RETIRES ITS PLAN. A vetoed move records a positional wall
     and clears the queue; a desynced avatar clears it too. Falsification must
     cost one action, not the length of the plan.
  4. GOAL INFERENCE IS THE ACTUAL WALL. The 25-game run at RHAE 0.0001 was not a
     planning failure: `_bfs_path` is fine and the toys are 3/3 + 2/2. It was
     `_nearest_target` picking the nearest distinctive colour on games whose goal
     is not that. Case 6 pins that failure in place rather than papering over it,
     and pins the two overrides that exist (decoy demotion, adopted goal hint).

Case 5 also pins the `np.False_` trap: `_explain` returns numpy comparisons, and
`np.False_ is False` is False, so an identity check against a raw numpy bool
silently never fires. `bool()` in `_explain` is load-bearing.

Run:  PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 python tests/test_execplanner.py
Never delete a case.
"""
import os
import sys

import numpy as np

os.environ.setdefault("ARC_NO_LLM", "1")

from my_agent import (                                         # noqa: E402
    ExecPlanner, StateEncoder, Timeline, base_action,
)

FAILS = []
WARNS = []
CHECKS = [0]


def check(name, got, want):
    CHECKS[0] += 1
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(f"{name}{': ' + detail if detail else ''}")
        print(f"  FAIL {name}  {detail}")
    else:
        print(f"  ok   {name}  {detail}" if detail else f"  ok   {name}")


def warn(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        WARNS.append(name)
        print(f"  WARN {name}  {detail}")
    else:
        print(f"  ok   {name}")


# ---------------------------------------------------------------- fixtures
BG, AV, GOAL, WALL, DECOY = 0, 3, 4, 5, 7
UP, DOWN, LEFT, RIGHT = "ACTION1", "ACTION2", "ACTION3", "ACTION4"
DISP = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
VALID = [UP, DOWN, LEFT, RIGHT]


def board(h=9, w=9, avatar=(4, 1), goal=None, walls=(), decoy=None):
    """A scale-1 world: one avatar pixel, plain background, explicit wall cells.

    Scale 1 keeps the lattice equal to the pixel grid, so an expected path can be
    written down by hand and a failure points at the planner rather than at the
    centroid arithmetic (which test_goalmodel covers).
    """
    g = np.full((h, w), BG, dtype=np.int32)
    for r, c in walls:
        g[r, c] = WALL
    if goal:
        g[goal] = GOAL
    if decoy:
        g[decoy] = DECOY
    g[avatar] = AV
    return g


def planner(obstacles=(WALL,)):
    p = ExecPlanner()
    p.avatar_color = AV
    p.action_disp = dict(DISP)
    p.obstacle_colors = set(obstacles)
    p.timeline = Timeline()
    p._cur_scale = 1
    return p


def Enc():
    """The real encoder. `update()` asks it for the background colour, and a stub
    that guesses wrong would quietly change which colours become obstacles --
    which is most of what this file is about."""
    return StateEncoder()


def entry(p, prev, action, nxt, level=0, reward=0.0):
    p.timeline.append(prev, action, nxt, reward, level)
    return p.timeline.entries[-1]


def moved(prev, frm, to):
    """The same board with the avatar pixel relocated -- one observed transition."""
    nxt = prev.copy()
    nxt[frm] = BG
    nxt[to] = AV
    return nxt


# ============================================================================
print("\n--- 1. planning is UNSCORED: the search reads, it never writes ------")
# ============================================================================
p = planner()
g = board(avatar=(4, 1), goal=(4, 7))
before = g.copy()
path = p._bfs_path(g, VALID, 4, 1, lambda r, c: (r, c) == (4, 7))
ok("bfs_finds_a_path", bool(path), f"{path}")
ok("bfs_does_not_mutate_the_grid", np.array_equal(g, before),
   "the grid is the world; a planner that edits it is playing a different game")
check("bfs_commits_nothing", len(p._plan), 0)
check("bfs_spends_no_visits", p._visits, {})
# Six cells apart, six actions. Anything longer is spent budget, squared.
check("bfs_path_is_shortest", len(path), 6)
ok("bfs_path_is_all_RIGHT", set(path) == {RIGHT}, f"{path}")

# Repeatability: the same question asked twice must give the same answer, or the
# TIE_EPS tie-break has leaked out of exploration and into goal pursuit.
p2 = planner()
check("bfs_is_deterministic", p2._bfs_path(g, VALID, 4, 1,
                                           lambda r, c: (r, c) == (4, 7)), path)

# A wall the model knows about must be routed AROUND, not through, and the detour
# must still be minimal.
def walk(path, r, c):
    """The cells a path visits, so 'routed around' can be checked, not assumed."""
    out = [(r, c)]
    for a in path or []:
        dr, dc = DISP[base_action(a)]
        r, c = r + dr, c + dc
        out.append((r, c))
    return out


walls = [(r, 4) for r in range(0, 9) if r != 0]     # a wall with a gap at row 0
gw = board(avatar=(4, 1), goal=(4, 7), walls=walls)
p3 = planner()
pw = p3._bfs_path(gw, VALID, 4, 1, lambda r, c: (r, c) == (4, 7))
ok("bfs_routes_around_a_wall",
   bool(pw) and all(gw[r, c] != WALL for r, c in walk(pw, 4, 1)), f"{pw}")
# Up 4, across the gap, down 4, and 6 columns across: 4 + 4 + 6 = 14.
ok("bfs_detour_is_minimal", pw is not None and len(pw) == 14,
   f"len {len(pw or [])} -- every extra step here is a real action, squared")

gb = board(avatar=(4, 1), goal=(4, 7), walls=[(r, 4) for r in range(9)])
check("bfs_returns_None_when_boxed_in",
      planner()._bfs_path(gb, VALID, 4, 1, lambda r, c: (r, c) == (4, 7)), None)

# ============================================================================
print("\n--- 2. commit records the model's PREDICTION, step by step ----------")
# ============================================================================
p = planner()
p._commit(4, 1, [RIGHT, RIGHT, DOWN])
check("commit_length", len(p._plan), 3)
check("commit_expects_the_start_cell", p._plan[0][0], (4, 1))
check("commit_walks_the_prediction_forward",
      [c for c, _ in p._plan], [(4, 1), (4, 2), (4, 3)])

# act() executes a committed step ONLY while reality still matches the prediction.
p._target = (4, 7)
a = p.act(board(avatar=(4, 1), goal=(4, 7)), VALID, Enc())
check("act_consumes_the_plan_when_in_sync", a, RIGHT)
check("act_tags_the_route", p._sub, "plan")

# The avatar is somewhere the plan did not predict -> the plan is about a world
# that no longer exists. The stale step must NOT be executed. act() then replans
# from where the avatar actually is, so the test is that the queue was rebuilt
# against reality -- asserting the queue is merely EMPTY would fail on a correct
# replan and pass on a planner that had stopped planning.
p2 = planner()
p2._commit(4, 1, [RIGHT, RIGHT, RIGHT])
stale = list(p2._plan)
a2 = p2.act(board(avatar=(0, 0), goal=(4, 7)), VALID, Enc())
ok("desync_does_not_execute_the_stale_step", list(p2._plan) != stale[1:],
   f"issued {a2}, queue {list(p2._plan)[:3]}")
ok("desync_replans_from_where_the_avatar_ACTUALLY_is",
   not p2._plan or p2._plan[0][0] == (0, 1) or p2._plan[0][0] == (1, 0),
   f"first expectation {p2._plan[0][0] if p2._plan else None} after acting "
   f"from (0,0)")

# ============================================================================
print("\n--- 3. certify(): the theory must reproduce the WHOLE record --------")
# ============================================================================
p = planner()
check("empty_timeline_is_green", p.certify(), True)

g0 = board(avatar=(4, 1))
entry(p, g0, RIGHT, moved(g0, (4, 1), (4, 2)))
check("a_transition_the_theory_predicts_is_green", p.certify(), True)

# A transition the theory cannot produce: RIGHT, and the avatar went LEFT.
p_bad = planner()
gb0 = board(avatar=(4, 4))
entry(p_bad, gb0, RIGHT, moved(gb0, (4, 4), (4, 3)))
check("a_counterexample_turns_the_theory_red", p_bad.certify(), False)
ok("the_counterexample_is_kept", p_bad._cert_fail is not None,
   "a red light with no counterexample cannot be experimented on")

# THE np.False_ TRAP. `_explain` compares numpy scalars; np.False_ == False is
# True but `np.False_ is False` is False, so callers doing an identity check
# against a raw numpy bool would silently never fire and every theory would read
# green forever. The bool() in _explain is what makes this pass.
e = p_bad.timeline.entries[-1]
v = p_bad._explain(e)
ok("explain_returns_a_real_python_bool", v is False or v is True or v is None,
   f"got {v!r} of type {type(v).__name__} -- np.False_ would fail this")
check("explain_type", type(v), bool)

# ARC_NO_CERT is the documented bisect switch: it makes every premise read green.
os.environ["ARC_NO_CERT"] = "1"
check("ARC_NO_CERT_forces_green", p_bad.certify(), True)
os.environ.pop("ARC_NO_CERT")
check("switch_removed_restores_red", p_bad.certify(), False)

# Out of scope is not the same as refuted: an action with no learned displacement
# says nothing about a displacement theory.
p_os = planner()
g1 = board(avatar=(4, 1))
e_os = entry(p_os, g1, "ACTION6", moved(g1, (4, 1), (4, 2)))
check("unknown_action_is_out_of_scope", p_os._explain(e_os), None)
check("out_of_scope_does_not_refute", p_os.certify(), True)

# ============================================================================
print("\n--- 4. RED means at most ONE vetted step, never a committed queue ---")
# ============================================================================
# This is property 2 and the single most expensive thing to get wrong: a queue of
# N actions issued through a falsified model spends N real actions before reality
# is consulted again, and RHAE squares that.
p = planner()
gr = board(avatar=(4, 4), goal=(4, 7))
entry(p, gr, RIGHT, moved(gr, (4, 4), (4, 3)))       # refutes the theory
check("setup_is_red", p.certify(), False)
p._exp = None
p._cert_fail = None                                   # skip the experiment branch
p._cert_green = False
a = p.act(gr, VALID, Enc())
ok("red_still_acts", a is not None, f"{a}")
check("red_commits_nothing", len(p._plan), 0)
ok("red_route_is_a_single_step", p._sub in ("goal1", "frontier1", "exper"),
   f"_sub={p._sub!r} -- a committed route tag here would mean the green gate "
   f"is decorative")

# Green, same board: the whole path is committed and the first step returned.
p_g = planner()
a = p_g.act(board(avatar=(4, 4), goal=(4, 7)), VALID, Enc())
check("green_route_is_goal", p_g._sub, "goal")
ok("green_commits_the_rest_of_the_path", len(p_g._plan) >= 1,
   f"queued {len(p_g._plan)} after issuing {a}")

# ============================================================================
print("\n--- 5. falsification costs ONE action: wall learned, plan retired ---")
# ============================================================================
p = planner(obstacles=())
g = board(avatar=(4, 1))
p._commit(4, 1, [RIGHT, RIGHT, RIGHT])
p.update(g, RIGHT, g.copy(), Enc())     # identical frame = the world vetoed it
check("veto_clears_the_plan", len(p._plan), 0)
ok("veto_records_a_positional_wall", (4, 2) in p.walls, f"walls={sorted(p.walls)}")
ok("the_wall_blocks_replanning",
   p._bfs_path(g, VALID, 4, 1, lambda r, c: (r, c) == (4, 2)) is None,
   "a learned wall the planner still routes through has not been learned")

# The experiment: a reproduced counterexample becomes an anomaly, and an anomaly
# is OUT OF SCOPE rather than a permanent red light -- otherwise one unexplained
# transition freezes the planner into single-stepping for the rest of the level.
p = planner()
gx = board(avatar=(4, 4))
entry(p, gx, RIGHT, moved(gx, (4, 4), (4, 3)))
check("experiment_starts_red", p.certify(), False)
first = p._experiment(gx, VALID, 4, 4)
check("experiment_re_runs_the_disputed_action", first, RIGHT)
check("experiment_counts_a_try", p._exp[1], p.EXP_TRIES - 1)
for _ in range(p.EXP_TRIES + 1):
    p._experiment(gx, VALID, 4, 4)
ok("reproduced_becomes_an_anomaly_or_a_revision",
   ((4, 3) in {c for c, _ in p._anomalies} or (4, 4) in {c for c, _ in p._anomalies}
    or p.walls or p._unwalled),
   f"anomalies={p._anomalies} walls={sorted(p.walls)} unwalled={p._unwalled}")

# _revise_representation, both directions. Theory said BLOCKED, avatar moved
# through -> the colour is walkable, un-mark it (Schema's 'valid landing cell').
p = planner()
gw2 = board(avatar=(4, 1), walls=[(4, 2)])
p.obstacle_colors = {WALL}
e = entry(p, gw2, RIGHT, moved(gw2, (4, 1), (4, 3)))
check("revision_returns_true_when_it_revises", p._revise_representation(e), True)
ok("walked_through_colour_is_unmarked", WALL not in p.obstacle_colors,
   f"obstacles={p.obstacle_colors}")
check("unmark_is_once_per_level", p._unwalled, {WALL})

# Theory said FREE, the avatar did not move -> an invisible barrier the colour
# layer cannot see. Record it positionally.
p = planner(obstacles=())
gf = board(avatar=(4, 1))
e = entry(p, gf, RIGHT, gf.copy())
check("revision_records_an_invisible_wall", p._revise_representation(e), True)
ok("invisible_wall_is_positional", (4, 2) in p.walls, f"{sorted(p.walls)}")

# ============================================================================
print("\n--- 6. goal inference: the documented wall, and its two overrides ---")
# ============================================================================
# THE FAILURE THE 25-GAME RUN FOUND. `_nearest_target` answers "which distinctive
# colour is nearest", and on ls20/tr87/tu93 the goal is not the nearest anything.
# This case does not assert that the heuristic is right -- it asserts what the
# heuristic DOES, so that a change to it shows up here as a diff instead of as a
# mystery on the bench three hours later.
p = planner()
gd = board(avatar=(4, 1), goal=(4, 8), decoy=(4, 3))
t = p._nearest_target(gd, 4, 1)
ok("nearest_target_picks_the_NEAREST_distinctive_colour", t == (4, 3),
   f"target {t} -- the decoy at (4,3), not the goal at (4,8). This is the "
   f"documented real-game failure, pinned deliberately: layered games' goals "
   f"are not 'the nearest odd colour'.")

# Override 1: a colour that demonstrably paid nothing is demoted out of candidacy.
p.non_goal_colors.add(DECOY)
check("a_known_decoy_is_skipped", p._nearest_target(gd, 4, 1), (4, 8))

# Override 2: an adopted goal hint outranks the heuristic entirely.
p2 = planner()
ok("goal_hint_is_adopted_when_in_bounds",
   p2.propose_theory({"goal": [4, 8]}, gd) and p2._llm_goal == (4, 8),
   f"{p2._llm_goal}")
a = p2.act(gd, VALID, Enc())
check("hint_redirects_pursuit_right_not_left", a, RIGHT)
check("hint_is_still_planned_through_the_model", p2._sub, "goal")

p3 = planner()
check("out_of_bounds_hint_is_rejected",
      p3.propose_theory({"goal": [99, 99]}, gd), False)
check("garbage_proposal_is_rejected", p3.propose_theory("not a dict", gd), False)

# A hint that is reached WITHOUT a level-up is self-discrediting: it was wrong,
# and holding it would keep pursuit parked on the same cell.
p4 = planner()
p4._llm_goal = (4, 1)
p4.act(board(avatar=(4, 1), goal=(4, 8)), VALID, Enc())
check("arriving_at_a_hint_that_paid_nothing_drops_it", p4._llm_goal, None)

# ============================================================================
print("\n--- 7. the backtest gate on theorizer proposals ---------------------")
# ============================================================================
# The Timeline is ground truth and the theory is provisional: a proposal that
# explains LESS of the record than the theory it replaces is reverted WHOLESALE
# -- decoys included, because an untrustworthy theory gets no partial credit.
# A STRIDE-3 world, deliberately: `_explain` allows +-1px of slack to absorb
# centroid rounding on multi-pixel sprites, so at stride 1 "moved" and "was
# blocked and stayed" are within tolerance of each other and no colour theory is
# falsifiable at all. Real games move 3-7px per action (the GridDSL stride
# finding), which is exactly the regime where the backtest has teeth.
STRIDE = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}


def stride_planner():
    q = planner(obstacles=())
    q.action_disp = dict(STRIDE)
    return q


p = stride_planner()
# The record: the avatar walked ONTO a cell of colour DECOY. Any theory calling
# DECOY an obstacle predicts the move was vetoed, and this one transition says
# otherwise.
gp = board(avatar=(4, 1), decoy=(4, 4))
entry(p, gp, RIGHT, moved(gp, (4, 1), (4, 4)))
check("baseline_backtest_is_clean", p._backtest_mismatches(), 0)

OTHER = 8
adopted = p.propose_theory({"obstacle_colors": [DECOY],
                            "decoy_colors": [OTHER]}, gp)
check("a_contradicted_proposal_is_not_adopted", adopted, False)
ok("contradicted_proposal_is_reverted_wholesale",
   DECOY not in p.obstacle_colors and OTHER not in p.non_goal_colors,
   f"obstacles={p.obstacle_colors} decoys={p.non_goal_colors} -- the decoy "
   f"hint rides on the same proposal and gets no partial credit")
check("backtest_is_clean_again", p._backtest_mismatches(), 0)

# The background can never be proposed into an obstacle, whatever the proposal
# says: `_obstacle_set` discards it. That guard is what stops the border-poisoning
# stall (background == letterbox border, both 0 -> every cell blocked -> BFS
# always fails), so it outranks any theory, LLM-authored or otherwise.
p_bg = stride_planner()
entry(p_bg, gp, RIGHT, moved(gp, (4, 1), (4, 4)))
p_bg.propose_theory({"obstacle_colors": [BG]}, gp)
check("background_never_becomes_a_wall_however_it_is_proposed",
      p_bg._backtest_mismatches(), 0)
ok("background_proposal_cannot_block_the_planner",
   p_bg._bfs_path(gp, VALID, 4, 1, lambda r, c: (r, c) == (4, 7)) is not None,
   "if this returns None the border-poisoning stall is back")

# A proposal the record does not contradict is adopted, and adopting it forces a
# fresh full replay -- a revised grounding must be re-verified against ALL the
# evidence, not just the transition that prompted it.
p2 = stride_planner()
entry(p2, gp, RIGHT, moved(gp, (4, 1), (4, 4)))
p2.certify()
p2._last_replay_at = 0
ok("an_uncontradicted_proposal_is_adopted",
   p2.propose_theory({"obstacle_colors": [WALL]}, gp), "")
ok("adoption_invalidates_the_certification_key", p2._cert_key is None,
   "a revised theory that keeps its old certification is certified against a "
   "theory that no longer exists")

# ============================================================================
print("\n--- 8. level and life boundaries carry the right things -------------")
# ============================================================================
# reset() = same layout after a death: the record and the anomalies are properties
# of the GAME, not of the attempt. new_level() = new layout: positions are stale,
# physics is not.
p = planner()
tl = p.timeline
p._anomalies.add(((1, 1), RIGHT))
p.walls.add((2, 2))
p._commit(4, 1, [RIGHT])
p.reset()
ok("reset_keeps_the_timeline", p.timeline is tl,
   "severing it was a real bug: the record is append-only ground truth")
ok("reset_keeps_anomalies", ((1, 1), RIGHT) in p._anomalies, f"{p._anomalies}")

p.walls.add((2, 2))
p.action_disp = dict(DISP)
p._commit(4, 1, [RIGHT, RIGHT])
p._llm_goal = (1, 1)
p.new_level()
check("new_level_drops_positional_walls", p.walls, set())
check("new_level_drops_the_plan", len(p._plan), 0)
check("new_level_drops_the_hint", p._llm_goal, None)
check("new_level_keeps_the_physics", p.action_disp, dict(DISP))
ok("new_level_keeps_the_timeline", p.timeline is tl,
   "the level tag keeps old evidence out of scope; deleting it would throw the "
   "evidence away instead")

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}   HARD FAILS: {len(FAILS)}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

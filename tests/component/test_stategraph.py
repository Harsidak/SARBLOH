# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 5.8: STATE GRAPH (+ DeadActionTracker's
interaction with the exploration frontier).

Why this component gets its own harness: the HUD mask learned in here feeds
EVERY masked comparison in the agent -- the rule verifier (hud_mask(precise=True)),
the livelock breaker and the learner (changed_masked), and the state hash itself.
An error here is not local; it silently corrupts the two components that were
locked before it. So the graph is pinned to its three questions and nothing else:

    have I seen this world-state before? / what have I not yet tried here? /
    what is the shortest known path to the nearest untried action?

Doctrine: the component STAYS in my_agent.py. This file only imports it and
feeds hand-built inputs whose correct answer is known by construction -- no game
engine, no LLM. Sections follow the architecture doc's harness contract:

    1. HUD invariance          5. RESET persistence vs level reset
    2. SACRED-CELL GUARD       6. ACTION6 exclusion
    3. exactness / order       7. budget bounds
    4. frontier correctness    8. orchestrator wiring (note_outcome is called)

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_stategraph.py
"""
import os
import sys

import numpy as np

from my_agent import (StateGraph, DeadActionTracker, MyAgent,
                      GameState, base_action)

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
        print(f"  FAIL  {name}: {detail}")
        FAILS.append(name)


def warn(name, cond, detail):
    tag = "ok  " if cond else "WARN"
    if not cond:
        WARNS.append(name)
    print(f"  {tag}  {name}: {detail}")


# ======================================================================
print("== 1. HUD INVARIANCE: a ticking clock is not a new world-state ==")
# ======================================================================
ACTS2 = ["ACTION1", "ACTION2"]
ACTS3 = ["ACTION1", "ACTION2", "ACTION3"]
N_HUD = 40                      # > MASK_CHECK_EVERY, so the mask has formed


def hud_frame(world, t):
    """6x6 world with ONE avatar cell and ONE timer cell that ticks every step."""
    g = np.zeros((6, 6), dtype=np.int32)
    g[0, 0] = world
    g[5, 5] = 1 + (t % 9)
    return g


sg1 = StateGraph()
hf = [hud_frame(0, t) for t in range(N_HUD + 1)]
sg1.ensure_state(hf[0], ACTS3)
for t in range(N_HUD):
    sg1.record(hf[t], ACTS2[t % 2], hf[t + 1])
    sg1.ensure_state(hf[t + 1], ACTS3)

ok("hud_cell_masked", sg1._mask is not None and bool(sg1._mask[5, 5]),
   f"mask at the timer = {None if sg1._mask is None else sg1._mask[5, 5]}")
ok("world_cell_not_masked", sg1._mask is not None and not sg1._mask[0, 0],
   "the avatar cell must stay in the identity hash")
check("timer_only_frames_hash_equal",
      sg1._hash(hud_frame(0, 3)) == sg1._hash(hud_frame(0, 8)), True)
ok("world_change_hashes_differ",
   sg1._hash(hud_frame(0, 3)) != sg1._hash(hud_frame(7, 3)),
   "a real world change must still make a NEW state")
check("untested_does_not_explode", len(sg1.untested), 1)
check("hud_only_change_is_not_a_world_change",
      sg1.changed_masked(hf[0], hf[1]), False)
_w = hf[0].copy(); _w[0, 0] = 9
check("world_change_is_a_world_change", sg1.changed_masked(hf[0], _w), True)
# The collapsed state has A1/A2 on record as self-loops; A3 is the frontier.
check("collapsed_state_still_knows_its_frontier",
      sg1.nearest_untested_grid(hf[-1]), ([], "ACTION3"))

# A DRAINING BAR is the same hazard spread over a row: no single cell crosses
# MASK_RATE, but the row changes on every transition -> the strip rule catches it.
sgbar = StateGraph()


def bar_frame(world, t):
    g = np.zeros((6, 40), dtype=np.int32)
    g[0, 0] = world
    g[5, :40 - t] = 8
    return g


bf = [bar_frame(0, t) for t in range(N_HUD + 1)]
sgbar.ensure_state(bf[0], ACTS2)
for t in range(N_HUD):
    sgbar.record(bf[t], ACTS2[t % 2], bf[t + 1])
    sgbar.ensure_state(bf[t + 1], ACTS2)
ok("draining_bar_row_masked",
   sgbar._mask is not None and bool(sgbar._mask[5].all())
   and not sgbar._mask[0, 0],
   "the whole bar row is masked; the world row is not")
check("bar_frames_collapse_to_one_state", len(sgbar.untested), 1)
# The strips are the BROAD mask only -- the precise (per-cell) mask a rule
# verifier reads must NOT excuse a cell just for sharing a row with the bar.
ok("precise_mask_excludes_strips",
   sgbar.hud_mask(precise=True) is None
   or not bool(sgbar.hud_mask(precise=True)[5, 0]),
   "hud_mask(precise=True) is per-cell only")

# ======================================================================
print("\n== 2. SACRED CELLS: a score-relevant cell may NEVER be masked ==")
# ======================================================================
# The one unrecoverable error in this component. A world cell that flickers in
# MORE than MASK_RATE of transitions -- a blinking goal, a pulsing hazard -- is
# eaten by the per-cell rule and becomes invisible to the state hash: the agent
# then cannot tell a winning frame from a losing one. note_outcome() feeds the
# graph the engine's reward/terminal signal so those cells are pinned out.
N_SAC = 40


def sac_frame(t, goal_phase, world=0):
    """(0,0) world, (2,2) GOAL flickering in 60% of steps, (5,5) HUD timer."""
    g = np.zeros((6, 6), dtype=np.int32)
    g[0, 0] = world
    g[2, 2] = 4 + goal_phase
    g[5, 5] = 1 + (t % 9)
    return g


def sac_series():
    """Frames + the transition indices where the GOAL cell changed."""
    frames, changed, phase = [], [], 0
    for t in range(N_SAC + 1):
        frames.append(sac_frame(t, phase))
        if t % 5 < 3:               # 60% of steps: above MASK_RATE, so maskable
            changed.append(t)
            phase = 1 - phase
    return frames, changed


sf, goal_steps = sac_series()

# --- baseline: with NO outcome evidence the flickering goal IS masked --------
sgA = StateGraph()
sgA.ensure_state(sf[0], ACTS2)
for t in range(N_SAC):
    sgA.record(sf[t], ACTS2[t % 2], sf[t + 1])
    sgA.ensure_state(sf[t + 1], ACTS2)
ok("baseline_flickering_goal_is_masked",
   sgA._mask is not None and bool(sgA._mask[2, 2]),
   "without outcome evidence a 60%-flicker cell reads as HUD -- this is the bug")
ok("baseline_timer_masked", sgA._mask is not None and bool(sgA._mask[5, 5]), "")

# --- guarded: the same history, plus 4 deaths on frames where the goal moved --
sgB = StateGraph()
sgB.ensure_state(sf[0], ACTS2)
for t in range(N_SAC):
    sgB.record(sf[t], ACTS2[t % 2], sf[t + 1])
    sgB.ensure_state(sf[t + 1], ACTS2)
for t in goal_steps[:4]:
    sgB.note_outcome(terminal=True, prev=sf[t], nxt=sf[t + 1])

ok("sacred_goal_survives_the_mask",
   sgB._mask is not None and not bool(sgB._mask[2, 2]),
   "THE gate: a cell that moves whenever the run ends stays in the hash")
ok("sacred_flag_is_reported",
   sgB.sacred_cells() is not None and bool(sgB.sacred_cells()[2, 2]),
   "sacred_cells() names the protected cell")
ok("sacred_goal_also_out_of_the_precise_mask",
   sgB.hud_mask(precise=True) is None
   or not bool(sgB.hud_mask(precise=True)[2, 2]),
   "the rule verifier must not be told to ignore it either")
# ...and the guard did NOT cost us the mask. The timer changed on every one of
# those death frames too; protecting on bare coincidence would have pinned it.
ok("clock_not_pinned_by_coincidence",
   sgB._mask is not None and bool(sgB._mask[5, 5]),
   "the timer ticks at outcomes AND everywhere else -> margin 0 -> still masked")
ok("sacred_does_not_name_the_clock",
   sgB.sacred_cells() is not None and not bool(sgB.sacred_cells()[5, 5]), "")
# The protection has to be visible where it matters: identity.
_g1 = sac_frame(3, 0)
_g2 = sac_frame(3, 1)               # differs ONLY in the goal cell's phase
ok("goal_phase_makes_a_distinct_state", sgB._hash(_g1) != sgB._hash(_g2),
   "guarded graph tells the two phases apart")
ok("baseline_conflates_the_two_phases", sgA._hash(_g1) == sgA._hash(_g2),
   "unguarded graph cannot -- exactly the blindness the guard removes")

# --- a sacred cell INSIDE a masked HUD strip must be carved back out --------
N_STRIP = 40


def strip_frame(t, goal_phase):
    """Row 5 is a full-width HUD bar; (5,2) is a goal cell sitting IN it."""
    g = np.zeros((8, 8), dtype=np.int32)
    g[5, :] = 1 + (t % 9)                   # whole row ticks every step
    g[5, 2] = 4 + goal_phase
    return g


sgC = StateGraph()
phase, strip_steps, stf = 0, [], []
for t in range(N_STRIP + 1):
    stf.append(strip_frame(t, phase))
    if t % 5 < 3:
        strip_steps.append(t)
        phase = 1 - phase
sgC.ensure_state(stf[0], ACTS2)
for t in range(N_STRIP):
    sgC.record(stf[t], ACTS2[t % 2], stf[t + 1])
    sgC.ensure_state(stf[t + 1], ACTS2)
_row_masked_before = sgC._mask is not None and bool(sgC._mask[5, 2])
for t in strip_steps[:4]:
    sgC.note_outcome(reward=1.0, prev=stf[t], nxt=stf[t + 1])
ok("strip_swallowed_the_goal_before_the_guard", _row_masked_before,
   "the periodic-row rule paints the whole line, goal included")
ok("sacred_carved_out_of_a_masked_strip",
   sgC._mask is not None and not bool(sgC._mask[5, 2])
   and bool(sgC._mask[5, 0]) and bool(sgC._mask[5, 7]),
   "the bar stays masked AROUND the protected cell")

# --- the guard must not fire on nothing --------------------------------------
sgD = StateGraph()
sgD.ensure_state(sf[0], ACTS2)
for t in range(N_SAC):
    sgD.record(sf[t], ACTS2[t % 2], sf[t + 1])
check("no_outcomes_no_sacred", sgD.sacred_cells(), None)
sgD.note_outcome(reward=0.0, terminal=False, prev=sf[0], nxt=sf[1])
check("a_non_outcome_is_ignored", sgD._out_n, 0)
sgD.note_outcome(terminal=True, prev=sf[0], nxt=np.zeros((3, 3), dtype=np.int32))
check("shape_mismatch_is_ignored", sgD._out_n, 0)
sgE = StateGraph()
sgE.note_outcome(terminal=True)     # no transitions at all -> must not raise
check("note_outcome_on_an_empty_graph_is_safe", sgE.sacred_cells(), None)
sgF = StateGraph()
sgF.ensure_state(sf[0], ACTS2)
for t in range(N_SAC):
    sgF.record(sf[t], ACTS2[t % 2], sf[t + 1])
sgF.note_outcome(terminal=True)     # defaults to the LAST recorded transition
ok("note_outcome_defaults_to_the_last_transition", sgF._out_n == 1,
   "the outcome transition is the one that just happened")
# A cell that never moves at outcomes is never protected, however often it moves.
ok("quiet_cell_never_sacred",
   sgB.sacred_cells() is not None and not bool(sgB.sacred_cells()[0, 0]), "")

# --- KNOWN BLIND SPOT (pinned, not fixed) ------------------------------------
# A world cell that ticks on LITERALLY every transition is statistically
# identical to a clock: both change at every outcome and at every ordinary step,
# so the disproportionality margin is 0 for both. No signal available to this
# component separates them. Recorded so the limit is visible, not discovered.
N_ALW = 40
alw = []
for t in range(N_ALW + 1):
    g = np.zeros((6, 6), dtype=np.int32)
    g[2, 2] = 4 + (t % 2)           # "goal" blinks EVERY step
    g[5, 5] = 1 + (t % 9)
    alw.append(g)
sgG = StateGraph()
sgG.ensure_state(alw[0], ACTS2)
for t in range(N_ALW):
    sgG.record(alw[t], ACTS2[t % 2], alw[t + 1])
    sgG.ensure_state(alw[t + 1], ACTS2)
for t in range(4):
    sgG.note_outcome(terminal=True, prev=alw[t], nxt=alw[t + 1])
warn("every_step_blinker_reads_as_hud",
     sgG._mask is not None and not bool(sgG._mask[2, 2]),
     "a cell that changes at EVERY transition is indistinguishable from a clock "
     "by any statistic in this component; the guard needs a rate difference")

# ======================================================================
print("\n== 3. EXACTNESS: order-independence and rebuild-from-raw-logs ==")
# ======================================================================
gA = np.zeros((4, 4), dtype=np.int32)
gB = gA.copy(); gB[0, 0] = 1
gC = gA.copy(); gC[1, 1] = 2
EDGES = [(gA, "ACTION1", gB), (gB, "ACTION2", gC),
         (gA, "ACTION3", gC), (gC, "ACTION1", gA)]

sg_f, sg_r = StateGraph(), StateGraph()
for g in (gA, gB, gC):
    sg_f.ensure_state(g, ACTS3)
    sg_r.ensure_state(g, ACTS3)
for p, a, n in EDGES:
    sg_f.record(p, a, n)
for p, a, n in reversed(EDGES):
    sg_r.record(p, a, n)
check("edge_order_does_not_change_adj", sg_f.adj == sg_r.adj, True)
check("edge_order_does_not_change_frontier", sg_f.untested == sg_r.untested, True)
check("a_repeated_edge_is_idempotent",
      (sg_f.record(gA, "ACTION1", gB), sg_f.adj == sg_r.adj)[1], True)

# Rebuild exactness: a graph that learned its mask LATE and replayed its raw
# logs must end up bit-for-bit identical to one that knew the mask all along.
sg_late = StateGraph()
sg_late.ensure_state(hf[0], ACTS3)
for t in range(N_HUD):
    sg_late.record(hf[t], ACTS2[t % 2], hf[t + 1])
    sg_late.ensure_state(hf[t + 1], ACTS3)
sg_pre = StateGraph()
sg_pre.MASK_CHECK_EVERY = 10 ** 9            # never learns; is simply TOLD
sg_pre._mask = sg_late._mask.copy()
sg_pre.ensure_state(hf[0], ACTS3)
for t in range(N_HUD):
    sg_pre.record(hf[t], ACTS2[t % 2], hf[t + 1])
    sg_pre.ensure_state(hf[t + 1], ACTS3)
check("rebuild_reproduces_adj", sg_late.adj == sg_pre.adj, True)
check("rebuild_reproduces_frontier", sg_late.untested == sg_pre.untested, True)
check("rebuild_reproduces_state_set",
      set(sg_late.grids) == set(sg_pre.grids), True)
# ...and a rebuild drops the imagination-skip set, whose keys were hashes under
# the OLD mask and would otherwise silently suppress real frontier actions.
sg_late._noop_skip.add((12345, "ACTION1"))
sg_late._mask = None                          # force the next update to differ
sg_late._update_mask()
check("rebuild_clears_stale_noop_skips", sg_late._noop_skip, set())

# ======================================================================
print("\n== 4. FRONTIER: shortest known path to the nearest untried action ==")
# ======================================================================
S = np.zeros((5, 5), dtype=np.int32)
M = S.copy(); M[0, 0] = 1
T = S.copy(); T[0, 1] = 2
sg4 = StateGraph()
sg4.ensure_state(S, ACTS3)
sg4.record(S, "ACTION1", M)
sg4.record(S, "ACTION2", T)             # 1 hop to T
sg4.record(S, "ACTION3", S)
sg4.ensure_state(M, ACTS3)
sg4.record(M, "ACTION1", T)             # 2 hops to T the other way
sg4.record(M, "ACTION2", M)
sg4.record(M, "ACTION3", M)
sg4.ensure_state(T, ACTS3)
res4 = sg4.nearest_untested_grid(S)
ok("shortest_path_to_the_frontier",
   res4 is not None and res4[0] == ["ACTION2"] and res4[1] in ACTS3,
   f"got {res4}")
ok("own_untested_beats_any_walk",
   sg4.nearest_untested_grid(T)[0] == [], "path from T itself is empty")

sg4b = StateGraph()
sg4b.ensure_state(S, ACTS2)
sg4b.record(S, "ACTION1", S)
sg4b.record(S, "ACTION2", S)
check("none_only_at_true_exhaustion", sg4b.nearest_untested_grid(S), None)

# A dead action is not a frontier: it is proven inert from EVERY state.
sg4c = StateGraph()
sg4c.ensure_state(S, ACTS2)
dt = DeadActionTracker()
for _ in range(DeadActionTracker.MIN_TRIALS):
    dt.update("ACTION1", changed=False)
ok("dead_actions_are_not_offered",
   sg4c.nearest_untested_grid(S, dt) == ([], "ACTION2"),
   "A1 is dead everywhere, so only A2 is worth a scored action")

# Imagination-pruned pairs: the world model says both are no-ops -> the frontier
# is empty WITH the predicate, and the caller's noop-less retry re-offers them.
sg4d = StateGraph()
sg4d.ensure_state(S, ACTS2)
check("noop_predicate_empties_the_frontier",
      sg4d.nearest_untested_grid(S, None, noop=lambda g, a: True), None)
check("noop_skips_are_remembered", len(sg4d._noop_skip), 2)
ok("noop_skipped_pairs_are_re_offered_without_the_predicate",
   sg4d.nearest_untested_grid(S) is not None,
   "the model may be wrong -- exhaustion is only exhaustion unassisted")
sg4e = StateGraph()
sg4e.ensure_state(S, ACTS2)
res4e = sg4e.nearest_untested_grid(S, None, noop=lambda g, a: a == "ACTION1")
ok("noop_predicate_prunes_only_what_it_claims",
   res4e == ([], "ACTION2"), f"got {res4e}")

# ======================================================================
print("\n== 5. LIFECYCLE: survives intra-level RESET, dies at a level boundary ==")
# ======================================================================
sg5 = StateGraph()
sg5.ensure_state(S, ACTS2)
sg5.record(S, "ACTION1", M)
sg5.ensure_state(M, ACTS2)
_adj_before = {k: dict(v) for k, v in sg5.adj.items()}
# An intra-level RESET returns the SAME layout: nothing about the graph changes,
# and re-entering a known state must not re-offer an action already on record.
sg5.ensure_state(S, ACTS2)
check("reset_does_not_disturb_known_edges", sg5.adj, _adj_before)
ok("reset_does_not_re_offer_a_tested_action",
   "ACTION1" not in sg5.untested[sg5._hash(S)],
   "the S--A1-->M edge is still true after a restart")
ok("no_intra_level_wipe_exists",
   not any(hasattr(sg5, m) for m in ("reset", "new_level", "clear")),
   "the graph has no method to forget within a level -- a level-up makes a NEW one")
check("a_new_level_starts_empty", (len(StateGraph().adj),
                                   len(StateGraph().untested),
                                   len(StateGraph().grids)), (0, 0, 0))

# ======================================================================
print("\n== 6. ACTION6 EXCLUSION: clicks never enter adjacency ==")
# ======================================================================
sg6 = StateGraph()
sg6.ensure_state(S, ["ACTION1", "ACTION6"])
check("click_not_in_the_frontier", sg6.untested[sg6._hash(S)], {"ACTION1"})
_adj6 = {k: dict(v) for k, v in sg6.adj.items()}
sg6.record(S, "ACTION6", M)
sg6.record(S, "ACTION6_r3_c4", M)
check("click_adds_no_edge", sg6.adj, _adj6)
check("click_adds_no_raw_transition", len(sg6._trans), 0)
check("click_feeds_no_mask_statistics", sg6._change, None)
check("parameterised_click_is_still_a_click", base_action("ACTION6_r3_c4"),
      "ACTION6")
sg6.ensure_state(S, ["ACTION6_r0_c0", "ACTION2"])
check("parameterised_click_not_in_the_frontier",
      sg6.untested[sg6._hash(S)], {"ACTION1", "ACTION2"})

# ======================================================================
print("\n== 7. BUDGET BOUNDS: capped logs and a capped search ==")
# ======================================================================
sg7 = StateGraph()
sg7.MAX_LOG = 10
sg7.MASK_CHECK_EVERY = 10 ** 9      # masking is section 1's subject, not this one
for i in range(30):
    a = np.zeros((4, 4), dtype=np.int32); a[0, 0] = i + 1
    b = a.copy(); b[1, 1] = i + 1
    sg7.ensure_state(a, ACTS2)
    sg7.record(a, "ACTION1", b)
ok("raw_logs_respect_MAX_LOG",
   len(sg7._ensures) <= 10 and len(sg7._trans) <= 10,
   f"ensures={len(sg7._ensures)} trans={len(sg7._trans)} (cap 10)")
ok("graph_keeps_working_past_the_log_cap", len(sg7.adj) > 10,
   f"{len(sg7.adj)} states -- the LIVE graph is not capped, only the replay log")


def chain(n):
    """n PAIRWISE-DISTINCT 20x20 frames (two odometer digits), so the chain is a
    genuine n-state line and the search really has to walk it."""
    out = []
    for i in range(n):
        g = np.zeros((20, 20), dtype=np.int32)
        g[0, i % 20] = 3
        g[1, (i // 20) % 20] = 4
        out.append(g)
    return out


CH = chain(120)
sg7b = StateGraph()
sg7b.MAX_BFS_STATES = 20
sg7b.MASK_CHECK_EVERY = 10 ** 9     # ditto -- keep every frame a separate state
for i in range(len(CH) - 1):
    sg7b.ensure_state(CH[i], ["ACTION1"])
    sg7b.record(CH[i], "ACTION1", CH[i + 1])
sg7b.ensure_state(CH[-1], ["ACTION1"])          # only the far end has a frontier
ok("chain_did_not_collapse", len(sg7b.untested) > 100,
   f"{len(sg7b.untested)} distinct states -- the mask stayed off")
check("search_gives_up_gracefully_at_MAX_BFS_STATES",
      sg7b.nearest_untested_grid(CH[0]), None)
sg7c = StateGraph()
sg7c.MASK_CHECK_EVERY = 10 ** 9
for i in range(len(CH) - 1):
    sg7c.ensure_state(CH[i], ["ACTION1"])
    sg7c.record(CH[i], "ACTION1", CH[i + 1])
sg7c.ensure_state(CH[-1], ["ACTION1"])
res7 = sg7c.nearest_untested_grid(CH[0])
ok("same_search_succeeds_under_the_default_cap",
   res7 is not None and len(res7[0]) == len(CH) - 1,
   f"path length {None if res7 is None else len(res7[0])} (want {len(CH) - 1})")

# ======================================================================
print("\n== 8. WIRING: the orchestrator actually reports outcomes ==")
# ======================================================================
# The guard is worthless if nobody feeds it. These drive the REAL choose_action
# and assert note_outcome is called on the two signals the engine gives us.
os.environ["ARC_AGENT_SEED"] = "0"


class Frame:
    def __init__(self, g, actions, lvl=0, st=None):
        self.frame = g.tolist()
        self.available_actions = list(actions)
        self.levels_completed = lvl
        self.state = st if st is not None else GameState.NOT_FINISHED


def spy(agent):
    """Record note_outcome calls without changing what the graph does."""
    calls = []
    real = agent.sgraph.note_outcome

    def wrapped(reward=0.0, terminal=False, prev=None, nxt=None):
        calls.append((float(reward), bool(terminal),
                      prev is not None and nxt is not None))
        return real(reward=reward, terminal=terminal, prev=prev, nxt=nxt)
    agent.sgraph.note_outcome = wrapped
    return calls


w0 = np.zeros((12, 12), dtype=np.int32); w0[3, 3] = 2
w1 = w0.copy(); w1[3, 4] = 2

ag = MyAgent(game_id="stategraph-wiring-death")
calls = spy(ag)
f0 = Frame(w0, [1, 2, 3])
ag.choose_action([f0], f0)                      # first step: nothing to report yet
check("no_outcome_on_an_ordinary_step", calls, [])
fdead = Frame(w1, [1, 2, 3], st=GameState.GAME_OVER)
ag.choose_action([fdead], fdead)
ok("game_over_is_reported",
   len(calls) == 1 and calls[0][1] is True and calls[0][2] is True,
   f"calls={calls}")

ag2 = MyAgent(game_id="stategraph-wiring-score")
calls2 = spy(ag2)
fa = Frame(w0, [1, 2, 3], lvl=0)
ag2.choose_action([fa], fa)
fb = Frame(w1, [1, 2, 3], lvl=1)                # the score moved on this frame
ag2.choose_action([fb], fb)
ok("score_change_is_reported",
   len(calls2) == 1 and calls2[0][0] == 1.0 and calls2[0][2] is True,
   f"calls={calls2}")
ok("level_up_still_rebuilds_the_graph", ag2.sgraph.note_outcome is not calls2
   and getattr(ag2.sgraph, "_out_n", 0) == 0,
   "the outcome was reported to the OLD graph; the new level starts clean")

# A score REGRESSION is where this actually pays: the graph is NOT replaced, so
# the cells the engine just punished stay protected for the rest of the level.
ag3 = MyAgent(game_id="stategraph-wiring-regression")
fc = Frame(w0, [1, 2, 3], lvl=2)
ag3.choose_action([fc], fc)
_g_before = ag3.sgraph
calls3 = spy(ag3)
fd = Frame(w1, [1, 2, 3], lvl=1)                # score DROPPED
ag3.choose_action([fd], fd)
ok("score_regression_is_reported", len(calls3) == 1 and calls3[0][0] == 1.0,
   f"calls={calls3}")
ok("score_regression_keeps_the_graph", ag3.sgraph is _g_before,
   "no level-up -> the mask and its protections survive")
ok("regression_evidence_is_retained", ag3.sgraph._out_n == 1,
   f"_out_n={ag3.sgraph._out_n}")

print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

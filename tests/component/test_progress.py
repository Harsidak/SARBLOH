# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Component isolation tests for ProgressModel (COMPONENT 5.55, objective O3).

The component answers "which board is closer to winning?", learned from the trajectory
of a level the agent actually completed. It is the piece that was missing while every
other objective was satisfied: ClickPlanner's O3 was "an untried (state, button) pair is
worth one unit", which is a COVERAGE objective -- every untried pair scores the same, so
lp85's 251 explored level-2 boards were all equally attractive.

Everything here is hand-built input, no game engine. Cases are never removed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARC_AGENT_SEED", "0")

from my_agent import ProgressModel, ClickPlanner, StateEncoder      # noqa: E402

FAILS, WARNS = [], []


def check(name, got, want):
    if isinstance(got, np.bool_):
        got = bool(got)
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


def consolidating(n, size=16):
    """A trajectory that genuinely IS progress: a scattered board resolving into one
    block. Colour boundaries fall monotonically, which is what a puzzle looks like when
    it is being solved."""
    out = []
    for t in range(n):
        g = np.zeros((size, size), dtype=np.int16)
        # start: every other cell set (maximum boundary). end: a solid left block.
        if t == 0:
            g[::2, ::2] = 1
        else:
            g[:, : max(1, int(size * t / (n - 1)))] = 1
        out.append(g)
    return out


print("=== 1. features are total, bounded and shape-agnostic ===")

F = ProgressModel.features
check("feature_vector_has_the_declared_width", F(np.zeros((8, 8), np.int16)).shape,
      (ProgressModel.N_FEAT,))
_f = F(np.random.RandomState(0).randint(0, 10, (64, 64)).astype(np.int16))
ok("every_feature_is_a_fraction", bool(np.all(_f >= -1e-9) and np.all(_f <= 1 + 1e-9)),
   f"min={_f.min():.3f} max={_f.max():.3f}")
# TOTAL: garbage in must not raise. A crash inside a scoring function would take down
# the whole planner for an objective that is only ever advisory.
for _name, _bad in [("empty", np.zeros((0, 0), np.int16)),
                    ("1d", np.zeros(5, np.int16)),
                    ("3d", np.zeros((2, 2, 2), np.int16)),
                    ("none", None),
                    ("float", np.zeros((4, 4), np.float64)),
                    ("negative", -np.ones((4, 4), np.int16)),
                    ("huge_colour", np.full((4, 4), 9999, np.int16))]:
    try:
        _r = F(_bad)
        ok(f"features_total_on_{_name}", _r.shape == (ProgressModel.N_FEAT,), f"{_r.shape}")
    except Exception as e:                                   # noqa: BLE001
        ok(f"features_total_on_{_name}", False, f"raised {e!r}")
# Identical boards must describe identically, different boards must not.
_a, _b = np.zeros((8, 8), np.int16), np.zeros((8, 8), np.int16)
_b[0, 0] = 3
check("same_board_same_features", bool(np.array_equal(F(_a), F(_a))), True)
ok("different_boards_differ", not np.array_equal(F(_a), F(_b)),
   f"delta={np.abs(F(_a) - F(_b)).sum():.4f}")
# Consolidation must actually track consolidation: a chequerboard has the most
# colour boundaries a board can have, a solid block has none.
_chk = np.indices((16, 16)).sum(axis=0) % 2
_solid = np.zeros((16, 16), np.int16)
_k = ProgressModel.N_COLORS
ok("boundary_feature_is_maximal_on_a_chequerboard",
   F(_chk.astype(np.int16))[_k] > F(_solid)[_k],
   f"chequer={F(_chk.astype(np.int16))[_k]:.3f} solid={F(_solid)[_k]:.3f}")


print("\n=== 2. an unfitted model is a strict no-op (O4: never a gate) ===")

pm = ProgressModel()
check("unfitted_is_not_ready", pm.ready, False)
check("unfitted_scores_zero", pm.score(F(_chk.astype(np.int16))), 0.0)
check("unfitted_buckets_zero", pm.bucket(F(_chk.astype(np.int16))), 0)
# Every board scoring the same is exactly what makes it a no-op: the frontier ordering
# falls back to O2 (cost), which is the previously-measured behaviour.
_bs = {pm.bucket(F(g)) for g in consolidating(6)}
check("unfitted_ranks_nothing", _bs, {0})
check("score_survives_none", pm.score(None), 0.0)
check("score_survives_a_wrong_width", pm.score(np.zeros(3)), 0.0)


print("\n=== 3. fitting on a winning trajectory (the GROUNDED channel) ===")

pm2 = ProgressModel()
traj = [F(g) for g in consolidating(10)]
check("fit_accepts_a_real_trajectory", pm2.fit(traj), True)
check("fitted_is_ready", pm2.ready, True)
# The point of fitting: the model must now RANK the trajectory it learned from.
_s = [pm2.score(t) for t in traj]
ok("score_rises_along_the_learned_trajectory", _s[-1] > _s[0],
   f"{_s[0]:.4f} -> {_s[-1]:.4f}")
ok("learned_trajectory_is_mostly_monotone", pm2.backtest(traj) >= 0.7,
   f"monotone fraction {pm2.backtest(traj):.2f}")
# It must also discriminate BETWEEN boards, not just along one path -- ordering the
# frontier is the entire use. The solved board is this trajectory's OWN end state:
# `_solid` (all zeros) is not it. consolidating() resolves toward colour 1, so the model
# correctly learns "more colour 1", and an all-ZERO board is start-like, not solved.
# Comparing against it would have been testing the fixture, not the model.
_good_g, _bad_g = consolidating(10)[-1], consolidating(10)[0]
ok("a_solved_board_outranks_a_scattered_one",
   pm2.score(F(_good_g)) > pm2.score(F(_bad_g)),
   f"solved={pm2.score(F(_good_g)):.4f} scattered={pm2.score(F(_bad_g)):.4f}")
ok("an_all_background_board_is_not_mistaken_for_solved",
   pm2.score(F(_solid)) < pm2.score(F(_good_g)),
   f"blank={pm2.score(F(_solid)):.4f} solved={pm2.score(F(_good_g)):.4f}")
ok("buckets_are_coarser_than_scores",
   len({pm2.bucket(t) for t in traj}) <= len({round(x, 9) for x in _s}),
   f"{len({pm2.bucket(t) for t in traj})} buckets / {len(set(_s))} scores")

# Refusals: too little evidence is not evidence.
pm3 = ProgressModel()
check("a_two_frame_trace_teaches_nothing", pm3.fit(traj[:2]), False)
check("still_not_ready_after_a_refused_fit", pm3.ready, False)
check("an_empty_trace_is_refused", ProgressModel().fit([]), False)
check("a_wrong_width_trace_is_refused",
      ProgressModel().fit([np.zeros(3) for _ in range(8)]), False)
# A board that never changes has no direction to learn.
check("a_constant_trajectory_is_refused",
      ProgressModel().fit([F(_solid) for _ in range(8)]), False)


print("\n=== 4. O1 ON O3: the model retires itself when it is wrong (the teeth) ===")

# The doc's rule: a model that cannot be switched off by its own error signal does not
# ship. A LATER winning trajectory is a backtest -- if the score did not rise along a
# path that demonstrably WAS progress, the model is wrong about this game.
pm4 = ProgressModel()
pm4.fit([F(g) for g in consolidating(10)])
check("fitted_and_ready", pm4.ready, True)
# ...now hand it a win that runs the OTHER way (a board dissolving, not consolidating).
contra = [F(g) for g in reversed(consolidating(10))]
pm4.fit(contra)
check("a_contradicting_win_retires_the_model", pm4.ready, False)
check("a_retired_model_scores_zero", pm4.score(F(_solid)), 0.0)
check("a_retired_model_buckets_zero", pm4.bucket(F(_solid)), 0)
check("a_retired_model_refuses_further_fits",
      pm4.fit([F(g) for g in consolidating(10)]), False)
ok("the_backtest_that_retired_it_was_recorded",
   len(pm4._backtests) == 1 and pm4._backtests[0] < ProgressModel.BACKTEST_MIN,
   f"{pm4._backtests}")
# ...whereas a CONSISTENT second win must not retire it, and must be graded before it
# is absorbed (grading a model on data it already learned measures memorisation).
pm5 = ProgressModel()
pm5.fit([F(g) for g in consolidating(10)])
pm5.fit([F(g) for g in consolidating(12)])
check("a_consistent_second_win_keeps_the_model", pm5.ready, True)
check("consistent_wins_accumulate", pm5._fits, 2)
ok("the_second_win_was_backtested_before_absorption",
   len(pm5._backtests) == 1 and pm5._backtests[0] >= ProgressModel.BACKTEST_MIN,
   f"{pm5._backtests}")
# A model with no opinion anywhere scores 0.0 on the backtest, not 1.0 -- silence is
# not agreement.
check("a_flat_backtest_is_a_failure",
      ProgressModel.backtest(pm5, [F(_solid), F(_solid), F(_solid)]), 0.0)


print("\n=== 5. wired into ClickPlanner without becoming a gate ===")

cp = ClickPlanner()
check("planner_starts_with_an_unfitted_model", cp._pm.ready, False)
check("planner_starts_with_an_empty_trace", cp._trace, [])
# The trace must span the WHOLE level, coverage clicks included -- the path to a level
# completion is mostly coverage, and a trace of the searched part only is not the path
# that won.
_g8 = np.zeros((16, 16), dtype=np.int16)
for _i in range(5):
    _g8[_i, _i] = 1
    cp.act(_g8.copy(), StateEncoder(), 0.0)
ok("act_records_the_trajectory", len(cp._trace) == 5, f"{len(cp._trace)} frames")
ok("recorded_frames_are_feature_vectors",
   all(getattr(t, "shape", None) == (ProgressModel.N_FEAT,) for t in cp._trace),
   f"{[getattr(t, 'shape', None) for t in cp._trace][:3]}")

# A LEVEL-UP grounds the model and clears the trace.
cp2 = ClickPlanner()
cp2._trace = [F(g) for g in consolidating(10)]
cp2.new_level()
check("a_level_up_grounds_the_model", cp2._pm.ready, True)
check("a_level_up_clears_the_trace", cp2._trace, [])
# ...and the model SURVIVES the level change: level 1's trajectory is the only evidence
# level 2 will ever get, so a per-level wipe would make O3 permanently useless.
cp2.new_level()
check("the_model_survives_a_second_level", cp2._pm.ready, True)

# A GAME_OVER must NOT ground the model: the road to death is not the road to progress.
cp3 = ClickPlanner()
cp3._trace = [F(g) for g in consolidating(10)]
cp3.reset()
check("a_game_over_clears_the_trace", cp3._trace, [])
check("a_game_over_does_not_ground_the_model", cp3._pm.ready, False)

# The kill switch, and the guarantee that it restores the OLD behaviour exactly.
os.environ["ARC_NO_O3"] = "1"
cp4 = ClickPlanner()
_g4 = np.zeros((16, 16), dtype=np.int16)
cp4.act(_g4, StateEncoder(), 0.0)
check("kill_switch_records_no_trace", cp4._trace, [])
cp4._trace = [F(g) for g in consolidating(10)]
cp4.new_level()
check("kill_switch_never_grounds_the_model", cp4._pm.ready, False)
del os.environ["ARC_NO_O3"]

# O4: the descriptor map is per-search (hashes mean nothing once the layout changes),
# while the learned model is per-game.
cp5 = ClickPlanner()
cp5._feat[123] = np.zeros(ProgressModel.N_FEAT)
cp5._trace = [F(g) for g in consolidating(10)]
cp5.new_level()
check("state_descriptors_are_dropped_on_a_new_layout", cp5._feat, {})
check("but_the_learned_model_is_not", cp5._pm.ready, True)


print("\n=== 5b. grounding prefers the GRAPH path over the wall-clock walk ===")

# MEASURED, lp85 seed 0 (2026-08-02): fitting on `_trace` gave a direction to 2 features
# out of 20. The trace was 95 frames of which ~89 were coverage clicks -- a random walk
# that happened to end well, which has no monotone structure to find. `_G` holds the
# shortest BUTTON path to the board the level was won from: shorter, but every step
# deliberate.
_gg = consolidating(7)
cpG = ClickPlanner()
cpG._enter_search([(5, 5), (25, 25)], from_reset=True)
cpG._root = 100
cpG._G = {100: {0: 101}, 101: {0: 102}, 102: {0: 103}, 103: {}}
cpG._dist = {100: (), 101: (0,), 102: (0, 0), 103: (0, 0, 0)}
cpG._feat = {100 + i: F(_gg[i]) for i in range(4)}
cpG._cur = 103
_gt = cpG._graph_trace()
ok("graph_trace_walks_root_to_here", len(_gt) == 4, f"{len(_gt)} frames")
ok("graph_trace_is_in_path_order",
   bool(np.array_equal(_gt[0], F(_gg[0])) and np.array_equal(_gt[-1], F(_gg[3]))),
   "endpoints match the path")
# EXACT or nothing: a partial reconstruction would be learned from with full confidence.
cpG2 = ClickPlanner()
cpG2._enter_search([(5, 5), (25, 25)], from_reset=True)
cpG2._root, cpG2._cur = 100, 103
cpG2._G, cpG2._dist = {100: {0: 101}}, {103: (0, 0, 0)}     # edge 101->102 missing
cpG2._feat = {100: F(_gg[0]), 101: F(_gg[1])}
check("a_broken_path_yields_nothing", cpG2._graph_trace(), [])
cpG3 = ClickPlanner()
cpG3._enter_search([(5, 5), (25, 25)], from_reset=True)
cpG3._root, cpG3._cur = 100, 103
cpG3._G = {100: {0: 101}, 101: {0: 102}, 102: {0: 103}}
cpG3._dist = {100: (), 103: (0, 0, 0)}
cpG3._feat = {100: F(_gg[0]), 101: F(_gg[1]), 102: F(_gg[2])}   # 103 undescribed
check("a_missing_descriptor_yields_nothing", cpG3._graph_trace(), [])
check("no_root_yields_nothing", ClickPlanner()._graph_trace(), [])
# A dead-edge sentinel must never be walked through.
cpG4 = ClickPlanner()
cpG4._enter_search([(5, 5), (25, 25)], from_reset=True)
cpG4._root, cpG4._cur = 100, 101
cpG4._G, cpG4._dist = {100: {0: -1}}, {101: (0,)}
cpG4._feat = {100: F(_gg[0]), 101: F(_gg[1])}
check("a_dead_edge_yields_nothing", cpG4._graph_trace(), [])
# And the level-up uses it in preference to the walk when it is long enough.
cpG5 = ClickPlanner()
cpG5._enter_search([(5, 5), (25, 25)], from_reset=True)
cpG5._root, cpG5._cur = 100, 106
cpG5._G = {100 + i: {0: 101 + i} for i in range(6)}
cpG5._dist = {100 + i: tuple([0] * i) for i in range(7)}
_long = consolidating(7)
cpG5._feat = {100 + i: F(_long[i]) for i in range(7)}
cpG5._trace = [F(g) for g in reversed(consolidating(10))]   # a CONTRADICTORY walk
cpG5.new_level()
ok("the_graph_path_is_what_grounded_the_model", cpG5._pm.ready, f"fits={cpG5._pm._fits}")
_ss = [cpG5._pm.score(F(g)) for g in _long]
ok("the_model_learned_the_graph_direction_not_the_walk", _ss[-1] > _ss[0],
   f"{_ss[0]:.4f} -> {_ss[-1]:.4f}")
# ...but when there is no usable graph path (level won during coverage), the walk is
# still better than nothing.
cpG6 = ClickPlanner()
cpG6._trace = [F(g) for g in consolidating(10)]
cpG6.new_level()
check("falls_back_to_the_walk_when_the_graph_has_no_path", cpG6._pm.ready, True)


print("\n=== 6. progress ORDERS the frontier; cost breaks the tie (O3 over O2) ===")

# The objectives table says O2 decides "which plan among EQUALS". Keying the frontier on
# cost first meant the search was aimed at nothing and merely aimed cheaply.
def _rig(planner, states):
    """states: {hash: (feature_vector, nav_path_length)}"""
    planner._enter_search([(5, 5), (25, 25)], from_reset=True)
    planner._cur = None
    planner._G = {h: {} for h in states}
    planner._dist = {h: tuple([0] * ln) for h, (_f, ln) in states.items()}
    planner._feat = {h: fv for h, (fv, _l) in states.items()}
    from collections import deque as _dq
    planner._queue = _dq(sorted(states))
    return planner

_good, _bad = F(_good_g), F(_bad_g)          # this trajectory's own end / start states
_win = [F(g) for g in consolidating(10)]
# Unfitted: no opinion -> the CHEAPER state wins, exactly as before this component.
cpA = _rig(ClickPlanner(), {11: (_good, 6), 22: (_bad, 1)})
cpA._build_next_plan()
check("without_a_fitted_model_cost_decides", cpA._pending[1], 22)
# Fitted: the state the model calls better wins even though it costs 6 instead of 1.
cpB = _rig(ClickPlanner(), {11: (_good, 6), 22: (_bad, 1)})
cpB._pm.fit(_win)
cpB._build_next_plan()
check("with_a_fitted_model_progress_decides", cpB._pending[1], 11)
# ...but among states of EQUAL progress, cost still decides. O3 leads, it does not
# swallow O2.
cpC = _rig(ClickPlanner(), {11: (_good, 7), 22: (_good, 2)})
cpC._pm.fit(_win)
cpC._build_next_plan()
check("among_equals_cost_still_decides", cpC._pending[1], 22)
# A missing descriptor scores as NEUTRAL (0), which is right -- an unseen board is not
# known to be bad. What must not happen is neutral beating a state the model knows to
# be GOOD, at equal cost.
cpD = _rig(ClickPlanner(), {11: (_good, 4), 22: (_bad, 4)})
cpD._pm.fit(_win)
del cpD._feat[22]
cpD._build_next_plan()
ok("an_undescribed_state_does_not_outrank_a_known_good_one",
   cpD._pending[1] == 11, f"chose {cpD._pending[1]}")
# O4: O3 must never make a reachable state unreachable. Even ranked last, every
# frontier state is still eventually probed.
cpE = _rig(ClickPlanner(), {11: (_good, 3), 22: (_bad, 3)})
cpE._pm.fit(_win)
_picked = set()
for _ in range(8):
    if not cpE._build_next_plan():
        break
    _picked.add(cpE._pending[1])
    _s2, _b2 = cpE._pending[1], cpE._pending[2]
    cpE._G.setdefault(_s2, {})[_b2] = _s2                 # record the probe as done
ok("a_low_ranked_state_is_still_reached", _picked == {11, 22}, f"probed {sorted(_picked)}")


print("\n=== 7. A1: deaths as a CONTROL group ===")

# The measured lp85 situation, in miniature. Feature 0 climbs on the winning trajectory
# AND on every losing one -- it is elapsed time. Feature 16 climbs only on the win.
def _traj(n, rising, falling=()):
    out = []
    for t in range(n):
        v = np.zeros(ProgressModel.N_FEAT, dtype=np.float64)
        for j in rising:
            v[j] = t / float(n)
        for j in falling:
            v[j] = 1.0 - t / float(n)
        out.append(v)
    return out


_w = _traj(10, rising=(0, 16))          # win: time-feature AND a real progress feature
_d1 = _traj(10, rising=(0,))            # deaths: only the time-feature moves
_d2 = _traj(8, rising=(0,))

pm = ProgressModel()
pm.fit(_w)
check("without_a_control_both_features_carry_weight",
      int(np.count_nonzero(pm.effective())), 2)
# One death is an anecdote: below DEATH_MIN the control must not fire.
pm.fit_death(_d1)
check("one_death_is_not_yet_a_control",
      int(np.count_nonzero(pm.effective())), 2)
pm.fit_death(_d2)
check("the_confounded_feature_is_cancelled",
      float(pm.effective()[0]), 0.0)
ok("the_real_feature_survives_the_control", pm.effective()[16] > 0,
   f"weight={pm.effective()[16]:.3f}")

# Deaths arrive AFTER the win on lp85, so the control must apply retroactively -- this
# is why effective() is lazy rather than folded into fit().
pmL = ProgressModel()
pmL.fit(_w)
_before = pmL.score(_traj(2, rising=(0,))[1])
pmL.fit_death(_d1)
pmL.fit_death(_d2)
ok("a_death_after_the_win_still_cleans_the_direction",
   pmL.score(_traj(2, rising=(0,))[1]) < _before,
   f"{_before:.4f} -> {pmL.score(_traj(2, rising=(0,))[1]):.4f}")

# Deduplication: lp85 replays the same graph path 5 times. Repeats must not count as
# independent evidence, or the control becomes a vote by whichever path search repeats.
pmD = ProgressModel()
pmD.fit(_w)
ok("the_first_death_is_absorbed", pmD.fit_death(_d1) is True)
ok("an_identical_death_path_is_not_new_evidence", pmD.fit_death(_d1) is False)
check("a_replayed_path_does_not_reach_DEATH_MIN", pmD._death_fits, 1)

# O4 / safety: the control can only SUBTRACT. Even a death channel that agrees with the
# win everywhere must degrade the model to neutral, never flip it to a wrong direction.
pmS = ProgressModel()
pmS.fit(_w)
pmS.fit_death(_traj(10, rising=(0, 16)))
pmS.fit_death(_traj(9, rising=(0, 16)))
ok("a_fully_confounded_model_degrades_to_neutral_not_to_wrong",
   float(np.abs(pmS.effective()).sum()) == 0.0,
   f"|eff|={float(np.abs(pmS.effective()).sum()):.4f}")
ok("a_neutral_model_scores_every_board_the_same",
   pmS.score(_traj(2, rising=(0, 16))[1]) == 0.0)

# The death channel must never manufacture a direction on its own: with no win, O3 is
# still not ready and still a strict no-op (the 23 games that never win).
pmN = ProgressModel()
pmN.fit_death(_d1)
pmN.fit_death(_d2)
ok("deaths_alone_do_not_make_the_model_ready", pmN.ready is False)
check("deaths_alone_score_nothing", pmN.score(_w[-1]), 0.0)

# TOTAL, like every other entry point: junk in must not raise.
pmT = ProgressModel()
for _nm, _bad in [("empty", []), ("short", _w[:2]), ("wrong_width", [np.zeros(3)] * 6),
                  ("ragged", [np.zeros(ProgressModel.N_FEAT), np.zeros(3)])]:
    try:
        pmT.fit_death(_bad)
        ok(f"fit_death_survives_{_nm}", True)
    except Exception as e:
        ok(f"fit_death_survives_{_nm}", False, repr(e))

# A retired model stays retired -- deaths must not resurrect it.
pmR = ProgressModel()
pmR.fit(_w)
pmR._retired = True
ok("a_retired_model_refuses_deaths_too", pmR.fit_death(_d1) is False)


print("\n=== 8. C1: the object half of the basis ===")

_K = ProgressModel.N_COLORS + 4          # first object slot
check("the_vector_carries_an_object_block", ProgressModel.N_FEAT - _K,
      ProgressModel.N_OBJ)

# Same contract as the pixel half: TOTAL, and every entry a fraction.
for _nm, _bad in [("empty", np.zeros((0, 0), np.int16)),
                  ("1d", np.zeros(5, np.int16)),
                  ("3d", np.zeros((2, 2, 2), np.int16)),
                  ("none", None),
                  ("uniform", np.full((8, 8), 3, np.int16)),
                  ("negative", -np.ones((4, 4), np.int16)),
                  ("huge_colour", np.full((4, 4), 999, np.int16))]:
    try:
        _v = ProgressModel.features(_bad)
        ok(f"object_block_is_total_on_{_nm}",
           _v.shape == (ProgressModel.N_FEAT,)
           and bool(np.all(np.isfinite(_v)))
           and bool(np.all(_v >= -1e-9)) and bool(np.all(_v <= 1 + 1e-9)))
    except Exception as e:
        ok(f"object_block_is_total_on_{_nm}", False, repr(e))

# It must actually SEE objects: one blob and four blobs of the same total mass differ in
# the object half while being identical in the colour histogram. This is the whole reason
# C1 exists -- the histogram cannot tell "assembled" from "scattered".
_one = np.zeros((16, 16), np.int16)
_one[2:6, 2:6] = 1                                   # 16 cells, one block
_four = np.zeros((16, 16), np.int16)
for _r, _c in [(2, 2), (2, 10), (10, 2), (10, 10)]:
    _four[_r:_r + 2, _c:_c + 2] = 1                  # 16 cells, four blocks
_vo, _vf = ProgressModel.features(_one), ProgressModel.features(_four)
ok("the_colour_histogram_cannot_tell_them_apart",
   bool(np.allclose(_vo[:ProgressModel.N_COLORS], _vf[:ProgressModel.N_COLORS])))
ok("the_object_block_can", not np.allclose(_vo[_K:], _vf[_K:]),
   f"n_objects {_vo[_K]:.4f} vs {_vf[_K]:.4f}")

# Scale-free: the same picture at twice the size must describe the same situation.
_big = np.zeros((32, 32), np.int16)
_big[4:12, 4:12] = 1
ok("the_object_block_is_scale_free",
   abs(ProgressModel.features(_big)[_K + 2] - _vo[_K + 2]) < 0.02,
   f"largest_frac {_vo[_K + 2]:.4f} vs {ProgressModel.features(_big)[_K + 2]:.4f}")

# The kill switch must zero the block WITHOUT changing the vector width -- a switch that
# resized the basis would make a saved direction silently mean something else.
_saved = ProgressModel._OBJ_ON
try:
    ProgressModel._OBJ_ON = False
    _off = ProgressModel.features(_four)
    check("the_kill_switch_keeps_the_width", _off.shape, (ProgressModel.N_FEAT,))
    ok("the_kill_switch_zeroes_the_object_block",
       bool(np.all(_off[_K:] == 0.0)))
    ok("the_kill_switch_leaves_the_pixel_half_alone",
       bool(np.allclose(_off[:_K], _vf[:_K])))
finally:
    ProgressModel._OBJ_ON = _saved

# A1 and C1 must compose: the control has to be able to cancel an object feature too.
_wc = []
_dc = []
for _t in range(8):
    _g = np.zeros((16, 16), np.int16)
    _g[2:2 + _t + 1, 2:6] = 1
    _wc.append(ProgressModel.features(_g))
    _dc.append(ProgressModel.features(_g))
_pmC = ProgressModel()
_pmC.fit(_wc)
_n_before = int(np.count_nonzero(_pmC.effective()))
_pmC.fit_death(_dc)
_pmC.fit_death(_dc[:-1])
ok("the_control_also_cancels_object_features",
   int(np.count_nonzero(_pmC.effective())) < _n_before,
   f"{_n_before} -> {int(np.count_nonzero(_pmC.effective()))} features survive")


print("\n=== 9. progress ranks are relative to the frontier, not to an absolute grid ===")

# The measured defect: real lp85 level-2 boards score within 0.036 of each other while
# one absolute bucket step is 1/BUCKETS = 0.125, so bucket() gave them all the SAME rank
# and O3 ordered nothing. buckets() must separate boards whose scores differ at all.
pm9 = ProgressModel()
pm9.fit([ProgressModel.features(g) for g in consolidating(8)])

# Five boards separated along a feature the model actually weights, by a margin far
# below one absolute bucket step (1/BUCKETS = 0.125). This is the real lp85 geometry:
# a genuine ordering hiding inside a span the absolute grid rounds away.
_effj = int(np.argmax(np.abs(pm9.effective())))
_base9 = ProgressModel.features(np.zeros((16, 16), np.int16))
_tiny = []
for _i in range(5):
    _v = _base9.copy()
    _v[_effj] += _i * 0.002                 # total span ~0.008 << 0.125
    _tiny.append(_v)
_sc9 = [pm9.score(f) for f in _tiny]
_abs = {pm9.bucket(f) for f in _tiny}
_rel = set(pm9.buckets(_tiny))
ok("the_absolute_grid_collapses_a_narrow_frontier",
   len(_abs) == 1, f"absolute buckets={_abs} over score spread "
                   f"{max(_sc9) - min(_sc9):.6f}")
ok("relative_ranks_separate_it", len(_rel) > 1, f"relative buckets={sorted(_rel)}")

# The reason bucketing exists at all must survive: ranks stay COARSE, so O2 can still
# break ties. Distinct ranks must never exceed BUCKETS + 1.
ok("ranks_are_still_coarse", len(_rel) <= ProgressModel.BUCKETS + 1,
   f"{len(_rel)} distinct ranks, cap {ProgressModel.BUCKETS + 1}")

# Order must match the score order -- a rank that disagrees with the model is worse
# than no rank at all.
_pairs = sorted(zip(_sc9, pm9.buckets(_tiny)))
ok("ranks_agree_with_the_underlying_score",
   all(b1 <= b2 for (_, b1), (_, b2) in zip(_pairs, _pairs[1:])),
   f"{[b for _, b in _pairs]}")

# Genuinely identical boards must NOT be given a spurious ordering.
_same = [ProgressModel.features(np.zeros((8, 8), np.int16))] * 4
check("identical_boards_get_identical_ranks", len(set(pm9.buckets(_same))), 1)
check("an_empty_frontier_is_handled", pm9.buckets([]), [])

# An unfitted / retired model still says nothing at all.
check("an_unfitted_model_ranks_everything_zero",
      set(ProgressModel().buckets(_tiny)), {0})
pm9r = ProgressModel()
pm9r.fit([ProgressModel.features(g) for g in consolidating(8)])
pm9r._retired = True
check("a_retired_model_ranks_everything_zero", set(pm9r.buckets(_tiny)), {0})

# An undescribed board ranks with the worst known one, never above it.
_mixed = [_tiny[0], None, _tiny[-1]]
_mb = pm9.buckets(_mixed)
ok("an_undescribed_board_never_outranks_a_known_one",
   _mb[1] <= max(_mb[0], _mb[2]), f"ranks={_mb}")
check("a_frontier_of_only_unknowns_is_flat", set(pm9.buckets([None, None])), {0})

# O4, again and at the new granularity: a finer ordering must still not make any
# frontier state unreachable.
cp9 = _rig(ClickPlanner(), {11: (_good, 3), 22: (_bad, 3)})
cp9._pm.fit(_win)
_seen9 = set()
for _ in range(8):
    if not cp9._build_next_plan():
        break
    _seen9.add(cp9._pending[1])
    _s9, _b9 = cp9._pending[1], cp9._pending[2]
    cp9._G.setdefault(_s9, {})[_b9] = _s9
ok("relative_ranking_still_reaches_every_state", _seen9 == {11, 22},
   f"probed {sorted(_seen9)}")


print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)

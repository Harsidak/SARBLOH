# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 2b: the SYNTHESIZER
(masked_diff / masked_equal, EnumerativeSynthesizer, WorldModelManager.verify).

Doctrine: the component STAYS in my_agent.py. This file only imports it and
feeds hand-built transitions whose correct answer we know by construction, so a
failure is visible for the synthesizer alone -- no game engine, no LLM, no
planner. test_griddsl.py already covers the OPS themselves; what is under test
here is which EVIDENCE the synthesizer decides to judge a rule on, and under
which comparison.

The thing this pins down. The world model used to compare grids RAW while
StateGraph.changed_masked, the state hash and the PatchWorldModel all masked the
HUD, so one ticking timer pixel could mark a correct movement rule wrong. The
fix is a single shared comparison (masked_diff/masked_equal) plus a per-action
premise check (EnumerativeSynthesizer.evidence): the mask's premise is "these
cells tick no matter what you do, so they say nothing about your action", and for
an action whose EVERY observed change lies inside the mask that premise is false,
so such an action -- and only such an action -- is judged raw.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_synth.py
"""
import sys
import time

import numpy as np

from my_agent import (Transition, StateEncoder, EnumerativeSynthesizer,
                      WorldModelManager, GridDSL, masked_diff, masked_equal,
                      MIN_SAMPLES_PER_RULE, MIN_CHANGED_FOR_VERIFY,
                      FIT_MEMO_MAX)

E = StateEncoder()
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
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        FAILS.append(name)


def warn(name, cond, detail):
    tag = "ok  " if cond else "WARN"
    if not cond:
        WARNS.append(name)
    print(f"  {tag}  {name}: {detail}")


def T(prev, action, nxt):
    return Transition(prev=prev, action=action, next=nxt, reward=0.0,
                      diff_encoding="", timestamp=time.time())


def blank(n=10):
    return np.zeros((n, n), dtype=int)


def syn(mask=None):
    """A synthesizer whose mask_fn hands back exactly `mask` (None => raw)."""
    return EnumerativeSynthesizer(E, mask_fn=(None if mask is None else (lambda: mask)))


# ---------------------------------------------------------------- fixtures
# A sprite of colour 2 walking right along row 5, one column per ACTION2, with a
# one-cell "timer" at (0,0) that ticks on EVERY transition. The sprite move is
# the world; the timer is HUD. Correct answer by construction: ACTION2 shifts
# colour 2 by (0, +1).
#
# The timer counts in 8..13 on purpose. A colour-scoped op moves EVERY pixel of
# its colour, so a timer that happens to read `2` is a second colour-2 object and
# the rule correctly-but-uselessly translates it too -- the fixture would then be
# asserting a colour collision rather than the masking behaviour under test.
TIMER = (0, 0)
MOVER = []
for i in range(5):
    p, n = blank(), blank()
    p[5, 2 + i] = 2
    n[5, 3 + i] = 2
    p[TIMER] = 8 + i                      # timer ticks 8->9, 9->10, ... 12->13
    n[TIMER] = 9 + i
    MOVER.append(T(p, "ACTION2", n))

# The same timer ticking with NOTHING else happening: a HUD-only action.
HUD_ONLY = []
for i in range(3):
    p, n = blank(), blank()
    p[5, 2] = 2                           # sprite present but stationary
    n[5, 2] = 2
    p[TIMER] = 8 + i
    n[TIMER] = 9 + i
    HUD_ONLY.append(T(p, "ACTION3", n))

# A single changed transition -- below MIN_SAMPLES_PER_RULE on its own.
_p, _n = blank(), blank()
_p[7, 7] = 4
_n[7, 8] = 4
LONE = [T(_p, "ACTION4", _n)]

# A transition where nothing at all moved, and one where the grid RESIZED.
_s = blank()
NOOP = [T(_s.copy(), "ACTION5", _s.copy())]
RESIZE = [T(blank(10), "ACTION7", blank(12))]

MASK = np.zeros((10, 10), dtype=bool)
MASK[TIMER] = True                        # precise: the timer cell only

WIDE = np.zeros((10, 10), dtype=bool)
WIDE[0, :] = True                         # broad: the whole top row is HUD

# A sprite of colour 3 moving INSIDE the masked row -- every observed change of
# this action lies under WIDE, so the mask's premise fails for it.
INROW = []
for i in range(5):
    p, n = blank(), blank()
    p[0, 1 + i] = 3
    n[0, 2 + i] = 3
    INROW.append(T(p, "ACTION9", n))


print("== masked_diff / masked_equal: the shared comparison ==")
a = blank(); a[5, 5] = 2
b = a.copy()
ok("identical_grids_equal", masked_equal(a, b))
ok("identical_diff_all_false", masked_diff(a, b) is not None
   and not masked_diff(a, b).any())
# A shape change is not a cell-wise question at all -> None, and never "equal".
ok("shape_change_diff_is_none", masked_diff(blank(10), blank(12)) is None)
ok("shape_change_not_equal", not masked_equal(blank(10), blank(12)))
ok("shape_change_not_equal_with_mask", not masked_equal(blank(10), blank(12), MASK))
# Differing ONLY in a masked cell: unequal raw, equal under the mask. This is
# the exact situation that used to mark a correct movement rule wrong.
c = a.copy(); c[TIMER] = 7
ok("timer_only_differs_raw", not masked_equal(a, c))
ok("timer_only_equal_masked", masked_equal(a, c, MASK))
check("timer_only_raw_diff_count", int(masked_diff(a, c).sum()), 1)
check("timer_only_masked_diff_count", int(masked_diff(a, c, MASK).sum()), 0)
# A real change plus a masked change: still unequal, and the mask subtracts only
# the HUD cell. Masking must never hide a genuine difference.
d = a.copy(); d[TIMER] = 7; d[5, 5] = 0; d[5, 6] = 2
ok("real_change_survives_mask", not masked_equal(a, d, MASK))
check("real_change_raw_diff_count", int(masked_diff(a, d).sum()), 3)
check("real_change_masked_diff_count", int(masked_diff(a, d, MASK).sum()), 2)
# A mask learned on a different grid size is ignored, not raised: masks are
# built online and a level change can resize the grid under them.
ok("wrong_shape_mask_ignored", not masked_equal(a, c, np.ones((3, 3), dtype=bool)))
check("wrong_shape_mask_diff_count",
      int(masked_diff(a, c, np.ones((3, 3), dtype=bool)).sum()), 1)
# mask=None is exactly the raw comparison.
ok("none_mask_is_raw", masked_equal(a, b, None) and not masked_equal(a, c, None))
# An all-True mask makes everything equal -- degenerate but must not crash.
ok("all_true_mask_equalises", masked_equal(a, d, np.ones((10, 10), dtype=bool)))

print("\n== hud_mask(): late binding, and never raising ==")
ok("no_mask_fn_is_none", syn().hud_mask() is None)
ok("mask_fn_returns_mask", syn(MASK).hud_mask() is MASK)


def _boom():
    raise RuntimeError("graph rebuilt under us")


ok("raising_mask_fn_is_none",
   EnumerativeSynthesizer(E, mask_fn=_boom).hud_mask() is None)
ok("non_array_mask_fn_is_none",
   EnumerativeSynthesizer(E, mask_fn=lambda: "not a grid").hud_mask() is None)
# Late binding: the value is read on every call, so a graph that improves its
# mask later is picked up without rebuilding the synthesizer.
_box = {"m": None}
_late = EnumerativeSynthesizer(E, mask_fn=lambda: _box["m"])
ok("late_bound_before", _late.hud_mask() is None)
_box["m"] = MASK
ok("late_bound_after", _late.hud_mask() is MASK)

print("\n== evidence(): which samples, under which comparison ==")
ev_raw = syn().evidence(MOVER + HUD_ONLY)
ok("raw_mode_mask_is_none_a2", ev_raw["ACTION2"][1] is None)
ok("raw_mode_mask_is_none_a3", ev_raw["ACTION3"][1] is None)
check("raw_mode_keeps_all_a2", len(ev_raw["ACTION2"][0]), 5)
check("raw_mode_keeps_all_a3", len(ev_raw["ACTION3"][0]), 3)

ev = syn(MASK).evidence(MOVER + HUD_ONLY + LONE + NOOP + RESIZE)
# ACTION2 moves the sprite outside the mask -> plenty of live evidence, so the
# mask legitimately applies and is handed back for scoring.
ok("mover_keeps_mask", ev["ACTION2"][1] is MASK)
check("mover_live_count", len(ev["ACTION2"][0]), 5)
# ACTION3 only ever ticks the timer -> masking would leave it NOTHING, so the
# premise fails and it is judged raw on its full sample set.
ok("hud_only_falls_back_to_raw", ev["ACTION3"][1] is None)
check("hud_only_keeps_raw_samples", len(ev["ACTION3"][0]), 3)
# One changed sample is under MIN_SAMPLES_PER_RULE, so it also cannot support the
# mask; it is returned raw and synthesize will skip it for being too small.
ok("lone_sample_falls_back_to_raw", ev["ACTION4"][1] is None)
check("lone_sample_count", len(ev["ACTION4"][0]), 1)
ok("min_samples_is_two", MIN_SAMPLES_PER_RULE == 2)
# Unchanged and resized transitions are not evidence about anything.
ok("noop_action_absent", "ACTION5" not in ev)
ok("resized_action_absent", "ACTION7" not in ev)
# Every returned sample really is changed and same-shape.
ok("all_returned_samples_changed",
   all(t.changed and t.prev.shape == t.next.shape
       for s, _ in ev.values() for t in s))
# Temporal order within an action is preserved -- synthesize fits samples[-5:].
ok("evidence_preserves_order",
   [t.prev[TIMER] for t in ev["ACTION2"][0]] == [8, 9, 10, 11, 12])
# The broad mask swallows the whole top row, so ACTION9's every change is masked
# -> raw, while ACTION2 (which moves in row 5) keeps the mask. Two actions, two
# comparisons, decided independently and from the same statistic.
ev_w = syn(WIDE).evidence(MOVER + INROW)
ok("inrow_falls_back_to_raw", ev_w["ACTION9"][1] is None)
check("inrow_keeps_all_samples", len(ev_w["ACTION9"][0]), 5)
ok("mover_keeps_wide_mask", ev_w["ACTION2"][1] is WIDE)
# Adding genuine world-changes to a previously HUD-only action re-enables the
# mask and drops the pure ticks: the premise check is a fallback, not a latch.
mixed = [T(t.prev, "ACTION3", t.next) for t in MOVER]
ev_m = syn(MASK).evidence(HUD_ONLY + mixed)
ok("live_evidence_re_enables_mask", ev_m["ACTION3"][1] is MASK)
check("live_evidence_drops_ticks", len(ev_m["ACTION3"][0]), 5)
check("empty_evidence", syn(MASK).evidence([]), {})

print("\n== _score / _fits under a mask ==")
use = [(t.prev, t.next) for t in MOVER]
bgs = [E.detect_background(p) for p, _ in use]
RULE = ("translate_color", {"color": 2, "dr": 0, "dc": 1, "vacate": 0})
# The rule is exactly right about the WORLD and says nothing about the timer:
# scored raw it gets nothing, scored under the mask it gets everything.
check("exact_rule_scores_zero_raw", syn()._score(*RULE, use, bgs), 0)
check("exact_rule_scores_all_masked", syn()._score(*RULE, use, bgs, mask=MASK), 5)
ok("exact_rule_fits_masked", syn()._fits(*RULE, use, bgs, mask=MASK))
ok("exact_rule_does_not_fit_raw", not syn()._fits(*RULE, use, bgs))
# A wrong rule stays wrong under the mask: masking is not a free pass.
WRONG = ("translate_color", {"color": 2, "dr": 0, "dc": -1, "vacate": 0})
check("wrong_rule_scores_zero_masked", syn()._score(*WRONG, use, bgs, mask=MASK), 0)
# `floor` only ever returns a lower bound, never a score above the true one.
ok("floor_never_inflates",
   syn()._score(*RULE, use, bgs, floor=4, mask=MASK) <= 5)
# A rule whose op raises must be survived, not propagated.
check("bad_args_score_zero",
      syn()._score("translate_color", {"color": 2, "dr": 0, "dc": 1,
                                       "nonsense": True}, use, bgs, mask=MASK), 0)

print("\n== synthesize(): the mask decides whether a rule exists at all ==")
# Raw, the timer breaks every candidate on every sample -> no program.
ok("unmasked_synthesis_returns_none", syn().synthesize(MOVER) is None)
prog = syn(MASK).synthesize(MOVER)
ok("masked_synthesis_finds_program", prog is not None)
if prog is not None:
    check("masked_program_rule_count", len(prog.spec), 1)
    check("masked_program_action", prog.spec[0]["action"], "ACTION2")
    ok("masked_program_is_a_shift",
       prog.spec[0]["op"] in ("translate_color", "step_color"),
       f"op was {prog.spec[0]['op']}")
    ok("masked_program_compiled", callable(prog.fn))
    check("masked_program_source", prog.source, "enumerative")
    # It must actually reproduce the world it was fitted on.
    pred = prog.fn(MOVER[0].prev.copy(), "ACTION2")
    ok("masked_program_predicts_move", isinstance(pred, np.ndarray)
       and masked_equal(pred, MOVER[0].next, MASK))
# Degenerate inputs must return None rather than an empty program.
ok("empty_transitions_none", syn(MASK).synthesize([]) is None)
ok("all_noop_transitions_none", syn(MASK).synthesize(NOOP * 5) is None)
ok("all_resized_transitions_none", syn(MASK).synthesize(RESIZE * 5) is None)
ok("lone_sample_gets_no_rule", syn(MASK).synthesize(LONE) is None)
# Banning the rule it found must not hand back the same rule again.
if prog is not None:
    import json
    ban = {json.dumps(prog.spec[0], sort_keys=True)}
    again = syn(MASK).synthesize(MOVER, banned=ban)
    ok("banned_rule_not_returned",
       again is None or again.spec[0] != prog.spec[0],
       f"got {again.spec[0] if again else None} again")
# The premise check's deliberate cost: an action with only masked evidence is
# judged raw, so a rule DESCRIBING the HUD can be learned. That is accepted --
# a rule that predicts the timer correctly cannot mislead the planners (novelty
# and dead-action pruning run off the masked state hash, not the world model),
# whereas refusing to judge the action loses the model outright.
inrow_prog = syn(WIDE).synthesize(INROW)
ok("inrow_gets_a_rule_via_premise_check", inrow_prog is not None)
if inrow_prog is not None:
    check("inrow_rule_action", inrow_prog.spec[0]["action"], "ACTION9")
# Both actions can be served at once, each under its own comparison.
both = syn(WIDE).synthesize(MOVER + INROW)
ok("both_actions_get_rules", both is not None and len(both.spec) == 2,
   f"got {both.spec if both else None}")

print("\n== WorldModelManager.verify(): same split as the fitter ==")
mgr_masked = WorldModelManager(None, E, mask_fn=lambda: MASK)
mgr_raw = WorldModelManager(None, E)
if prog is not None:
    check("verify_exact_rule_masked", mgr_masked.verify(prog.fn, MOVER), 1.0)
    # Raw, the very same rule and the very same transitions score zero: this is
    # the asymmetry that used to zero out verify on every live-HUD game.
    check("verify_exact_rule_raw", mgr_raw.verify(prog.fn, MOVER), 0.0)
# Too little evidence to judge -> 0.0, not a lucky 1.0 off one sample.
ok("min_changed_for_verify_is_four", MIN_CHANGED_FOR_VERIFY == 4)
if prog is not None:
    check("verify_below_min_evidence", mgr_masked.verify(prog.fn, MOVER[:3]), 0.0)
    # A mixed sample must NOT dilute the score: `fn.covers` scopes verify to the
    # actions the program actually claims, because the enumerator's rules are
    # per-action and scoring a one-action program on actions it never claimed
    # measured nothing about it (it rejected every partial program). So adding 3
    # unclaimed ACTION3 transitions leaves the ACTION2 verdict untouched.
    check("verify_scoped_to_claimed_actions",
          mgr_masked.verify(prog.fn, MOVER + HUD_ONLY), 1.0)
    ok("compiled_fn_advertises_covers", getattr(prog.fn, "covers", None) == {"ACTION2"},
       f"covers was {getattr(prog.fn, 'covers', None)!r}")
    # Scoping is not a loophole: if the CLAIMED action itself lacks enough
    # evidence, verify refuses to score rather than passing on a thin sample.
    check("verify_thin_claim_refuses",
          mgr_masked.verify(prog.fn, MOVER[:3] + HUD_ONLY), 0.0)
if inrow_prog is not None:
    mgr_wide = WorldModelManager(None, E, mask_fn=lambda: WIDE)
    check("verify_premise_checked_rule", mgr_wide.verify(inrow_prog.fn, INROW), 1.0)
# A function that raises or returns junk must score 0.0, never crash the agent.
check("verify_raising_fn", mgr_masked.verify(
    lambda g, a: (_ for _ in ()).throw(RuntimeError("boom")), MOVER), 0.0)
check("verify_non_array_fn", mgr_masked.verify(lambda g, a: "nope", MOVER), 0.0)
check("verify_no_transitions", mgr_masked.verify(lambda g, a: g, []), 0.0)

print("\n== the fitter and the verifier cannot disagree ==")
# The point of routing both through evidence(): whatever comparison a rule was
# LEARNED under is the one it is SCORED under. Assert it directly -- for every
# action, the mask evidence() hands the fitter is the mask verify() will use.
for tset in (MOVER + HUD_ONLY, MOVER + INROW, HUD_ONLY + LONE):
    for s in (syn(MASK), syn(WIDE)):
        e1 = s.evidence(tset)
        e2 = s.evidence(tset)
        ok("evidence_is_deterministic",
           all(e1[k][1] is e2[k][1] and len(e1[k][0]) == len(e2[k][0]) for k in e1))
# And a rule fitted under the mask verifies at 1.0 under that same mask, for
# every mask granularity -- precise cell, broad row, and no mask at all.
for name, m in (("precise", MASK), ("broad", WIDE)):
    pr = syn(m).synthesize(MOVER)
    if pr is None:
        ok(f"roundtrip_{name}_synthesised", False, "no program")
        continue
    v = WorldModelManager(None, E, mask_fn=lambda mm=m: mm).verify(pr.fn, MOVER)
    check(f"roundtrip_{name}_verifies_1.0", v, 1.0)
# With no HUD in the data at all, masking changes nothing: a clean mover is
# learned and verified identically raw or masked. The mask must be inert when
# there is nothing to mask.
CLEAN = []
for i in range(5):
    p, n = blank(), blank()
    p[5, 2 + i] = 2
    n[5, 3 + i] = 2
    CLEAN.append(T(p, "ACTION2", n))
p_raw, p_msk = syn().synthesize(CLEAN), syn(MASK).synthesize(CLEAN)
ok("clean_data_raw_synthesises", p_raw is not None)
ok("clean_data_masked_synthesises", p_msk is not None)
if p_raw is not None and p_msk is not None:
    check("mask_inert_on_clean_data", p_msk.spec, p_raw.spec)
    check("clean_verify_raw", mgr_raw.verify(p_raw.fn, CLEAN), 1.0)
    check("clean_verify_masked", mgr_masked.verify(p_msk.fn, CLEAN), 1.0)

print("\n== judge(): the ONLINE accuracy uses the comparison verify used ==")
# This is the asymmetry in the place it did the most damage. The online tracker
# is what DEMOTES a model and bans its rules, and it compared raw: a timer tick
# alongside a real move scored a perfect model 0.0 every step, so a correct model
# decayed past DEMOTE_ACCURACY and banned its own correct rules.
if prog is not None:
    jm = WorldModelManager(None, E, mask_fn=lambda: MASK)
    ok("judge_none_without_active_model",
       jm.judge(MOVER[0].prev, "ACTION2", MOVER[0].next) is None)
    jm.verify(prog.fn, MOVER)                 # leaves the per-action mask snapshot
    ok("verify_snapshots_mask_for", jm._mask_for.get("ACTION2") is MASK)
    jm.active_model = prog
    # A real move that also ticked the timer: correct model, so 1.0.
    check("judge_correct_move_masked",
          jm.judge(MOVER[0].prev, "ACTION2", MOVER[0].next), 1.0)
    # The identical call on a manager that compares raw returns 0.0 -- the bug,
    # pinned so it cannot come back.
    jr = WorldModelManager(None, E)
    jr.verify(prog.fn, MOVER)
    jr.active_model = prog
    check("judge_correct_move_raw_is_the_bug",
          jr.judge(MOVER[0].prev, "ACTION2", MOVER[0].next), 0.0)
    # A HUD-only tick is not evidence for or against the model -> not scored.
    tick_p, tick_n = MOVER[0].prev.copy(), MOVER[0].prev.copy()
    tick_n[TIMER] = 3
    ok("judge_hud_only_tick_unscored",
       jm.judge(tick_p, "ACTION2", tick_n) is None)
    # An action the program never claimed is not scored either.
    ok("judge_unclaimed_action_unscored",
       jm.judge(HUD_ONLY[0].prev, "ACTION3", HUD_ONLY[0].next) is None)
    # A genuinely wrong prediction still scores 0.0: masking is not a free pass.
    wrong_p, wrong_n = blank(), blank()
    wrong_p[5, 5] = 2
    wrong_n[5, 4] = 2                         # moved LEFT; the rule says right
    wrong_n[TIMER] = 9
    check("judge_wrong_move_scores_zero", jm.judge(wrong_p, "ACTION2", wrong_n), 0.0)
    # Nothing at all happened -> None, never a free 1.0.
    ok("judge_identical_grids_unscored",
       jm.judge(MOVER[0].prev, "ACTION2", MOVER[0].prev.copy()) is None)
    # An action no verify ever saw falls back to the graph's mask rather than to
    # a raw comparison: the model models the world, not the chrome.
    jf = WorldModelManager(None, E, mask_fn=lambda: MASK)
    jf.active_model = prog
    check("judge_falls_back_to_graph_mask", jf._mask_for, {})
    check("judge_fallback_still_masked",
          jf.judge(MOVER[0].prev, "ACTION2", MOVER[0].next), 1.0)
    # A shape change cannot be judged cell-wise -> unscored, not a miss.
    ok("judge_shape_change_unscored",
       jf.judge(MOVER[0].prev, "ACTION2", blank(12)) is None)


# ======================================================================
print("\n== MECHANICAL EXPRESSIVENESS: colour permutations (remap_colors) ==")
# ======================================================================
# WHY THIS SECTION EXISTS. The 25-game diagnostic showed the mechanical majority
# of these games are colour permutations, state toggles and base-state resets,
# and the DSL could say none of them: swap_colors covers a 2-cycle and recolor a
# single repaint, so a 3-cycle had no representation at all and the whole game
# ended with world_model = None. The mapping is READABLE OFF THE DATA (every
# changed cell says "src became dst"), so it is proposed by counting rather than
# swept -- the space of colour maps is |palette|^|palette| and no enumeration
# could reach it. Everything proposed is then verified exactly like a swept rule.

# --- a 3-cycle, with the same HUD timer ticking underneath ----------------
# 1->2->3->1 on a static board. Correct answer by construction.
CYCLE = []
_cyc = {1: 2, 2: 3, 3: 1}
_base = blank()
_base[3, 1:4] = [1, 2, 3]
_base[6, 1:4] = [3, 1, 2]
_cur = _base.copy()
for i in range(5):
    p = _cur.copy()
    n = GridDSL.remap_colors(p, 0, mapping=_cyc)
    p[TIMER] = 8 + i                      # HUD ticks on every transition
    n[TIMER] = 9 + i
    CYCLE.append(T(p, "ACTION5", n))
    _cur = n.copy()
    _cur[TIMER] = 0

cyc_prog = syn(MASK).synthesize(CYCLE)
ok("three_cycle_synthesises", cyc_prog is not None,
   "a 3-cycle had NO representation before remap_colors")
if cyc_prog is not None:
    check("three_cycle_rule_count", len(cyc_prog.spec), 1)
    check("three_cycle_op", cyc_prog.spec[0]["op"], "remap_colors")
    check("three_cycle_mapping", cyc_prog.spec[0]["args"]["mapping"], _cyc)
    # The bar the goal sets: it must VERIFY 1.0 under the masked comparison.
    check("three_cycle_verifies_masked",
          WorldModelManager(None, E, mask_fn=lambda: MASK).verify(cyc_prog.fn, CYCLE),
          1.0)
    # ... and reproduce a frame it was never fitted on.
    _held = blank()
    _held[8, 0:3] = [2, 3, 1]
    ok("three_cycle_generalises",
       masked_equal(cyc_prog.fn(_held.copy(), "ACTION5"),
                    GridDSL.remap_colors(_held, 0, mapping=_cyc), MASK))
    # A single serialisable spec: the ban ledger and the LLM prompt both go
    # through json, and an int-keyed mapping has to survive that.
    import json as _json
    ok("three_cycle_spec_serialises",
       isinstance(_json.dumps(cyc_prog.spec[0], sort_keys=True), str))

# --- a two-config TOGGLE: the map is its own inverse ----------------------
# The board flips between two colour configurations on every press. This is the
# "state toggle" class; note the samples alternate direction, so a rule that is
# not an involution fits at most half of them.
TOGGLE = []
_tog = {1: 3, 3: 1, 2: 4, 4: 2}
_t = blank()
_t[2, 2:6] = [1, 2, 3, 4]
_t[7, 2:6] = [4, 3, 2, 1]
for i in range(6):
    p = _t.copy()
    n = GridDSL.remap_colors(p, 0, mapping=_tog)
    p[TIMER] = 8 + i
    n[TIMER] = 9 + i
    TOGGLE.append(T(p, "ACTION6", n))
    _t = n.copy()
    _t[TIMER] = 0

tog_prog = syn(MASK).synthesize(TOGGLE)
ok("toggle_synthesises", tog_prog is not None)
if tog_prog is not None:
    check("toggle_op", tog_prog.spec[0]["op"], "remap_colors")
    check("toggle_verifies_masked",
          WorldModelManager(None, E, mask_fn=lambda: MASK).verify(tog_prog.fn, TOGGLE),
          1.0)
    # An involution: applying the learned rule twice returns the original board.
    _once = tog_prog.fn(TOGGLE[0].prev.copy(), "ACTION6")
    _twice = tog_prog.fn(_once.copy(), "ACTION6")
    ok("toggle_is_an_involution",
       masked_equal(_twice, TOGGLE[0].prev, MASK),
       "two presses must return the board it started on")

# --- the proposal is COUNTED, not swept ----------------------------------
_s5 = syn(MASK)
_maps = _s5._remap_candidates([(t.prev, t.next) for t in CYCLE], MASK)
ok("remap_candidates_are_few", 0 < len(_maps) <= 2,
   f"{len(_maps)} candidates proposed -- counting, not sweeping")
check("remap_candidate_reads_the_cycle", _maps[0], _cyc)
# The HUD must not vote: unmasked, the ticking timer injects bogus entries.
_raw_maps = _s5._remap_candidates([(t.prev, t.next) for t in CYCLE], None)
ok("hud_pollutes_the_map_when_unmasked", _raw_maps and _raw_maps[0] != _cyc,
   f"raw map {_raw_maps[0] if _raw_maps else None} picks up the timer digits")
ok("mask_keeps_the_map_clean", _maps[0] == _cyc,
   "the fitter and the verifier read the same comparison")
# Degenerate evidence proposes nothing rather than a junk map.
check("remap_no_change_no_candidate",
      _s5._remap_candidates([(blank(), blank())], MASK), [])
check("remap_empty_samples_no_candidate", _s5._remap_candidates([], MASK), [])
check("remap_resize_samples_no_candidate",
      _s5._remap_candidates([(blank(10), blank(12))], MASK), [])
# A purely SPATIAL transition may well produce a tidy histogram; the point is
# that nothing is trusted on the histogram's word -- synthesize still picks the
# rule that actually reproduces the grids, which for MOVER is the shift.
_mv_prog = syn(MASK).synthesize(MOVER)
ok("spatial_transition_keeps_its_shift_rule",
   _mv_prog is not None and _mv_prog.spec[0]["op"] in ("translate_color", "step_color"),
   f"op was {_mv_prog.spec[0]['op'] if _mv_prog else None} -- "
   "a proposable colour map must not displace an exact object rule")

# --- the per-action fit MEMO is an equivalence, not an approximation ------
# synthesize() re-fits every action's rule on every call, but only one action
# gains a sample per step: profiling ka59 charged 344.8s of a 398.7s run to
# synthesize, 294.5s of it to _score. The memo must therefore be provably
# output-identical -- a cache that changes ANY rule is a silent behaviour
# change wearing a performance change's clothes.
_LOG = MOVER + HUD_ONLY + CYCLE + TOGGLE

# 1. Two records made in the same clock tick must not share an identity.
#    time.time() has ~16ms resolution on Windows and the agent records far
#    faster than that, so `timestamp` collides; `_seq` is why the key is exact.
_t1, _t2 = T(blank(), "ACTION1", blank()), T(blank(), "ACTION1", blank())
ok("transition_seq_is_unique", _t1._seq != _t2._seq,
   f"{_t1._seq} vs {_t2._seq} -- a collision would reuse another action's rule")

# 2. A WARM synthesizer walked up the log prefix by prefix must agree, at every
#    prefix, with a COLD one that has never seen anything. This is the live
#    call pattern: choose_action re-synthesises as the log grows.
_warm = syn(MASK)
_mismatch = None
for _i in range(2, len(_LOG) + 1):
    _pre = _LOG[:_i]
    _w = _warm.synthesize(_pre)
    _c = syn(MASK).synthesize(_pre)
    _ws = None if _w is None else sorted(map(repr, _w.spec))
    _cs = None if _c is None else sorted(map(repr, _c.spec))
    if _ws != _cs:
        _mismatch = (_i, _ws, _cs)
        break
ok("memo_matches_a_cold_synthesizer_at_every_prefix", _mismatch is None,
   f"prefix {_mismatch[0]}: warm {_mismatch[1]} vs cold {_mismatch[2]}" if _mismatch else "")

# 3. And it must actually save the work it exists to save. Second call on the
#    SAME log does no scoring at all.
_calls = {"n": 0}
_real_score = EnumerativeSynthesizer._score


def _counting_score(self, *a, **k):
    _calls["n"] += 1
    return _real_score(self, *a, **k)


EnumerativeSynthesizer._score = _counting_score
try:
    _hot = syn(MASK)
    _hot.synthesize(_LOG)
    _first = _calls["n"]
    _calls["n"] = 0
    _hot.synthesize(_LOG)
    _second = _calls["n"]
finally:
    EnumerativeSynthesizer._score = _real_score
ok("memo_eliminates_the_repeat_search", _first > 0 and _second == 0,
   f"{_first} scorings cold, {_second} warm")

# 4. The key carries the inputs the fit depends on, so changing one INVALIDATES.
#    A banned rule must still be banned even though the samples are identical --
#    keying on len(banned) would let one demotion be swapped for another.
_ban_free = syn(MASK).synthesize(MOVER)
_banned_op = None if _ban_free is None else _ban_free.spec[0]["op"]
_bs = syn(MASK)
_bs.synthesize(MOVER)                     # warm the memo with the unbanned fit
if _banned_op is not None:
    import json as _json
    _key = _json.dumps({"action": "ACTION2", "op": _banned_op,
                        "args": _ban_free.spec[0]["args"]}, sort_keys=True)
    _after = _bs.synthesize(MOVER, banned={_key})
    _after_op = None if _after is None else _after.spec[0]["op"]
    ok("memo_respects_a_changed_banned_set", _after_op != _banned_op,
       f"still returned {_after_op} after it was banned")
    # ...and a DIFFERENT ban of the same size must not reuse that entry.
    _other = _bs.synthesize(MOVER, banned={"some other rule entirely"})
    ok("memo_key_is_the_banned_set_not_its_size",
       _other is not None and _other.spec[0]["op"] == _banned_op,
       f"a same-size but different ban returned {None if _other is None else _other.spec[0]['op']}")

# 5. Same for the mask: masked and raw are different questions. StateGraph
#    rebuilds the mask mid-run, so this flip happens in live play. The bar is
#    exact agreement with a cold raw synthesizer, not merely "something changed"
#    -- a stale masked entry would still be served for the raw question.
_ms = syn(MASK)
_ms.synthesize(_LOG)                      # warm under the mask
_ms.mask_fn = None
_after_flip = _ms.synthesize(_LOG)
_cold_raw = syn(None).synthesize(_LOG)
_fs = None if _after_flip is None else sorted(map(repr, _after_flip.spec))
_cs2 = None if _cold_raw is None else sorted(map(repr, _cold_raw.spec))
check("memo_key_includes_the_mask", _fs, _cs2)

# 6. The cache is bounded -- a 1500-action game must not grow it without limit.
_bs2 = syn(MASK)
for _i in range(2, len(_LOG) + 1):
    _bs2.synthesize(_LOG[:_i])
ok("memo_is_bounded", len(_bs2._fit_memo) <= FIT_MEMO_MAX,
   f"{len(_bs2._fit_memo)} entries, cap {FIT_MEMO_MAX}")

print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
